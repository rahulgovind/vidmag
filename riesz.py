import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy
from itertools import islice
import skimage
import gc
import time
import yappi
import os
import re

GRAYSCALE = True
PROFILE = True


def get_gaussian_pyramid(image, levels=5):
    # Ref: https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html
    pyramid = [image]
    for i in range(levels):
        width, height = pyramid[i].shape[1], pyramid[i].shape[0]
        pyramid.append(cv2.pyrDown(pyramid[i], dstsize=(width // 2, height // 2)))
    return pyramid


def get_laplacian_pyramid(image, levels=5):
    # Ref: https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html
    gaussian_pyramid = get_gaussian_pyramid(image, levels - 1)
    res = []

    for i in range(levels - 1):
        width, height = gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]
        res.append(gaussian_pyramid[i] -
                   cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=(width, height)))
    res.append(gaussian_pyramid[levels - 1])
    return res


def get_riesz_pyramid(image, levels=5):
    # Get a grayscale image
    assert image.ndim == 2

    kernel_x = np.array([[0.0, 0.0, 0.0],
                         [0.5, 0.0, -0.5],
                         [0.0, 0.0, 0.0]])
    kernel_y = np.array([[0.0, 0.5, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, -0.5, 0.0]])
    laplacian_pyramid = get_laplacian_pyramid(image, levels + 1)

    # Ignore the last level which is just the low-pass residual
    riesz_x = [scipy.ndimage.filters.convolve(laplacian_pyramid[level], kernel_x)
               for level in range(levels)]
    riesz_y = [scipy.ndimage.filters.convolve(laplacian_pyramid[level], kernel_y)
               for level in range(levels)]

    return laplacian_pyramid, riesz_x, riesz_y, laplacian_pyramid[-1]


def get_phase_difference_and_amplitude(a_prev, b_prev, c_prev, a_curr, b_curr, c_curr):
    # Quaternion representation here = a + b * i + c * j + 0 * k
    # We want to find phase difference between q_prev and q_curr
    # So we calculate q_curr / q_prev = q_curr * conjugate(q_prev) / | q_prev | ^2
    # Phase difference is invariant to scale so we just find phase difference
    # using q_curr * conjugate(q_prev) = p
    p_real = (a_prev * a_curr + b_prev * b_curr + c_prev * c_curr)
    p_x = -a_curr * b_prev + a_prev * b_curr
    p_y = -a_curr * c_prev + a_prev * c_curr
    p_amplitude = (p_real ** 2 + p_x ** 2 + p_y ** 2) ** 0.5 + 1e-6
    phase_difference = np.arccos(p_real / p_amplitude)
    # print("Phase difference: ", phase_difference)

    amp = (p_x ** 2 + p_y ** 2) ** 0.5
    cos_orientation = p_x / amp
    sin_orientation = p_y / amp

    phase_difference_cos = phase_difference * cos_orientation
    phase_difference_sin = phase_difference * sin_orientation

    amplitude = p_amplitude ** 0.5 + 1e-6
    return phase_difference_cos, phase_difference_sin, amplitude


def amplitude_weighted_blur(filtered_phase, amplitude):
    denominator = scipy.ndimage.filters.gaussian_filter(amplitude, 2)
    numerator = scipy.ndimage.filters.gaussian_filter(filtered_phase * amplitude, 2)
    return numerator / denominator


def load_video(fname, max_frames=-1):
    """
    :param fname: Video filename
    :param max_frames: Maximum number of frames to include
    :return: Iterator that iterates over each frame in video
    """
    capture = cv2.VideoCapture(fname)

    i = 0
    while capture.isOpened():
        retval, frame = capture.read()
        if retval:
            if GRAYSCALE:
                yield (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)) / 255
            else:
                yield bgr2yiq(frame.astype(np.float64) / 255)
            # yield cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            i += 1
        else:
            break
        if max_frames != -1 and i >= max_frames:
            break


class IdealTemporalFilter(object):
    def __init__(self, low, high):
        self.low_pass1 = None
        self.low_pass2 = None
        self._initialized = False
        self.low = low
        self.high = high

    def initialize(self, low_pass1, low_pass2):
        if self._initialized:
            raise ValueError("Already initialized")
        assert low_pass1.shape == low_pass2.shape

        self._initialized = True
        self.low_pass1 = low_pass1
        self.low_pass2 = low_pass2

    def filter(self, data):
        if not self._initialized:
            self.low_pass1 = data
            self.low_pass2 = data
            self._initialized = True
        else:
            self.low_pass1 = (1 - self.high) * self.low_pass1 + self.high * data
            self.low_pass2 = (1 - self.low) * self.low_pass2 + self.low * data
        return self.low_pass1 - self.low_pass2


def phase_shift(a, b, c, phase_cos, phase_sin):
    phase = (phase_cos ** 2 + phase_sin ** 2) ** 0.5
    shift_a = np.cos(phase)
    shift_b = phase_cos / phase * np.sin(phase)
    shift_c = phase_sin / phase * np.sin(phase)
    return a * shift_a - b * shift_b - c * shift_c


def get_video_details(fname):
    """
    :param fname: Video filename
    :return: FPS, Number of frames, Width, Height
    """
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    print(f"Frame count: {frame_count}, FPS: {fps}\n"
          f"Width: {width}, Height: {height}")

    return fps, frame_count, width, height


def reconstruct(laplacian_pyramid):
    res = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        width, height = laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]
        res = cv2.pyrUp(res, dstsize=(width, height))
        res += laplacian_pyramid[i]

    return res


class VideoWriter(object):
    def __init__(self, out_filename, width, height):
        self.width = width
        self.height = height

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.writer = cv2.VideoWriter(out_filename, fourcc, 30, (width, height),
                                      not GRAYSCALE)

    def write(self, frame):
        if GRAYSCALE:
            image = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        else:
            image = (np.clip(yiq2bgr(frame), 0, 1) * 255).astype(np.uint8)
        self.writer.write(cv2.convertScaleAbs(image))

    def release(self):
        self.writer.release()


def bgr2yiq(image):
    T = np.array([[0.114, 0.587, 0.299],
                  [-0.322, -0.274, 0.596],
                  [0.312, -0.523, 0.211]])
    return (T @ image.reshape(-1, 3).T).T.reshape(image.shape)


def yiq2bgr(image):
    T = np.array([[1, -1.106, 1.703],
                  [1, -0.272, -0.647],
                  [1, 0.956, 0.619]])
    return (T @ image.reshape(-1, 3).T).T.reshape(image.shape)


imshow = skimage.io.imshow


def save_riesz_pyramid(fname, max_frames, levels=3):
    fps, frame_count, width, height = get_video_details(fname)
    if max_frames is None:
        max_frames = frame_count

    frame_gen = load_video(fname)
    scale_factor = 0.1
    writer = VideoWriter("_ignore.mp4",
                         int(width * scale_factor),
                         int(height * scale_factor))

    pseudo_frame = skimage.transform.rescale(np.zeros((height, width)),
                                             scale_factor)
    output_shape = pseudo_frame.shape

    pyr_shapes = [output_shape]
    print("output shape: ", output_shape)
    for i in range(1, levels + 1):
        pyr_shapes.append((pyr_shapes[-1][0] // 2,
                           pyr_shapes[-1][1] // 2))

    fname0 = re.findall("([a-zA-Z_-]+).mp4", fname)[0]
    output_dir = f"pre-{fname0}_{max_frames}_{levels}_{GRAYSCALE}"
    az = [np.memmap(os.path.join(output_dir, f'az-{i}.npz'), dtype='float32',
                    mode='w+', shape=(max_frames,) + pyr_shapes[i])
          for i in range(levels)]
    bz = [np.memmap(os.path.join(output_dir, f'bz-{i}.npz'), dtype='float32',
                    mode='w+', shape=(max_frames,) + pyr_shapes[i])
          for i in range(levels)]
    cz = [np.memmap(os.path.join(output_dir, f'cz-{i}.npz'), dtype='float32',
                    mode='w+', shape=(max_frames,) + pyr_shapes[i])
          for i in range(levels)]
    rz = np.memmap(os.path.join(output_dir, f'rz.npz'), dtype='float32',
                   mode='w+', shape=(max_frames,) + pyr_shapes[levels])

    print("Starting")
    if PROFILE:
        yappi.start()

    for i, color_frame in enumerate(islice(frame_gen, max_frames)):
        start_time = time.time()
        color_frame = skimage.transform.rescale(color_frame, scale_factor)

        print(color_frame.shape)
        if GRAYSCALE:
            frame = color_frame
        else:
            frame = color_frame[:, :, 0]
        a_curr, b_curr, c_curr, residual = get_riesz_pyramid(frame, levels)
        for level in range(levels):
            az[level][i] = a_curr[level]
            bz[level][i] = b_curr[level]
            cz[level][i] = c_curr[level]
        rz[i] = residual

        end_time = time.time()
        print(f"{i + 1} frames processed. "
              f"Frame {i + 1} required {end_time - start_time} seconds")

    for level in range(levels):
        az[level].flush()
        bz[level].flush()
        cz[level].flush()
    rz.flush()

    if PROFILE:
        yappi.get_func_stats().print_all()
        yappi.get_thread_stats().print_all()


def amplify_video(fname, max_frames, low_cutoff, high_cutoff, levels=3):
    fps, frame_count, width, height = get_video_details(fname)
    frame_gen = load_video(fname)
    scale_factor = 0.15

    pseudo_frame = skimage.transform.rescale(np.zeros((height, width)),
                                             scale_factor)
    output_shape = pseudo_frame.shape

    writer = VideoWriter("out-webcam.mp4",
                         output_shape[1],
                         output_shape[0])

    print("Gray scale: ", GRAYSCALE)
    # frames = [frame for frame in islice(frame_gen, max_frames)]

    cos_filters = [IdealTemporalFilter(low_cutoff / fps, high_cutoff / fps)
                   for _ in range(levels)]
    sin_filters = [IdealTemporalFilter(low_cutoff / fps, high_cutoff / fps)
                   for _ in range(levels)]

    a_prev, b_prev, c_prev = None, None, None

    # Just to keep my IDE happy
    none = np.zeros(1)

    pd_cos, pd_sin = [none for _ in range(levels)], [none for _ in range(levels)]

    for i in range(10):
        next(frame_gen)

    if PROFILE:
        yappi.start()

    for i, color_frame in enumerate(islice(frame_gen, max_frames)):
        start_time = time.time()
        color_frame = skimage.transform.rescale(color_frame, scale_factor)
        print(color_frame.shape)
        if GRAYSCALE:
            frame = color_frame
        else:
            frame = color_frame[:, :, 0]

        a_curr, b_curr, c_curr, residual = get_riesz_pyramid(frame, levels)
        mag_pyr = [none] * (levels + 1)

        for level in range(levels):
            if i == 0:
                cos_filters[level].initialize(np.zeros_like(a_curr[level]),
                                              np.zeros_like(a_curr[level]))
                sin_filters[level].initialize(np.zeros_like(a_curr[level]),
                                              np.zeros_like(a_curr[level]))
                pd_cos[level] = np.zeros_like(a_curr[level])
                pd_sin[level] = np.zeros_like(a_curr[level])
            else:
                pd_cos_here, pd_sin_here, amplitude = get_phase_difference_and_amplitude(
                    a_prev[level], b_prev[level], c_prev[level],
                    a_curr[level], b_curr[level], c_curr[level]
                )

                pd_cos[level] = pd_cos[level] + pd_cos_here
                pd_sin[level] = pd_sin[level] + pd_sin_here

                pd_cosf = cos_filters[level].filter(pd_cos[level])
                pd_sinf = sin_filters[level].filter(pd_sin[level])

                pd_cosf = amplitude_weighted_blur(pd_cosf,
                                                  amplitude)
                pd_sinf = amplitude_weighted_blur(pd_sinf,
                                                  amplitude)

                pd_cosf2 = pd_cosf * 50
                pd_sinf2 = pd_sinf * 50

                mag_pyr[level] = phase_shift(a_curr[level], b_curr[level], c_curr[level],
                                             pd_cosf2, pd_sinf2)

        a_prev, b_prev, c_prev = a_curr, b_curr, c_curr
        if i == 0:
            writer.write(frame)
        else:
            mag_pyr[levels] = residual
            result = np.clip(reconstruct([np.nan_to_num(mag_pyr[level])
                                          for level in range(levels + 1)]), 0, 1)
            if GRAYSCALE:
                writer.write(result)
            else:
                color_result = np.dstack([result, color_frame[:, :, 1],
                                          color_frame[:, :, 2]])
                writer.write(color_result)
        # gc.collect()
        end_time = time.time()
        print(f"{i + 1} frames processed. "
              f"Frame {i + 1} required {end_time - start_time} seconds")
    writer.release()
    if PROFILE:
        yappi.get_func_stats().print_all()
        yappi.get_thread_stats().print_all()


def main():
    amplify_video("Chips2-2200Hz-Mary_MIDI-input.avi", None,
                  250 * 30 / 2200, 300 * 30 / 2200)
    # save_riesz_pyramid("baby.mp4", None, 30 / 60, 120 / 60)


if __name__ == "__main__":
    main()
