import numpy as np
import scipy
import cv2
import skimage
from itertools import islice
import re
import os
from collections import deque


def load_video(fname, max_frames=-1, grayscale=True):
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
            if grayscale:
                yield (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)) / 255
            else:
                yield bgr2yiq(frame.astype(np.float64) / 255)
            # yield cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            i += 1
        else:
            break
        if max_frames != -1 and i >= max_frames:
            break


def get_video_details(fname):
    """
    :param fname: Video filename
    :return: FPS, Number of frames, Width, Height
    """
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    print(f"Frame count: {frame_count}, FPS: {fps}\n"
          f"Width: {width}, Height: {height}")

    return fps, frame_count, width, height


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


def reconstruct(laplacian_pyramid):
    res = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        width, height = laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]
        res = cv2.pyrUp(res, dstsize=(width, height))
        res += laplacian_pyramid[i]

    return res


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


class IIRFilter(object):
    def __init__(self, b, a):
        self.n = len(a)
        self.m = len(b)

        self.x_ = deque([None] * self.m)
        self.y_ = deque([None] * self.n)

        self.a = a
        self.b = b

    def filter(self, data):
        self.x_.pop()
        self.x_.appendleft(np.array(data))

        res = np.zeros_like(data)
        for i in range(self.m):
            if self.x_[i] is None:
                continue
            res += self.b[i] * self.x_[i]

        for i in range(1, self.n):
            if self.y_[i] is None:
                continue
            res -= self.a[i] * self.y_[i]

        res /= self.a[0]

        self.y_.pop()
        self.y_[0] = np.array(res)
        self.y_.appendleft(None)

        return res


def phase_shift(a, b, c, phase_cos, phase_sin):
    phase = (phase_cos ** 2 + phase_sin ** 2) ** 0.5
    shift_a = np.cos(phase)
    shift_b = phase_cos / phase * np.sin(phase)
    shift_c = phase_sin / phase * np.sin(phase)
    return a * shift_a - b * shift_b - c * shift_c


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


class VideoWriter(object):
    def __init__(self, out_filename, width, height, grayscale):
        self.width = width
        self.height = height

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.writer = cv2.VideoWriter(out_filename, fourcc, 30, (width, height),
                                      not grayscale)
        self.grayscale = grayscale

    def write(self, frame):
        if self.grayscale:
            image = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        else:
            image = (np.clip(yiq2bgr(frame), 0, 1) * 255).astype(np.uint8)
        self.writer.write(cv2.convertScaleAbs(image))

    def release(self):
        self.writer.release()


imshow = skimage.io.imshow


def _precompute_riesz(fname, levels, grayscale, scale, max_frames):
    fps, frame_count, width, height = get_video_details(fname)
    if max_frames is None:
        max_frames = frame_count

    pseudo_frame = skimage.transform.rescale(np.zeros((height, width)), scale)
    output_shape = pseudo_frame.shape

    pyr_shapes = [output_shape]
    print("output shape: ", output_shape)
    for i in range(1, levels + 1):
        pyr_shapes.append((pyr_shapes[-1][0] // 2,
                           pyr_shapes[-1][1] // 2))
    print(pyr_shapes)
    fname0 = re.findall("([a-zA-Z_-]+).(mp4|avi)", fname)[0][0]
    if not os.path.isdir("_precomputed"):
        os.mkdir("_precomputed")
    output_dir = "_precomputed/" + "_".join(str(_) for _ in [fname0, levels,
                                                             grayscale, scale,
                                                             max_frames])

    if os.path.isdir(output_dir):
        mode = "r"
    else:
        os.mkdir(output_dir)
        mode = "w+"

    az = [np.memmap(os.path.join(output_dir, f'az-{i}.npz'), dtype='float32',
                    mode=mode, shape=(max_frames,) + pyr_shapes[i])
          for i in range(levels)]
    bz = [np.memmap(os.path.join(output_dir, f'bz-{i}.npz'), dtype='float32',
                    mode=mode, shape=(max_frames,) + pyr_shapes[i])
          for i in range(levels)]
    cz = [np.memmap(os.path.join(output_dir, f'cz-{i}.npz'), dtype='float32',
                    mode=mode, shape=(max_frames,) + pyr_shapes[i])
          for i in range(levels)]
    rz = np.memmap(os.path.join(output_dir, f'rz.npz'), dtype='float32',
                   mode=mode, shape=(max_frames,) + pyr_shapes[levels])
    fz = np.memmap(os.path.join(output_dir, f'fz.npz'), dtype='float32',
                   mode=mode, shape=(max_frames,) + output_shape)
    if grayscale:
        input_shape = output_shape
    else:
        input_shape = output_shape + (3,)

    print("Output shape: ", output_shape)
    print("Input shape: ", input_shape)
    ifz = np.memmap(os.path.join(output_dir, f'ifz.npz'), dtype='float32',
                    mode=mode, shape=(max_frames,) + input_shape)

    if mode == "w+":
        print("Precomputing riesz")
        gen = load_riesz_pyramid(fname, levels, grayscale, scale, max_frames, False)

        for i, (a_curr, b_curr, c_curr, residual, frame, input_frame) in enumerate(gen):
            print(f"Precompute: Step {i + 1} / {max_frames}")
            for level in range(levels):
                az[level][i] = a_curr[level]
                bz[level][i] = b_curr[level]
                cz[level][i] = c_curr[level]
            rz[i] = residual
            fz[i] = frame
            ifz[i] = input_frame

        for level in range(levels):
            for ary in [az[level], bz[level], cz[level]]:
                ary.flush()
        for ary in [rz, fz, ifz]:
            ary.flush()
    return az, bz, cz, rz, fz, ifz


def load_riesz_pyramid(fname, levels, grayscale, scale, max_frames, pregen=True):
    if not pregen:
        frame_gen = load_video(fname, grayscale=grayscale)

        for i, input_frame in enumerate(islice(frame_gen, max_frames)):
            input_frame = skimage.transform.rescale(input_frame, scale)
            if grayscale:
                frame = input_frame
            else:
                frame = input_frame[:, :, 0]
            print(input_frame.shape)
            a_curr, b_curr, c_curr, residual = get_riesz_pyramid(frame, levels)

            yield a_curr, b_curr, c_curr, residual, frame, input_frame
    else:
        az, bz, cz, rz, fz, ifz = _precompute_riesz(fname, levels,
                                                    grayscale,
                                                    scale, max_frames)
        chunk_size = 512
        for i in range(0, rz.shape[0], chunk_size):
            j = 0
            az_ = [az[level][i:i + chunk_size] for level in range(levels)]
            bz_ = [bz[level][i:i + chunk_size] for level in range(levels)]
            cz_ = [cz[level][i:i + chunk_size] for level in range(levels)]
            rz_ = rz[i:i + chunk_size]
            fz_ = fz[i:i + chunk_size]
            ifz_ = ifz[i:i + chunk_size]
            while j < chunk_size and i + j < rz.shape[0]:
                yield ([az_[level][j] for level in range(levels)],
                       [bz_[level][j] for level in range(levels)],
                       [cz_[level][j] for level in range(levels)],
                       rz_[j], fz_[j], ifz_[j])
                j += 1
