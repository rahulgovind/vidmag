import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import signal
import gc
import scipy


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
            yield bgr2yiq(frame.astype(np.float64)) / 255
            # yield cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            i += 1
        else:
            break
        if max_frames != -1 and i >= max_frames:
            break


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

    return laplacian_pyramid[:-1], riesz_x, riesz_y, laplacian_pyramid[-1]


def reconstruct(laplacian_pyramid):
    res = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        width, height = laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]
        res = cv2.pyrUp(res, dstsize=(width, height))
        res += laplacian_pyramid[i]

    return res


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


def yiq2rgb(image):
    T = np.array([[1, 0.956, 0.619],
                  [1, -0.272, -0.647],
                  [1, -1.106, 1.703]])
    return (T @ image.reshape(-1, 3).T).T.reshape(image.shape)


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


class VideoWriter(object):
    def __init__(self, out_filename, width, height):
        self.width = width
        self.height = height

        if out_filename == 0:
            self.online = True
            return
        self.online = False
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.writer = cv2.VideoWriter(out_filename, fourcc, 30, (width, height), 1)

    def write(self, frame):
        image = (np.clip(yiq2bgr(frame), 0, 1) * 255).astype(np.uint8)
        self.writer.write(cv2.convertScaleAbs(image))

    def release(self):
        self.writer.release()


def pdiff_amp(a_prev, b_prev, c_prev, a_curr, b_curr, c_curr):
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
    cos_orientation = p_x / (p_x ** 2 + p_y ** 2) ** 0.5
    sin_orientation = p_y / (p_x ** 2 + p_y ** 2) ** 0.5

    phase_difference_cos = phase_difference * cos_orientation
    phase_difference_sin = phase_difference * sin_orientation

    amplitude = p_amplitude ** 0.5 + 1e-6
    return phase_difference_cos, phase_difference_sin, amplitude


def amplitude_weighted_blur(filtered_phase, amplitude):
    denominator = scipy.ndimage.filters.gaussian_filter(amplitude, 2)
    numerator = scipy.ndimage.filters.gaussian_filter(filtered_phase * amplitude, 2)
    return numerator / denominator


def phase_shift(a, b, c, phase_cos, phase_sin):
    phase = (phase_cos ** 2 + phase_sin ** 2) ** 0.5
    shift_a = np.cos(phase)
    shift_b = phase_cos / phase * np.sin(phase)
    shift_c = phase_sin / phase * np.sin(phase)
    return a * shift_a - b * shift_b - c * shift_c


def amplify_video(video_fname, max_frames=-1, levels=5):
    assert levels > 0

    fps, frame_count, width, height = get_video_details(video_fname)
    filters = [IdealTemporalFilter(30 / 60 / fps, 120 / 60 / fps) for _ in range(levels)]

    # Setup writer
    vid_writer = VideoWriter("out3.mp4", width, height)
    # vid_writer = VideoWriter("out2.mp4", width, height)

    counter = 0
    empty = np.empty(1)

    # Initialization
    # Not needed in this format but my ide likes it
    a_prev, b_prev, c_prev = [empty] * levels, [empty] * levels, [empty] * levels
    pcos, psin = [empty] * levels, [empty] * levels
    pcos_filters = [IdealTemporalFilter(30 / 60 / fps, 120 / 60 / fps)
                    for _ in range(levels)]
    psin_filters = [IdealTemporalFilter(30 / 60 / fps, 120 / 60 / fps)
                    for _ in range(levels)]

    for frame_num, color_frame in enumerate(load_video(video_fname, max_frames)):
        frame = color_frame[:, :, 0]

        alpha = 1

        # Spatial filtering
        a, b, c, residual = get_riesz_pyramid(frame)

        if frame_num == 0:
            a_prev, b_prev, c_prev = a, b, c
            for level in range(levels):
                pcos[level] = np.zeros_like(a[level])
                psin[level] = np.zeros_like(a[level])
                pcos_filters[level].initialize(np.zeros_like(a[level]),
                                               np.zeros_like(a[level]))
                psin_filters[level].initialize(np.zeros_like(a[level]),
                                               np.zeros_like(a[level]))
            vid_writer.write(color_frame)
            continue

        reconstruction_pyramid = []
        for level in range(levels):
            pcos_diff, psin_diff, amp = pdiff_amp(a_prev[level], b_prev[level],
                                                  c_prev[level], a[level],
                                                  b[level], c[level])
            pcos[level] += pcos_diff
            psin[level] += psin_diff

            pcosf = pcos_filters[level].filter(pcos[level])
            psinf = psin_filters[level].filter(psin[level])

            pcosf = amplitude_weighted_blur(pcosf, amp) * alpha
            psinf = amplitude_weighted_blur(psinf, amp) * alpha

            print("pcosf", pcosf.shape)
            print("psinf", psinf.shape)
            print("a[level]", a[level].shape)
            print("b[level]", b[level].shape)
            print("c[level]", c[level].shape)
            reconstruction_pyramid.append(phase_shift(a[level], b[level], c[level],
                                                      pcosf, psinf))
            a_prev, b_prev, c_prev = a, b, c
        reconstruction_pyramid.append(residual)
        reconstructed = np.nan_to_num(reconstruct(reconstruction_pyramid))
        yiq_plot(reconstructed)
        plt.show()
        new_frame = np.stack([reconstructed,
                              color_frame[:, :, 1],
                              color_frame[:, :, 2]],
                             axis=-1)
        print("New shape: ", new_frame.shape)
        vid_writer.write(np.stack([reconstructed,
                                   color_frame[:, :, 1],
                                   color_frame[:, :, 2]],
                                  axis=-1))
        counter += 1

        print(f"Done processing {counter} frames")
        gc.collect()
    vid_writer.release()


def yiq_plot(img):
    img = yiq2rgb(img)
    # print(np.min(img, axis=1), np.max(img, axis=1))
    skimage.io.imshow(np.clip(yiq2rgb(img), 0, 1))


if __name__ == "__main__":
    amplify_video("baby.mp4", max_frames=20, levels=5)
