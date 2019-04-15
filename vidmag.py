import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import signal
import gc


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

    def filter(self, data):
        if not self._initialized:
            self.low_pass1 = data
            self.low_pass2 = data
            self._initialized = True
        else:
            self.low_pass1 = (1 - self.high) * self.low_pass1 + self.high * data
            self.low_pass2 = (1 - self.low) * self.low_pass2 + self.low * data
        return self.low_pass1 - self.low_pass2


def save_video(video_tensor):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    [height, width] = video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter("out.mp4", fourcc, 30, (width, height), 1)
    for i in range(0, video_tensor.shape[0]):
        image = cv2.cvtColor(np.clip(video_tensor[i], 0, 255).astype(np.uint8),
                             cv2.COLOR_LAB2BGR)
        writer.write(cv2.convertScaleAbs(image))
    writer.release()


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
        if self.online:
            print("Plotting")
            cv2.imshow("Output", image)
        else:
            # image = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8),
            #                      cv2.COLOR_LAB2BGR)
            self.writer.write(cv2.convertScaleAbs(image))

    def release(self):
        self.writer.release()


def amplify_video(video_fname, max_frames=-1, levels=5):
    assert levels > 0

    fps, frame_count, width, height = get_video_details(video_fname)
    filters = [IdealTemporalFilter(30 / 60 / fps, 120 / 60 / fps) for _ in range(levels)]

    # Setup writer
    vid_writer = VideoWriter("out2.mp4", width, height)
    # vid_writer = VideoWriter("out2.mp4", width, height)

    counter = 0
    for frame in load_video(video_fname, max_frames):
        alpha = 15
        exaggeration_factor = 2

        # Spatial filtering
        lap_pyramid = get_laplacian_pyramid(frame.astype('float64'), levels)

        lambda_c = 16
        delta = lambda_c / 8.0 / (1.0 + alpha)

        lambda_ = ((lap_pyramid[0].shape[0] ** 2 +
                    lap_pyramid[0].shape[1] ** 2) **
                   0.5) / 3
        filtered = [np.empty(lap_pyramid[level].shape, dtype='float64')
                    for level in range(levels)]
        amplified = [np.empty(lap_pyramid[level].shape, dtype='float64')
                     for level in range(levels)]
        # print(amplified[0].dtype)
        # Temporal filtering

        for level in range(levels - 1, -1, -1):
            filtered[level][:] = filters[level].filter(lap_pyramid[level])

            curr_alpha = max((lambda_ / delta / 8) - 1, 0)
            curr_alpha *= exaggeration_factor
            # print(curr_alpha, alpha)
            # print("Curr_alpha", curr_alpha)
            amplified[level] = min(alpha, curr_alpha) * filtered[level]

            # Attenuation
            amplified[level][:, :, 1] *= 0.1
            amplified[level][:, :, 2] *= 0.1

            if level == 0 or level == levels - 1:
                amplified[level] *= 0
            # amplified[level] += lap_pyramid[level]

            lambda_ /= 2

        # Reconstruction
        res_frame = reconstruct([amplified[level] for level in range(levels)])

        # if counter == 10:
        #     # skimage.io.imshow(res_frame)
        #     for i in range(6):
        #         plt.subplot(2, 3, i + 1)
        #         yiq_plot(amplified[i] / np.max(amplified[i]))
        #         print("level: ", i + 1, " max: ", np.max(amplified[i]))
        #     plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        res_frame += frame
        vid_writer.write(res_frame)
        counter += 1

        print(f"Done processing {counter} frames")
        gc.collect()
    vid_writer.release()


def lab_plot(img):
    skimage.io.imshow(
        cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


def yiq_plot(img):
    img = yiq2rgb(img)
    # print(np.min(img, axis=1), np.max(img, axis=1))
    skimage.io.imshow(np.clip(yiq2rgb(img), 0, 1))


if __name__ == "__main__":
    amplify_video("baby.mp4", max_frames=-1, levels=7)
