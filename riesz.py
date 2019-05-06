import numpy as np
import skimage
import time
import yappi
from common import VideoWriter, IdealTemporalFilter, get_video_details, load_video
from common import get_phase_difference_and_amplitude, get_riesz_pyramid, \
    reconstruct, amplitude_weighted_blur, phase_shift, load_riesz_pyramid, \
    IIRFilter
from scipy import signal
import matplotlib.pyplot as plt

PROFILE = True


def amplify_video(fname, output_filename, max_frames, low_cutoff, high_cutoff,
                  levels=3, grayscale=True, scale=0.25,
                  amplification=1.0):
    fps, frame_count, width, height = get_video_details(fname)
    scale_factor = scale

    riesz_gen = load_riesz_pyramid(fname, levels, grayscale, scale_factor, max_frames)

    pseudo_frame = skimage.transform.rescale(np.zeros((height, width)),
                                             scale_factor)
    output_shape = pseudo_frame.shape

    writer = VideoWriter(output_filename,
                         output_shape[1],
                         output_shape[0],
                         grayscale)

    cos_filters = [IdealTemporalFilter(low_cutoff / fps, high_cutoff / fps)
                   for _ in range(levels)]
    sin_filters = [IdealTemporalFilter(low_cutoff / fps, high_cutoff / fps)
                   for _ in range(levels)]
    a_prev, b_prev, c_prev = None, None, None

    # Just to keep my IDE happy
    none = np.zeros(1)

    pd_cos, pd_sin = [none for _ in range(levels)], [none for _ in range(levels)]

    for i in range(10):
        next(riesz_gen)

    if PROFILE:
        yappi.start()


    for i, (a_curr, b_curr, c_curr, residual, frame, raw_frame) in enumerate(riesz_gen):
        start_time = time.time()
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

                pd_cosf2 = pd_cosf * amplification
                pd_sinf2 = pd_sinf * amplification

                mag_pyr[level] = phase_shift(a_curr[level], b_curr[level], c_curr[level],
                                             pd_cosf2, pd_sinf2)

        end_time = time.time()
        a_prev, b_prev, c_prev = a_curr, b_curr, c_curr
        if i == 0:
            writer.write(raw_frame)
        else:
            if i == 90:
                for level in range(1, levels - 1):
                    plt.subplot(levels - 2, 1, level)
                    skimage.io.imshow(a_curr[level])

                plt.savefig('images/a_curr.png', bbox_inches='tight')
                plt.clf()
                for level in range(1, levels - 1):
                    plt.subplot(levels - 2, 1, level)
                    skimage.io.imshow(b_curr[level])
                plt.savefig('images/b_curr.png', bbox_inches='tight')
                # plt.show()
                plt.clf()
                for level in range(1, levels - 1):
                    plt.subplot(levels - 2, 1, level)
                    skimage.io.imshow(c_curr[level])
                plt.savefig('images/c_curr.png', bbox_inches='tight')
                # plt.show()
                plt.clf()
            mag_pyr[levels] = residual
            result = np.clip(reconstruct([np.nan_to_num(mag_pyr[level])
                                          for level in range(levels + 1)]), 0, 1)
            if grayscale:
                writer.write(result)
            else:
                color_result = np.dstack([result, raw_frame[:, :, 1],
                                          raw_frame[:, :, 2]])
                writer.write(color_result)
        # gc.collect()
        print(f"{i + 1} frames processed. "
              f"Frame {i + 1} required {end_time - start_time} seconds")
    writer.release()

    if PROFILE:
        yappi.get_func_stats().print_all()
        yappi.get_thread_stats().print_all()


def main():
    amplify_video("baby.mp4",
                  "amplified-videos/phase-based/baby.mp4",
                  None, 30 / 60, 120 / 60,
                  amplification=10.0,
                  scale=1.00,
                  levels=5,
                  grayscale=False)


if __name__ == "__main__":
    main()
