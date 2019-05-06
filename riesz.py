import numpy as np
from itertools import islice
import skimage
import time
import yappi
from common import VideoWriter, IdealTemporalFilter, get_video_details, load_video
from common import get_phase_difference_and_amplitude, get_riesz_pyramid, \
    reconstruct, amplitude_weighted_blur, phase_shift, load_riesz_pyramid

PROFILE = True


def amplify_video(fname, max_frames, low_cutoff, high_cutoff,
                  levels=3, grayscale=True, scale=0.25,
                  amplification=1.0):
    fps, frame_count, width, height = get_video_details(fname)
    scale_factor = scale

    riesz_gen = load_riesz_pyramid(fname, levels, grayscale, scale_factor, max_frames)

    pseudo_frame = skimage.transform.rescale(np.zeros((height, width)),
                                             scale_factor)
    output_shape = pseudo_frame.shape

    writer = VideoWriter("out-webcam.mp4",
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

    signals1 = [[None] for _ in range(levels)]
    signals2 = [[None] for _ in range(levels)]
    amps = [[None] for _ in range(levels)]

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
                signals1[level].append(pd_cosf2)
                signals2[level].append(pd_sinf2)
                amps[level].append(amplitude)
                mag_pyr[level] = phase_shift(a_curr[level], b_curr[level], c_curr[level],
                                             pd_cosf2, pd_sinf2)

        end_time = time.time()
        a_prev, b_prev, c_prev = a_curr, b_curr, c_curr
        if i == 0:
            writer.write(frame)
        else:
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

    np.savez('signals1.npz', signals1, signals2, amps)

    if PROFILE:
        yappi.get_func_stats().print_all()
        yappi.get_thread_stats().print_all()


def save_res(fname, outfname, max_frames, low_cutoff, high_cutoff,
             levels=3, grayscale=True, scale=0.25):
    fps, frame_count, width, height = get_video_details(fname)
    scale_factor = scale

    riesz_gen = load_riesz_pyramid(fname, levels, grayscale, scale_factor, max_frames)

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

    signal_orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    signals = [[[None] for _ in range(len(signal_orientations))]
               for _ in range(levels)]

    for i, (a_curr, b_curr, c_curr, residual, frame, raw_frame) in enumerate(riesz_gen):
        start_time = time.time()

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

                pd_cosf2 = pd_cosf
                pd_sinf2 = pd_sinf

                for idx, orientation in enumerate(signal_orientations):
                    signals[level][idx].append(
                        np.sum(
                            (pd_cosf2 * np.cos(orientation) +
                             pd_sinf2 * np.sin(orientation)) * amplitude ** 2
                        ))
                # amps[level].append(np.sum(amplitude))

        end_time = time.time()
        a_prev, b_prev, c_prev = a_curr, b_curr, c_curr
        print(f"{i + 1} frames processed. "
              f"Frame {i + 1} required {end_time - start_time} seconds")

    np.savez(outfname, signals)

    if PROFILE:
        yappi.get_func_stats().print_all()
        yappi.get_thread_stats().print_all()


def main():
    # amplify_video("Chips2-2200Hz-Mary_MIDI-input.avi", 17600,
    #               50 * 30 / 2200, 1500 * 30 / 2200,
    #               scale=0.25,
    #               amplification=1.0)
    # save_res("Chips2-2200Hz-Mary_MIDI-input.avi", None,
    #          50 * 30 / 2200, 1500 * 30 / 2200,
    #          scale=0.25)
    # amplify_video("baby.mp4", None, 30 / 60, 120 / 60)
    save_res("Plant-2200Hz-Mary_MIDI-input.avi", "plant-mary.npz",
             None,
             50 * 30 / 2200, 1500 * 30 / 2200,
             scale=0.25)
    # save_res("Chips1-2200Hz-Mary_Had-input.avi", "chips1-mary-voice.npz",
    #          None,
    #          50 * 30 / 2200, 1500 * 30 / 2200,
    #          scale=0.25)


if __name__ == "__main__":
    main()
