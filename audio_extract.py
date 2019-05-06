import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from logmmse import logmmse_from_file


def align_and_add(res, x1, x2):
    print(x1.shape, x2.shape)

    q = [np.sum(x1[i:] * x2[:-i]) for i in range(1, 250)]
    tmax = np.argmax(q) + 1
    # print(q[:10])
    # print(tmax)
    res[tmax:] += x2[:-tmax]


def combine(arrays, base=0, filt=None):
    assert arrays.ndim == 2
    res = np.zeros_like(arrays[0])
    for i in range(len(arrays)):
        x1 = arrays[base]
        x2 = arrays[i]
        if filt is not None:
            x1 = filt(x1)
            x2 = filt(x2)
        align_and_add(res, x1, x2)
    return res


def reconstruct_audio(signal_filename, output_filename, low_cutoff, high_cutoff):
    data = np.load(signal_filename)
    arr = data['arr_0'][:, ::2, 1:].astype(np.float64)
    arr = arr.reshape(-1, arr.shape[2])

    filter = lambda x: signal.sosfilt(signal.butter(11, (low_cutoff - 50,
                                                         high_cutoff + 50),
                                                    'bandpass', fs=2200,
                                                    output='sos'), x)
    combined_signal = combine(arr, 4, filter)

    sos = signal.butter(11, (low_cutoff, high_cutoff), 'bandpass', fs=2200, output='sos')
    filtered = signal.sosfilt(sos, combined_signal)
    wav_res = np.clip((32767 * filtered / np.max(filtered)).astype('i2'), -32767, 32767)
    wavfile.write(output_filename, 2200, wav_res)

    denoised_filename = output_filename[:output_filename.rindex('.')] + '-denoised.wav'
    output = logmmse_from_file(output_filename)
    wavfile.write(denoised_filename, 2200, output)


def main():
    reconstruct_audio("reconstructed-signals/chips2-mary.npz",
                      "reconstructed-audio/chips2-mary.wav", 150, 550)
    # reconstruct_audio("reconstructed-signals/plant-mary.npz",
    #                   "reconstructed-audio/plant-mary.wav", 150, 550)
    # reconstruct_audio("reconstructed-signals/chips1-mary-voice.npz",
    #                   "reconstructed-audio/chips1-mary-voice.wav", 180, 1000)


if __name__ == "__main__":
    main()
