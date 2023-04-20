import sklearn
import numpy as np

import librosa
import librosa.display
import matplotlib.pyplot as plt

# my import
import constants as const
from utils import features_computation


def visualize_features():
    # audio signal of every genre
    genres = const.FEATURES_VISUALIZATION_PATH

    # print simple characteristics
    print("\nPick one signal per genre: ")
    for key, value in genres.items():
        signal, sample_rate = librosa.load(value)
        print("{} -> {}".format(key, value))
        print(" - Signal: \033[92m{}\033[0m".format(signal))
        print(" - Signal Shape: \033[92m{}\033[0m".format(np.shape(signal)))
        print(" - Sample Rate (Khz): \033[92m{}\033[0m".format(sample_rate))
        print(" - Duration (s): \033[92m{}\033[0m\n".format(signal.shape[0] / sample_rate))

    # WAVEFORM
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16, 8.2))
    fig.suptitle("Waveform", fontsize=14)

    rows = 0
    columns = 0

    for key, value in genres.items():
        # load signal
        signal, sample_rate = librosa.load(value)
        # plot waveform
        librosa.display.waveshow(y=signal, sr=sample_rate, ax=ax[rows][columns])
        # set genre of signal as title
        ax[rows][columns].set_title(key)

        if columns == 1:
            rows += 1
            columns = 0
        else:
            columns += 1

    plt.tight_layout()
    plt.show()

    # SPECTRAL CENTROID
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16, 8.2))
    fig.suptitle("Spectral Centroid", fontsize=14)

    rows = 0
    columns = 0

    for key, value in genres.items():
        # load signal
        signal, sample_rate = librosa.load(value)
        # plot
        spectral_centroid = features_computation.compute_spectral_centroid(signal, sample_rate,
                                                                           const.NUM_FTT, const.HOP_LENGHT)[0]
        frames = range(len(spectral_centroid))
        t = librosa.frames_to_time(frames)
        librosa.display.waveshow(y=signal, sr=sample_rate, alpha=0.4, ax=ax[rows][columns])
        ax[rows][columns].plot(t, sklearn.preprocessing.minmax_scale(spectral_centroid, axis=0), color="red")

        # set genre of signal as title
        ax[rows][columns].set_title(key)

        if columns == 1:
            rows += 1
            columns = 0
        else:
            columns += 1

    plt.tight_layout()
    plt.show()

    # SPECTRAL BANDWIDTH
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16, 8.2))
    fig.suptitle("Spectral Bandwidth", fontsize=14)

    rows = 0
    columns = 0

    for key, value in genres.items():
        # load signal
        signal, sample_rate = librosa.load(value)
        # plot
        spectral_bandwidth = features_computation.compute_spectral_bandwidth(signal, sample_rate,
                                                                             const.NUM_FTT, const.HOP_LENGHT)[0]
        frames = range(len(spectral_bandwidth))
        t = librosa.frames_to_time(frames)
        librosa.display.waveshow(y=signal, sr=sample_rate, alpha=0.4, ax=ax[rows][columns])
        ax[rows][columns].plot(t, sklearn.preprocessing.minmax_scale(spectral_bandwidth, axis=0), color="red")

        # set genre of signal as title
        ax[rows][columns].set_title(key)

        if columns == 1:
            rows += 1
            columns = 0
        else:
            columns += 1

    plt.tight_layout()
    plt.show()

    # SPECTRAL ROLLOFF
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16, 8.2))
    fig.suptitle("Spectral Rolloff", fontsize=14)

    rows = 0
    columns = 0

    for key, value in genres.items():
        # load signal
        signal, sample_rate = librosa.load(value)
        # plot
        spectral_rolloff = features_computation.compute_spectral_rolloff(signal, sample_rate,
                                                                         const.NUM_FTT, const.HOP_LENGHT)[0]
        frames = range(len(spectral_rolloff))
        t = librosa.frames_to_time(frames)
        librosa.display.waveshow(y=signal, sr=sample_rate, alpha=0.4, ax=ax[rows][columns])
        ax[rows][columns].plot(t, sklearn.preprocessing.minmax_scale(spectral_rolloff, axis=0), color="red")

        # set genre of signal as title
        ax[rows][columns].set_title(key)

        if columns == 1:
            rows += 1
            columns = 0
        else:
            columns += 1

    plt.tight_layout()
    plt.show()

    # ZCR
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16, 8.2))
    fig.suptitle("Zero Crossing Rate", fontsize=14)

    rows = 0
    columns = 0

    for key, value in genres.items():
        # load signal
        signal, sample_rate = librosa.load(value)
        # plot
        zcr = librosa.feature.zero_crossing_rate(y=signal, hop_length=const.HOP_LENGHT)[0]
        ax[rows][columns].plot(zcr)
        # set genre of signal as title
        ax[rows][columns].set_title(key)

        if columns == 1:
            rows += 1
            columns = 0
        else:
            columns += 1

    plt.tight_layout()
    plt.show()

    # SPECTROGRAM
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16, 8.2))
    fig.suptitle("Spectrogram", fontsize=14)

    rows = 0
    columns = 0

    for key, value in genres.items():
        # load signal
        signal, sample_rate = librosa.load(value)
        # plot
        stft = librosa.core.stft(signal, hop_length=const.HOP_LENGHT, n_fft=const.NUM_FTT)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
        log_spec_plot = librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=const.HOP_LENGHT,
                                                 x_axis="time", y_axis="log", ax=ax[rows][columns])
        fig.colorbar(log_spec_plot, ax=ax[rows][columns])

        # set genre of signal as title
        ax[rows][columns].set_title(key)

        if columns == 1:
            rows += 1
            columns = 0
        else:
            columns += 1

    plt.tight_layout()
    plt.show()

    # CHROMAGRAM
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16, 8.2))
    fig.suptitle("Chromagram", fontsize=14)

    rows = 0
    columns = 0

    for key, value in genres.items():
        # load signal
        signal, sample_rate = librosa.load(value)
        # plot
        chromagram = features_computation.compute_chroma_stft(signal, sample_rate, const.NUM_FTT, const.HOP_LENGHT)
        plot_chromagram = librosa.display.specshow(chromagram, sr=sample_rate, hop_length=const.HOP_LENGHT,
                                                   x_axis="time", y_axis="chroma", cmap="coolwarm",
                                                   ax=ax[rows][columns])
        fig.colorbar(plot_chromagram, ax=ax[rows][columns])

        # set genre of signal as title
        ax[rows][columns].set_title(key)

        if columns == 1:
            rows += 1
            columns = 0
        else:
            columns += 1

    plt.tight_layout()
    plt.show()

    # MFCCs
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16, 8.2))
    fig.suptitle("MFCCs", fontsize=14)

    rows = 0
    columns = 0

    for key, value in genres.items():
        # load signal
        signal, sample_rate = librosa.load(value)
        # plot
        mfcc = features_computation.compute_mfcc(signal, sample_rate, const.NUM_MFCC, const.NUM_FTT, const.HOP_LENGHT)
        plot_mfcc = librosa.display.specshow(mfcc, sr=sample_rate, hop_length=const.HOP_LENGHT,
                                             x_axis="time", ax=ax[rows][columns])
        fig.colorbar(plot_mfcc, ax=ax[rows][columns])

        # set genre of signal as title
        ax[rows][columns].set_title(key)

        if columns == 1:
            rows += 1
            columns = 0
        else:
            columns += 1

    plt.tight_layout()
    plt.show()
