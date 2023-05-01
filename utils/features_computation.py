import sys
import numpy as np
import librosa.beat


def compute_energy(signal):
    # The sum of squares of the signal values,
    # normalized by the respective frame length.
    energy = np.sum(signal ** 2) / np.float64(len(signal))

    return energy


def compute_entropy_of_energy(signal, num_of_short_blocks=10):
    # The entropy of sub-frames of normalized energies.
    # It can be interpreted as a measure of abrupt changes.
    epsilon = sys.float_info.epsilon
    # frame energy
    frame_energy = np.sum(signal ** 2)
    frame_length = len(signal)

    sub_win_len = int(np.floor(frame_length / num_of_short_blocks))
    if frame_length != sub_win_len * num_of_short_blocks:
        signal = signal[0:sub_win_len * num_of_short_blocks]

    # sub-window of size: [num_of_short_blocks * L]
    sub_win = signal.reshape(sub_win_len, num_of_short_blocks, order="F").copy()

    # compute normalized sub-frame energy
    norm_sub_frame_energy = np.sum(sub_win ** 2, axis=0) / (frame_energy + epsilon)

    # compute entropy
    entropy = -np.sum(epsilon * np.log2(norm_sub_frame_energy + epsilon))

    return entropy


def compute_tempo(signal, sample_rate):
    tempo = librosa.beat.tempo(y=signal, sr=sample_rate)
    return tempo


def compute_rms(signal, frame_length, hop_length):
    rms = librosa.feature.rms(y=signal,
                              frame_length=frame_length,
                              hop_length=hop_length)
    return rms


def compute_zcr(signal, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(y=signal,
                                             frame_length=frame_length,
                                             hop_length=hop_length)
    return zcr


def compute_mfcc(signal, sample_rate, num_mfcc, num_fft, hop_length):
    mfcc = librosa.feature.mfcc(y=signal,
                                sr=sample_rate,
                                n_mfcc=num_mfcc,
                                n_fft=num_fft,
                                hop_length=hop_length)

    return mfcc


def compute_chroma_stft(signal, sample_rate, n_fft, hop_length):
    chroma_stft = librosa.feature.chroma_stft(y=signal,
                                              sr=sample_rate,
                                              n_fft=n_fft,
                                              hop_length=hop_length)
    return chroma_stft


def compute_spectral_centroid(signal, sample_rate, n_fft, hop_length):
    spectral_centroid = librosa.feature.spectral_centroid(y=signal,
                                                          sr=sample_rate,
                                                          n_fft=n_fft,
                                                          hop_length=hop_length)
    return spectral_centroid


def compute_spectral_bandwidth(signal, sample_rate, n_fft, hop_length):
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal,
                                                            sr=sample_rate,
                                                            n_fft=n_fft,
                                                            hop_length=hop_length)
    return spectral_bandwidth


def compute_spectral_rolloff(signal, sample_rate, n_fft, hop_length):
    spectral_rollof = librosa.feature.spectral_rolloff(y=signal,
                                                       sr=sample_rate,
                                                       n_fft=n_fft,
                                                       hop_length=hop_length)
    return spectral_rollof
