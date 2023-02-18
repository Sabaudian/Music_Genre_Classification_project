import librosa.beat


def compute_tempo(signal, sample_rate):

    tempo = librosa.beat.tempo(y=signal, sr=sample_rate)
    return tempo


def compute_rms(signal, frame_length, hop_length):

    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)
    return rms


def compute_zcr(signal, frame_length, hop_length):

    zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_length, hop_length=hop_length)
    return zcr


def compute_mfcc(signal, sample_rate, num_mfcc, num_fft, hop_length):

    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=num_fft, hop_length=hop_length)

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
