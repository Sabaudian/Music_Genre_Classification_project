import os
import csv
import math
import librosa
import numpy as np

# my import functions
import constants as const
from utils import features_computation as fc


def makedir(dir_path):
    # create a new directory
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def features_extraction_to_csv(dataset_path, data_folder, data_path):
    # make new directory for extracted data
    makedir(data_folder)

    # generate a dataset
    header = const.FEATURE_HEADER

    # update header with MFCCs and genre label
    for n_mfcc in range(1, 21):
        header += f" mfcc{n_mfcc}_mean"
        header += f" mfcc{n_mfcc}_var"
    header += " genre"

    header = header.split()

    # generate file.csv
    file = open(data_path, "w", newline="")
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    for dirpath, dirnames, filenames in sorted(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            print("\nExtracting Features from Folder: \033[92m{}\033[0m".format(semantic_label))

             # process all audio files in genre sub-dir
            for f in sorted(filenames):

                print("\033[92m{}\033[0m checked!".format(f))

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=const.SAMPLE_RATE, duration=const.DURATION,
                                                   mono=True)

                # chromagram stft
                chroma_stft = fc.compute_chroma_stft(signal, sample_rate, const.NUM_FTT, const.HOP_LENGHT)
                # rms
                rms = fc.compute_rms(signal, const.FRAME_SIZE, const.HOP_LENGHT)
                # spectral centroid
                spec_centroid = fc.compute_spectral_centroid(signal, sample_rate, const.NUM_FTT, const.HOP_LENGHT)
                # spectral bandwidth
                spec_bandwidth = fc.compute_spectral_bandwidth(signal, sample_rate, const.NUM_FTT, const.HOP_LENGHT)
                # spectral rolloff
                rolloff = fc.compute_spectral_rolloff(signal, sample_rate, const.NUM_FTT, const.HOP_LENGHT)
                # zcr
                zcr = fc.compute_zcr(signal, const.FRAME_SIZE, const.HOP_LENGHT)
                # tempo
                tempo = fc.compute_tempo(signal, sample_rate)
                # energy
                energy = fc.compute_energy(signal)
                # entropy of energy
                entropy_of_energy = fc.compute_entropy_of_energy(signal, num_of_short_blocks=math.ceil(
                    len(signal) / const.FRAME_SIZE))
                # mfcc
                mfcc = fc.compute_mfcc(signal, sample_rate, const.NUM_MFCC, const.NUM_FTT, const.HOP_LENGHT)

                to_append = f"{f} {np.mean(chroma_stft)} {np.var(chroma_stft)} {np.mean(rms)} {np.var(rms)} " \
                            f"{np.mean(spec_centroid)} {np.var(spec_centroid)} {np.mean(spec_bandwidth)} " \
                            f"{np.var(spec_bandwidth)} {np.mean(rolloff)} {np.var(rolloff)} {np.mean(zcr)} " \
                            f"{np.var(zcr)} {np.mean(tempo)} {np.mean(energy)} {np.mean(entropy_of_energy)}"

                # MFCCs_
                for n in mfcc:
                    to_append += f" {np.mean(n)}"  # append MFCCs_mean
                    to_append += f" {np.var(n)}"   # append MFCCs_var
                to_append += f" {semantic_label}"  # append genre
                data_file = open(data_path, "a", newline="")

                # write data to file
                with data_file:  # write data to file
                    writer = csv.writer(data_file)
                    writer.writerow(to_append.split())
