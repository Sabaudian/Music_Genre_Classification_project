import os
import csv
import librosa
import numpy as np
import pandas as pd

import constants as const
from utils import features_computation as fc


def features_extraction_to_csv(dataset_path, data_path):

    # generate a dataset
    header = const.FEATURE_HEADER

    for n_mfcc in range(1, 21):
        header += f" mfcc{n_mfcc}_mean"
    header += " genre"

    header = header.split()

    # generate file.csv
    file = open(data_path, "w", newline="")
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    for dirpath, dirnames, filenames in os.walk(dataset_path):
        # exclude this folder
        dirnames[:] = [d for d in dirnames if d not in const.EXCLUDE_FOLDER]

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:
            print("\ndirpath: {}".format(dirpath))

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            print("\nsemantic label: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in sorted(filenames):

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
                # mfcc
                mfcc = fc.compute_mfcc(signal, sample_rate, const.NUM_MFCC, const.NUM_FTT, const.HOP_LENGHT)

                to_append = f"{f} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_centroid)} {np.mean(spec_bandwidth)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(tempo)}"

                for n in mfcc:
                    to_append += f" {np.mean(n)}"
                to_append += f" {semantic_label}"
                data_file = open(data_path, "a", newline="")
                with data_file:
                    writer = csv.writer(data_file)
                    writer.writerow(to_append.split())

    # # check correct creation of CSV file
    # if os.path.exists(data_path):
    #     # sorting per filename and genres
    #     dataFrame = pd.read_csv(data_path)
    #     dataFrame.sort_values(["filename", "genre"], ascending=True, inplace=True, ignore_index=True)
    #     dataFrame.to_csv(data_path, index=False)
    #     # Printing messages
    #     print("\nCSV Saved!")
    #     print("Features Extractions Completed!")



