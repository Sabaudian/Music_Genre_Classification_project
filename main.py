import genres_sl_functions
import genres_ul_functions

from utils import prepare_dataset as prepare
from utils import features_extraction as feature
from utils import features_visualization as visualize

import constants as const

# main class calling methods for feature extraction, clustering and classification
if __name__ == '__main__':

    # Pre-processing data
    check_preprocessing_data = input("> START PRE-PROCESSING DATA? [Y/N]: ")
    if check_preprocessing_data.upper() == "Y":

        # check file extension
        check_if_extension_is_wav = input("\n>>> CHECK FILE EXTENSION? [Y/N]: ")
        if check_if_extension_is_wav.upper() == "Y":
            prepare.check_file_extension(dataset_path=const.DATASET_PATH,
                                         file_extension=const.WAV)

        # check duration of sound file
        check_if_sound_duration = input("\n>>> CHECK SOUND DURATION? [Y/N]: ")
        if check_if_sound_duration.upper() == "Y":
            prepare.check_sound_duration(dataset_path=const.DATASET_PATH,
                                         milliseconds_duration=const.DURATION_MS)


        # make chunk from og. sound file
        check_if_chunks_exist = input("\n>>> PERMORFORMING DATA AUGMENTATION? [Y/N]: ")
        if check_if_chunks_exist.upper() == "Y":
            prepare.make_chunks_from_data(dataset_path=const.DATASET_PATH,
                                          chunk_length=const.CHUNK_LENGTH,
                                          new_dir_path=const.GENRES_3S_PATH)

    # Extract and visualize features from samples
    check_feature_extraction = input("\n> EXTRACT FEATURES FROM DATA? [Y/N]: ")
    if check_feature_extraction.upper() == "Y":

        feature.features_extraction_to_csv(dataset_path=const.GENRES_3S_PATH,
                                           data_folder=const.DATA_FOLDER,
                                           data_path=const.DATA_PATH)
    else:  # visualization of audio signals' features
        check_features_visualization = input("\n> FEATURES VISUALIZATION? [Y/N]: ")
        if check_features_visualization.upper() == "Y":
            visualize.visualize_features()

    # K-Means clustering and evaluation
    check_k_mean_cluster = input("\n> PERFORM K-MEANS ALGORITHM? [Y/N]: ")
    if check_k_mean_cluster.upper() == "Y":
        genres_ul_functions.clustering_and_evaluation(data_path=const.DATA_PATH)
    # Classification models and evaluation
    check_classification_models = input("\n> PERFORM CLASSIFICATION? [Y/N]: ")
    if check_classification_models.upper() == "Y":
        genres_sl_functions.classification_and_evaluation(data_path=const.DATA_PATH)
