import genres_sl_functions
import genres_ul_functions

from utils import prepare_dataset as prepare
from utils import features_extraction as feature
from utils import features_visualization as visualize

import constants as const

# main class calling methods for feature extraction, classification and clustering of dataset
if __name__ == '__main__':

    # pre-processing data
    check_preprocessing_data = input("> START PRE-PROCESSING DATA? [Y/N]: ")
    if check_preprocessing_data.upper() == "Y":

        # check duration of sound file
        check_if_sound_duration = input("\n>>> CHECK SOUND QUALITY? [Y/N]: ")
        if check_if_sound_duration.upper() == "Y":
            prepare.check_sound_duration(dataset_path=const.DATASET_PATH)

        # make chunk from og. sound file
        check_if_chunks_exist = input("\n>>> PERMORFORMING DATA AUGMENTATION? [Y/N]: ")
        if check_if_chunks_exist.upper() == "Y":
            prepare.make_chunks_from_data(dataset_path=const.DATASET_PATH, chunk_length=const.CHUNK_LENGTH,
                                          new_dir_path=const.GENRES_3S_PATH)

    # extract and visualize features from samples
    check_feature_extraction = input("\n> EXTRACT FEATURES FROM DATA? [Y/N]: ")
    if check_feature_extraction.upper() == "Y":
        feature.features_extraction_to_csv(dataset_path=const.GENRES_3S_PATH, destination_path=const.DATA_PATH)
    else:  # visualization of features
        check_features_visualization = input("\n> VISUALIZE THE EXTRACTED FEATURES? [Y/N]: ")
        if check_features_visualization.upper() == "Y":
            visualize.visualize_features()

    # clustering and classification method
    check_cluster_and_classification = input("\n> START LEARNING? [Y/N]: ")
    if check_cluster_and_classification.upper() == "Y":
        check_k_mean_cluster = input("\n>>> PERFORM K-MEANS ALGORITHM? [Y/N]: ")
        if check_k_mean_cluster.upper() == "Y":
            genres_ul_functions.clustering_and_evaluation(data_path=const.DATA_PATH,
                                                          normalization_type=const.MIN_MAX_NORM)

        check_classification_models = input("\n>>> PERFORM CLASSIFICATION? [Y/N]: ")
        if check_classification_models.upper() == "Y":
            genres_sl_functions.classification_and_evaluation(data_path=const.DATA_PATH,
                                                              normalization_type=const.MIN_MAX_NORM)


