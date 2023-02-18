from utils import features_extraction as feature
from utils import prepare_dataset as prepare

import constants as const

# main class calling methods for feature extraction, classification and clustering of dataset
if __name__ == '__main__':

    # pre-processing data
    check_preprocessing_data = input("DO YOU WANT TO START PREPROCESSING DATA? [Y/N]: ")
    if check_preprocessing_data.upper() == "Y":

        check_if_ds_folder_exist = input("\n>> MAKING FOLDER FOR DATASET? [Y/N]: ")
        if check_if_ds_folder_exist.upper() == "Y":
            prepare.make_directory_for_dataset(new_dataset_path=const.MY_DATASET_PATH)

        check_if_ds_sub_folder_exist = input("\n>> MAKING SUB-FOLDER FOR DATASET? [Y/N]: ")
        if check_if_ds_sub_folder_exist.upper() == "Y":
            prepare.make_dataset_copy(source_folder=const.OG_DATASET_PATH, destination_folder=const.MY_DATASET_PATH)

        check_if_chunks_exist = input("\n>> PERMORFORMING DATA AUGMENTATION? [Y/N]: ")
        if check_if_chunks_exist.upper() == "Y":
            prepare.make_chunks_from_data(dataset_path=const.MY_DATASET_PATH, chunk_length=const.CHUNK_LENGTH)

    # extract feature from samples
    check_feature_extraction = input("\nDO YOU WANT TO EXTRACT FEATURES FROM DATASET? [Y/N]: ")
    if check_feature_extraction.upper() == "Y":
        feature.features_extraction_to_csv(dataset_path=const.MY_DATASET_PATH, data_path=const.DATA_PATH)
    else:
        print("\nWORK IN PROGRESS!!!")
        print("TO DO: \nClustering method\nClassification methods\nEvaluation of classification methods and comparison")
