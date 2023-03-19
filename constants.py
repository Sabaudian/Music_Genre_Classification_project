# DATASET AND DATA LOCATION
DATASET_PATH = "genres_original"  # location of original dataset
GENRES_3S_PATH = "genres_3s"  # location of data after pre-processing
DATA_FOLDER = "data"  # location of my data
DATA_PATH = "data/data.csv"  # extracted feature
CLF_REPORT_PATH = "clf_report"  # location of classification models' report
CHUNK_LENGTH = 3000  # new length of chunks of audio sample
FEATURE_HEADER = "filename chroma_stft_mean rms_mean spectral_centroid_mean spectral_bandwidth_mean rolloff_mean zero_crossing_rate_mean tempo energy entropy_of_entropy"
GENRES = "blues classical country disco hiphop jazz metal pop reggae rock"

# AUDIO CHARACTERISTICS
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
DURATION = 3  # duration of audio after pre-processing

# USED FOR FEATURES EXTRACTION
NUM_MFCC = 20
NUM_FTT = 2048
FRAME_SIZE = 1024
HOP_LENGHT = 512

# FOR VARIOUS COMPUTATION
STD_NORM = "std"
MIN_MAX_NORM = "min_max"
VARIANCE_RATIO = 0.8

# FOR PLOT AND SIMILAR
PLOT_FOLDER = "plot"
COLORS_LIST = {"#006400", "#00008b", "#b03060", "#ff0000", "#ffd700", "#deb887", "#00ff00", "#00ffff", "#ff00ff",
               "#6495ed"}
PRED_EVA_LIST = {"#006400", "#ffd700"}
GENRES_LIST = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
FEATURES_LIST = ["chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth", "rolloff", "zcr", "tempo", "energy",
                 "entropy", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9",
                 "mfcc_10", "mfcc_11", "mfcc_12", "mfcc_13", "mfcc_14", "mfcc_15", "mfcc_16", "mfcc_17", "mfcc_18",
                 "mfcc_19", "mfcc_20"]
FEATURES_VISUALIZATION_PATH = {"blues": "genres_original/blues/blues.00000.wav",
                               "classical": "genres_original/classical/classical.00000.wav",
                               "country": "genres_original/country/country.00000.wav",
                               "disco": "genres_original/disco/disco.00000.wav",
                               "hiphop": "genres_original/hiphop/hiphop.00000.wav",
                               "jazz": "genres_original/jazz/jazz.00000.wav",
                               "metal": "genres_original/metal/metal.00000.wav",
                               "pop": "genres_original/pop/pop.00000.wav",
                               "reggae": "genres_original/reggae/reggae.00000.wav",
                               "rock": "genres_original/rock/rock.00000.wav"}

# FOR NAMING FILE AND SIMILAR
STORE_PATH = "plot/"
JPG = ".jpg"
CORR_MATR_TAG = "correlation_matrix_plot"
OPT_N_COMP_TAG = "pca_opt_num_of_components_plot"
PCA_TAG = "pca_scatter_plot"
K_MEAN_PCA_CC_TAG = "pca_k-mean_cluster_centroids_plot"
K_MEAN_CONF_MATR_TAG = "k-means_confusion_matrix_plot"
CONF_MATR_TAG = "_confusion_matrix_plot"
ROC_CURVE_TAG = "_roc_plot"
PREDICT_BY_GENRES_TAG = "_compare_predictions_by_genres_plot"
PREDICT_EV_TAG = "_predictions_evaluation_plot"
CLF_REPORT_TAG = "_clf_report"
