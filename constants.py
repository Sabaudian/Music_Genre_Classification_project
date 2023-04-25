# DATASET AND DATA LOCATION
DATASET_PATH = "genres_original"  # location of original dataset
GENRES_3S_PATH = "genres_3s"  # location of data after pre-processing
DATA_FOLDER = "data"  # location of my data
DATA_PATH = "data/data.csv"  # extracted feature
CLF_REPORT_PATH = "clf_report"  # location of classification models' report
CHUNK_LENGTH = 3000  # new length of chunks of audio sample
DURATION_MS = 30000  # file duration in milliseconds
FEATURE_HEADER = "filename chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var tempo energy entropy_of_entropy"
GENRES = "blues classical country disco hiphop jazz metal pop reggae rock"

# AUDIO CHARACTERISTICS
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
DURATION = 3  # duration of audio after pre-processing in seconds

# USED FOR FEATURES EXTRACTION
NUM_MFCC = 20
NUM_FTT = 2048
FRAME_SIZE = 1024
HOP_LENGHT = 512

# FOR VARIOUS COMPUTATION
STD = "std"
MIN_MAX = "min_max"
VARIANCE_RATIO = 0.95

# FOR PLOT AND SIMILAR
PLOT_FOLDER = "plot"
CLASSIFICATION_PLOT_FOLDER = "classification_results"
CLUSTERING_PLOT_FOLDER = "clustering_results"
COLORS_LIST = {"#006400", "#00008b", "#b03060", "#ff0000", "#ffd700", "#deb887", "#00ff00", "#00ffff", "#ff00ff",
               "#6495ed"}
PRED_EVA_LIST = {"#006400", "#ffd700"}
GENRES_LIST = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
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
JPG = ".jpg"
CORR_MATR_TAG = "correlation_matrix_plot"
OPT_N_COMP_TAG = "pca_opt_num_of_components_plot"
PCA_TAG = "pca_scatter_plot"
K_MEAN_PCA_CC_TAG = "pca_k-means_cluster_centroids_plot"
K_MEAN_CONF_MATR_TAG = "k-means_confusion_matrix_plot"
CONF_MATR_TAG = "_confusion_matrix_plot"
ROC_CURVE_TAG = "_roc_curve_plot"
PREDICT_BY_GENRES_TAG = "_compare_predictions_by_genres_plot"
PREDICT_EV_TAG = "_predictions_evaluation_plot"
CLF_REPORT_TAG = "_clf_report"
