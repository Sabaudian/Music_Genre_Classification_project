# ********************************* #
# *********** CONSTANTS *********** #
# ********************************* #

# DATASET AND DATA PATH
# Location of the original dataset
DATASET_PATH = "genres_original"
# location of data after pre-processing
GENRES_3S_PATH = "genres_3s"
# location of my data
DATA_FOLDER = "data"
# extracted feature
DATA_PATH = "data/data.csv"
# location of classification models' report
CLF_REPORT_PATH = "clf_report"
# new length of the chunks
CHUNK_LENGTH = 3000
# file duration in milliseconds
DURATION_MS = 30000
# Header for extracted features
FEATURE_HEADER = ("filename chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean "
                  "spectral_centroid_var spectral_bandwidth_mean spectral_bandwidth_var rolloff_mean "
                  "rolloff_var zero_crossing_rate_mean zero_crossing_rate_var tempo energy entropy_of_entropy")
# List of genres
GENRES = "blues classical country disco hiphop jazz metal pop reggae rock"

# AUDIO CHARACTERISTICS
# Sample rate of audio files
SAMPLE_RATE = 22050
# Track duration in seconds
TRACK_DURATION = 30
# Duration of audio after pre-processing in seconds
DURATION = 3
# File extension for audio files
WAV = "wav"

# USED FOR FEATURES EXTRACTION
# Number of MFCC coefficients
NUM_MFCC = 20
# Number of points for Fast Fourier Transform
NUM_FTT = 2048
# Size of the frame for feature extraction
FRAME_SIZE = 1024
# Hop length for feature extraction
HOP_LENGHT = 512

# FOR VARIOUS COMPUTATION
# Method for feature scaling
MIN_MAX = "min_max"
# Variance ratio for PCA
VARIANCE_RATIO = 0.80
# Minimum number of clusters for K-Means
MIN_NUM_CLUSTERS = 2
# Maximum number of clusters for K-Means
MAX_NUM_CLUSTERS = 19

# FOR PLOT AND SIMILAR
# Folder for saving plots
PLOT_FOLDER = "plot"
# Folder for classification results plots
CLASSIFICATION_PLOT_FOLDER = "classification_results"
# Folder for clustering results plots
CLUSTERING_PLOT_FOLDER = "clustering_results"
# List of colors for plotting
COLORS_LIST = {"#006400", "#00008b", "#b03060", "#ff0000", "#ffd700", "#deb887", "#00ff00", "#00ffff", "#ff00ff",
               "#6495ed"}
# List of color for prediction evaluation
PRED_EVA_LIST = {"#006400", "#ffd700"}
# List of genres
GENRES_LIST = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
# Paths to example audio files for features visualization
FEATURES_VISUALIZATION_PATH = {
    "blues": "genres_original/blues/blues.00000.wav",
    "classical": "genres_original/classical/classical.00000.wav",
    "country": "genres_original/country/country.00000.wav",
    "disco": "genres_original/disco/disco.00000.wav",
    "hiphop": "genres_original/hiphop/hiphop.00000.wav",
    "jazz": "genres_original/jazz/jazz.00000.wav",
    "metal": "genres_original/metal/metal.00000.wav",
    "pop": "genres_original/pop/pop.00000.wav",
    "reggae": "genres_original/reggae/reggae.00000.wav",
    "rock": "genres_original/rock/rock.00000.wav"
}

# FOR NAMING FILE AND SIMILAR
# File extension for saving plots
JPG = ".jpg"

# Tags for various plot types
CORR_MATR_TAG = "correlation_matrix_plot"
OPT_N_COMP_TAG = "pca_opt_num_of_components_plot"
PCA_TAG = "pca_scatter_plot"
K_MEAN_PCA_CC_TAG = "pca_kMeans_cluster_centroids_plot"
K_MEAN_CONF_MATR_TAG = "kMeans_confusion_matrix_plot"
SILHOUETTE_TAG = "silhouette_analysis_plot"
CONF_MATR_TAG = "_confusion_matrix_plot"
ROC_CURVE_TAG = "_roc_curve_plot"
PREDICT_BY_GENRES_TAG = "_compare_predictions_by_genres_plot"
PREDICT_EV_TAG = "_predictions_evaluation_plot"
CLF_REPORT_TAG = "_clf_report"
