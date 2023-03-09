# DATASET AND DATA LOCATION
DATASET_PATH = "genres"  # location of original dataset
DATA_PATH = "data/data.csv"  # where to save extracted feature
CHUNK_LENGTH = 3000  # new length of chunks of audio sample
FEATURE_HEADER = "filename chroma_stft_mean rms_mean spectral_centroid_mean spectral_bandwidth_mean rolloff_mean zero_crossing_rate_mean tempo energy entropy_of_entropy"
GENRES = "blues classical country disco hiphop jazz metal pop reggae rock"


# AUDIO CHARACTERISTICS
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
DURATION = 3  # duration of audio after pre-processing

# USED FOR FEATURES EXTRACTION
EXCLUDE_FOLDER = "original_30s"  # exclude this folder from feature extraction
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
NUM_MFCC = 20
NUM_FTT = 2048
FRAME_SIZE = 1024
HOP_LENGHT = 512

# FOR K-MEAN AND SIMILAR
VARIANCE_RATIO = 0.8

# FOR PLOT AND SIMILAR
ROC_COLOR_LIST = {"#006400", "#00008b", "#b03060", "#ff0000", "#ffd700", "#deb887", "#00ff00", "#00ffff", "#ff00ff", "#6495ed"}
COLORS_LIST = {"#006400", "#00008b", "#b03060", "#ff0000", "#ffd700", "#deb887", "#00ff00", "#00ffff", "#ff00ff", "#6495ed"}
PRED_EVA_LIST = {"#006400", "#ffd700"}
GENRES_LIST = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# FOR NAMING FILE AND SIMILAR
STORE_PATH = "plot"
JPG = ".jpg"
CORR_MATR_TAG = "correlation_matrix"
PCA_TAG = "pca_scatter_plot"
K_MEAN_PCA_CC = "pca_k-mean_cluster_centroids_plot"
K_MEAN_CONF_MATR = "k-means_confusion_matrix_plot"
CONF_MATR = "_confusion_matrix"
ROC_CURVE = "_roc_plot"
