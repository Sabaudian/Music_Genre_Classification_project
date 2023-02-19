# DATASET AND DATA LOCATION
DATASET_PATH = "genres"  # location of original dataset
DATA_PATH = "data/data.csv"  # where to save extracted feature
CHUNK_LENGTH = 3000  # new length of chunks of audio sample
FEATURE_HEADER = "filename chroma_stft_mean rms_mean spectral_centroid_mean spectral_bandwidth_mean rolloff_mean zero_crossing_rate_mean tempo"
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

# FOR KMEAN AND SIMILAR
VARIANCE_RATIO = 0.8

# FOR PLOT AND SIMILAR
COLORS_LIST = {"red", "blue", "green", "purple", "orange", "deeppink", "skyblue", "aquamarine", "teal", "dodgerblue"}
GENRES_SET = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


