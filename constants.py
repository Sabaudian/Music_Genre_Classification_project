# DATASET AND DATA LOCATION
OG_DATASET_PATH = "genres_original"  # location of original dataset
MY_DATASET_PATH = "genres_updated"  # location of data after pre-processing
DATA_PATH = "data/data.csv"  # where to save extracted feature
EXCLUDE_FOLDER = "source"  # exclude this folder from feature extraction
CHUNK_LENGTH = 3000  # new length of chunks of audio sample
FEATURE_HEADER = "filename chroma_stft_mean rms_mean spectral_centroid_mean spectral_bandwidth_mean rolloff_mean zero_crossing_rate_mean tempo"
GENRES = "blues classical country disco hiphop jazz metal pop reggae rock"


# AUDIO CHARACTERISTICS
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
DURATION = 3

# USED FOR FEATURES EXTRACTION
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


