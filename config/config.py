# CONFIG
MAX_LENGTH = 120

# CONFIG TRAIN MODEL TENSORFLOW
PATH_TRAIN = '../data/summary.txt'
EMBEDDING_SIZE = 100
NUM_UNIT = 64
DROPOUT_RATE = 0.1
NUM_STEP = 3000
BATCH_SIZE = 128
DISPLAY_STEP = 100
SAVE_PATH_CHECKPOINT = './checkpoints/model'

# CONFIG SAVE PKL FROM TRAIN
PATH_ALLWORDS = '../config/allwords.pkl'
PATH_ALLTAGS = '../config/alltags.pkl'
PATH_WORD2IDX = '../config/word2idx.pkl'
PATH_IDX2WORD = '../config/idx2word.pkl'
PATH_TAG2IDX = '../config/tag2idx.pkl'
PATH_IDX2TAG = '../config/idx2tag.pkl'

# CONFIG LOAD MODEL FROM API
LOAD_CHECKPOINT = './train/checkpoints/'
LOAD_ALLWORDS = './config/allwords.pkl'
LOAD_ALLTAGS = './config/alltags.pkl'
LOAD_WORD2IDX = './config/word2idx.pkl'
LOAD_IDX2WORD = './config/idx2word.pkl'
LOAD_TAG2IDX = './config/tag2idx.pkl'
LOAD_IDX2TAG = './config/idx2tag.pkl'
