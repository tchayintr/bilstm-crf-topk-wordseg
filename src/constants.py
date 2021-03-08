global __version
__version__ = 'v0.1.0'

# common

GPU_DEVICE = 'GPU'
CUDA_VISIBLE_DEVICE = 'CUDA_VISIBLE_DEVICES'
CUDA_DEVICE = 'cuda'
CPU_DEVICE = 'cpu'

# task

TASK_SEG = 'seg'

# tagging unit

SINGLE_TAGGING = 'single'

# for analyzer

LOG_DIR = 'log'
MODEL_DIR = 'models/main'
DECODE_DIR = 'models/decode'

# for dictionary

UNK_SYMBOL = '<UNK>'
NUM_SYMBOL = '<NUM>'
NONE_SYMBOL = '<NONE>'

UNIGRAM = 'unigram'
SEG_LABEL = 'seg_label'

TYPE_THAI = '<TH>'
TYPE_ENG = '<EN>'
TYPE_HIRA = '<HR>'
TYPE_KATA = '<KT>'
TYPE_LONG = '<LG>'
TYPE_KANJI = '<KJ>'
TYPE_ALPHA = '<AL>'
TYPE_DIGIT = '<DG>'
TYPE_SPACE = '<SC>'
TYPE_SYMBOL = '<SY>'
TYPE_ASCII_OTHER = '<AO>'

# for character

SEG_LABELS = 'BIES'
B = 'B'
I = 'I'
E = 'E'
S = 'S'
O = 'O'

# for data io

PADDING_LABEL = -1
NUM_FOR_REPORTING = 100000

SL_COLUMN_DELIM = '\t'
SL_TOKEN_DELIM = ' '
SL_ATTR_DELIM = '_'
WL_TOKEN_DELIM = '\n'
WL_ATTR_DELIM = '\t'
KEY_VALUE_DELIM = '='
COMMENT_SYM = '#'

SL_FORMAT = 'sl'
WL_FORMAT = 'wl'

# dataset

MAX_VOCAB_SIZE = 128000

# token

INIT_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
