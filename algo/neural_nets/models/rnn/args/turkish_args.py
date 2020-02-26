TEMP_DIRECTORY = "turkish_temp/data"
TRAIN_FILE = "train.tsv"
DEV_FILE = "dev.tsv"
TEST_FILE = "test.tsv"
DEV_RESULT_FILE = "dev_result.tsv"
SUBMISSION_FOLDER = "results"
SUBMISSION_FILE = "turkish_rnn"
RESULT_FILE = "result.csv"
TRANSFER_LEARNING = True
MODEL_PATH = "turkish_temp"
MODEL_NAME = "model.hdfs"
GRAPH_NAME = "graph.png"
TURKISH_EMBEDDING_PATH = '/data/fasttext/grcorpus_def.vec'
MAX_SIZE = None
SPLIT_RATIO = 0.9
BATCH_SIZE = 256
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
N_EPOCHS = 20
N_FOLD = 5
LEARNING_RATE = 0.9e-3
REDUCE_LEARNING_RATE_THRESHOLD = 1e-6
REDUCE_LEARNING_RATE_FACTOR = 0.9
GRADUALLY_UNFREEZE = False
FREEZE_FOR = 20
