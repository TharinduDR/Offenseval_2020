TEMP_DIRECTORY = "english_temp/data"
TRAIN_FILE = "train.tsv"
TEST_FILE = "test.tsv"
RESULT_FILE = "task_a_rnn.csv"
MODEL_PATH = "english_temp"
MODEL_NAME = "model.hdfs"
GRAPH_NAME = "graph.png"
ENGLISH_EMBEDDING_PATH = '/data/fasttext/crawl-300d-2M-subword.vec'
MAX_SIZE = None
MAX_LEN = 70
SPLIT_RATIO = 0.9
BATCH_SIZE = 512
KERNEL_NUM = 128
FIXED_LENGTH = 70
KERNEL_SIZE = [1, 2, 3, 5]
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.1
N_EPOCHS = 20
N_FOLD = 5
LEARNING_RATE = 0.9e-3
REDUCE_LEARNING_RATE_THRESHOLD = 1e-6
REDUCE_LEARNING_RATE_FACTOR = 0.9
GRADUALLY_UNFREEZE = True
FREEZE_FOR = 10
