import logging
import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext import data
from torchtext import vocab

from algo.neural_nets.common.preprocessing import pipeline
from algo.neural_nets.models.rnn.model import RNN
from algo.neural_nets.models.rnn.model_config import SPLIT_RATIO, EMBEDDING_PATH, BATCH_SIZE, \
    N_EPOCHS, MODEL_PATH, TEMP_DIRECTORY, TRAIN_FILE, TEST_FILE
from algo.neural_nets.common.run_model import fit, predict, threshold_search
from algo.neural_nets.common.utility import evaluatation_scores

from project_config import SEED, DATA_PATH
from util.logginghandler import TQDMLoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[TQDMLoggingHandler()])

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(TEMP_DIRECTORY):os.makedirs(TEMP_DIRECTORY)

full = pd.read_csv(DATA_PATH, sep='\t')

le = LabelEncoder()
full['encoded_subtask_a'] = le.fit_transform(full["subtask_a"])
train, test = train_test_split(full, test_size=0.2, random_state=SEED)

train.to_csv(os.path.join(TEMP_DIRECTORY, TRAIN_FILE), header=True, sep='\t', index=False, encoding='utf-8')
test.to_csv(os.path.join(TEMP_DIRECTORY, TEST_FILE), header=True, sep='\t', index=False, encoding='utf-8')

id_variable = data.Field()
text_variable = data.Field(tokenize=pipeline)
target_variable = data.LabelField(dtype=torch.float)

train_fields = [
    ('id', None),  # we dont need this, so no processing
    ('tweet', text_variable),  # process it as text
    ('subtask_a', None),  # process it as label
    ('subtask_b', None),  # we dont need this, so no processing
    ('subtask_c', None),  # we dont need this, so no processing
    ('encoded_subtask_a', target_variable)
]

test_fields = [
    ('id', id_variable),  # we process this as id field
    ('tweet', text_variable),  # process it as text
    ('subtask_a', None),  # process it as label
    ('subtask_b', None),  # we dont need this, so no processing
    ('subtask_c', None),  # we dont need this, so no processing
    ('encoded_subtask_a', None)
]

# Creating our train and test data
train_data = data.TabularDataset(
    path=os.path.join(TEMP_DIRECTORY, TRAIN_FILE),
    format='tsv',
    skip_header=True,
    fields=train_fields
)

test_data = data.TabularDataset(
    path=os.path.join(TEMP_DIRECTORY, TEST_FILE),
    format='tsv',
    skip_header=True,
    fields=test_fields
)

train_data, valid_data = train_data.split(split_ratio=SPLIT_RATIO, random_state=random.seed(SEED))

logging.info(f'Number of training examples: {len(train_data)}')
logging.info(f'Number of validation examples: {len(valid_data)}')
logging.info(f'Number of test examples: {len(test_data)}')

vec = vocab.Vectors(EMBEDDING_PATH)

# Build the vocabulary using only the train dataset?,
# and also by specifying the pretrained embedding
text_variable.build_vocab(train_data, vectors=vec, max_size=None)
target_variable.build_vocab(train_data)
id_variable.build_vocab(test_data)

logging.info(f'Unique tokens in TEXT vocab: {len(text_variable.vocab)}')
logging.info(f'Unique tokens in TARGET vocab: {len(target_variable.vocab)}')

# Automatically shuffles and buckets the input sequences into
# sequences of similar length
train_iter, valid_iter = data.BucketIterator.splits(
    (train_data, valid_data),
    sort_key=lambda x: len(x.tweet),  # what function/field to use to group the data
    batch_size=BATCH_SIZE,
    device=device
)

# Don't want to shuffle test data, so use a standard iterator
test_iter = data.Iterator(
    test_data,
    batch_size=BATCH_SIZE,
    device=device,
    train=False,
    sort=False,
    sort_within_batch=False
)

emb_shape = text_variable.vocab.vectors.shape
input_dim = emb_shape[0]
embedding_dim = emb_shape[1]
output_dim = 1
pretrained_embeddings = text_variable.vocab.vectors

model = RNN(input_dim, embedding_dim, output_dim, pretrained_embeddings)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

trained_model = fit(model, train_iter, valid_iter, optimizer, criterion, N_EPOCHS, MODEL_PATH)

delta = threshold_search(trained_model, valid_iter)

test_pred, test_id = predict(trained_model, test_iter)

test_pred = (np.array(test_pred) >= delta).astype(int)
test_id = [id_variable.vocab.itos[i] for i in test_id]

test = pd.read_csv(os.path.join(TEMP_DIRECTORY, TEST_FILE), sep='\t')
test["predictions"] = test_pred

# Performing the evaluation
(tn, fp, fn, tp), accuracy, weighted_f1, weighted_recall, weighted_precision = evaluatation_scores(test,
                                                                                                   'encoded_subtask_a',
                                                                                                   "predictions")
logging.info("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))
logging.info("Accuracy {}".format(accuracy))
logging.info("Weighted F1 {}".format(weighted_f1))
logging.info("Weighted Recall {}".format(weighted_recall))
logging.info("Weighted Precision {}".format(weighted_precision))
