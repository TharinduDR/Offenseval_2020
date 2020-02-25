import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchtext import data
from torchtext import vocab

from algo.neural_nets.common.preprocessing.greek_preprocessing import pipeline
from algo.neural_nets.common.run_model import fit, predict, threshold_search
from algo.neural_nets.common.utility import evaluatation_scores, print_model, draw_graph
from algo.neural_nets.models.rnn.common.model import RNN
from algo.neural_nets.models.rnn.args.greek_args import SPLIT_RATIO, BATCH_SIZE, \
    N_EPOCHS, MODEL_PATH, TEMP_DIRECTORY, TRAIN_FILE, DEV_FILE, N_FOLD, LEARNING_RATE, REDUCE_LEARNING_RATE_THRESHOLD, \
    REDUCE_LEARNING_RATE_FACTOR, MODEL_NAME, GRAPH_NAME, GRADUALLY_UNFREEZE, FREEZE_FOR, DEV_RESULT_FILE, \
    GREEK_EMBEDDING_PATH, HIDDEN_DIM, BIDIRECTIONAL, N_LAYERS, DROPOUT, TEST_FILE, SUBMISSION_FILE
from project_config import SEED, VECTOR_CACHE, GREEK_DATA_PATH, GREEK_TEST_PATH
from util.logginghandler import TQDMLoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[TQDMLoggingHandler()])

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

full = pd.read_csv(GREEK_DATA_PATH, sep='\t')
test = pd.read_csv(GREEK_TEST_PATH, sep='\t')

le = LabelEncoder()
full['encoded_subtask_a'] = le.fit_transform(full["subtask_a"])
train, dev = train_test_split(full, test_size=0.2, random_state=SEED)

train.to_csv(os.path.join(TEMP_DIRECTORY, TRAIN_FILE), header=True, sep='\t', index=False, encoding='utf-8')
dev.to_csv(os.path.join(TEMP_DIRECTORY, DEV_FILE), header=True, sep='\t', index=False, encoding='utf-8')
test.to_csv(os.path.join(TEMP_DIRECTORY, TEST_FILE), header=True, sep='\t', index=False, encoding='utf-8')

id_variable = data.Field()
text_variable = data.Field(tokenize=pipeline)
target_variable = data.LabelField(dtype=torch.float)

train_fields = [
    ('id', None),  # we dont need this, so no processing
    ('tweet', text_variable),  # process it as text
    ('subtask_a', None),  # process it as label
    ('encoded_subtask_a', target_variable)
]

dev_fields = [
    ('id', id_variable),  # we process this as id field
    ('tweet', text_variable),  # process it as text
    ('subtask_a', None),  # process it as label
    ('encoded_subtask_a', None)
]

test_fields = [
    ('id', id_variable),  # we process this as id field
    ('tweet', text_variable),  # process it as text
    ('subtask_a', None),  # process it as label
    ('encoded_subtask_a', None)
]

# Creating our train and test data
train_data = data.TabularDataset(
    path=os.path.join(TEMP_DIRECTORY, TRAIN_FILE),
    format='tsv',
    skip_header=True,
    fields=train_fields
)

dev_data = data.TabularDataset(
    path=os.path.join(TEMP_DIRECTORY, DEV_FILE),
    format='tsv',
    skip_header=True,
    fields=dev_fields
)

test_data = data.TabularDataset(
    path=os.path.join(TEMP_DIRECTORY, TEST_FILE),
    format='tsv',
    skip_header=True,
    fields=dev_fields
)

vec = vocab.Vectors(GREEK_EMBEDDING_PATH, cache=VECTOR_CACHE)

dev_preds = np.zeros((len(dev_data), N_FOLD))
test_preds = np.zeros((len(test_data), N_FOLD))
deltas = []

for i in range(N_FOLD):
    logging.info("****** Fold {} ******".format(i + 1))

    train_data, valid_data = train_data.split(split_ratio=SPLIT_RATIO, random_state=random.seed(SEED * i))

    logging.info(f'Number of training examples: {len(train_data)}')
    logging.info(f'Number of validation examples: {len(valid_data)}')
    logging.info(f'Number of dev examples: {len(dev_data)}')
    logging.info(f'Number of test examples: {len(test_data)}')

    # Build the vocabulary using only the train dataset?,
    # and also by specifying the pretrained embedding
    text_variable.build_vocab(train_data, dev_data, valid_data, test_data, vectors=vec, max_size=None)
    target_variable.build_vocab(train_data)
    id_variable.build_vocab(dev_data)

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
    dev_iter = data.Iterator(
        dev_data,
        batch_size=BATCH_SIZE,
        device=device,
        train=False,
        sort=False,
        sort_within_batch=False
    )

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

    model = RNN(input_dim, embedding_dim, output_dim, pretrained_embeddings, HIDDEN_DIM, BIDIRECTIONAL,
                N_LAYERS, DROPOUT)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=REDUCE_LEARNING_RATE_FACTOR,
                                  threshold=REDUCE_LEARNING_RATE_THRESHOLD)

    path = os.path.join(MODEL_PATH, str(i + 1))
    if not os.path.exists(path): os.makedirs(path)

    model = model.to(device)
    if i == 0: print_model(model)
    criterion = criterion.to(device)

    trained_model, trained_losses, valid_losses = fit(model, train_iter, valid_iter, optimizer, criterion, scheduler,
                                                      N_EPOCHS, os.path.join(path, MODEL_NAME), GRADUALLY_UNFREEZE,
                                                      FREEZE_FOR)

    draw_graph(n_epohs=N_EPOCHS, valid_losses=valid_losses, trained_losses=trained_losses,
               path=os.path.join(path, GRAPH_NAME))

    delta = threshold_search(trained_model, valid_iter)
    dev_pred = predict(trained_model, dev_iter)
    test_pred = predict(trained_model, test_iter)

    dev_preds[:, i] = (np.array(dev_pred) >= delta).astype(int)
    test_preds[:, i] = (np.array(test_pred) >= delta).astype(int)

dev = pd.read_csv(os.path.join(TEMP_DIRECTORY, DEV_FILE), sep='\t')
dev["predictions"] = (dev_preds.mean(axis=1) > 0.5).astype(int)

test = pd.read_csv(os.path.join(TEMP_DIRECTORY, TEST_FILE), sep='\t')
test["subtask_a"] = le.inverse_transform((test_preds.mean(axis=1) > 0.5).astype(int))

# Performing the evaluation
(tn, fp, fn, tp), accuracy, weighted_f1, macro_f1, weighted_recall, weighted_precision = evaluatation_scores(dev,
                                                                                                             'encoded_subtask_a',
                                                                                                             "predictions")

dev.to_csv(os.path.join(TEMP_DIRECTORY, DEV_RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')

test = test[["id", "subtask_a"]]
test.to_csv(os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE), header=False, sep=',', index=False, encoding='utf-8')

logging.info("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))
logging.info("Accuracy {}".format(accuracy))
logging.info("Weighted F1 {}".format(weighted_f1))
logging.info("Macro F1 {}".format(macro_f1))
logging.info("Weighted Recall {}".format(weighted_recall))
logging.info("Weighted Precision {}".format(weighted_precision))
