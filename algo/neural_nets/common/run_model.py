import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from util.logginghandler import TQDMLoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[TQDMLoggingHandler()])


def init_model(model, method='xavier', exclude='embedding'):
    logging.info("Initialising the model")
    for name, w in model.named_parameters():
        if not exclude in name:
            if 'weight' in name:
                if method is 'xavier':
                    nn.init.xavier_normal_(w)
                elif method is 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0.0)
            else:
                pass


def threshold_search(trained_model, valid_iter):
    valid_pred = []
    valid_truth = []

    trained_model.eval()

    with torch.no_grad():
        for batch in valid_iter:
            valid_truth += batch.encoded_subtask_a.cpu().numpy().tolist()
            predictions = trained_model(batch.tweet).squeeze(1)
            valid_pred += torch.sigmoid(predictions).cpu().data.numpy().tolist()

    tmp = [0, 0, 0]  # idx, cur, max
    delta = 0
    for tmp[0] in np.arange(0.1, 0.501, 0.01):
        tmp[1] = f1_score(valid_truth, np.array(valid_pred) > tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    logging.info('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
    return delta


def predict(trained_model, test_iter):
    test_pred = []
    test_id = []

    trained_model.eval()

    with torch.no_grad():
        for batch in test_iter:
            predictions = trained_model(batch.tweet).squeeze(1)
            test_pred += torch.sigmoid(predictions).cpu().data.numpy().tolist()
            test_id += batch.id.view(-1).cpu().numpy().tolist()

    return test_pred, test_id


def train(model, iterator, optimizer, criterion):
    # Track the loss
    epoch_loss = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.tweet).squeeze(1)
        loss = criterion(predictions, batch.encoded_subtask_a)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return model, epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.tweet).squeeze(1)
            loss = criterion(predictions, batch.encoded_subtask_a)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def fit(model, train_iter, valid_iter, optimizer, criterion, scheduler, epochs, path):
    init_model(model)
    # Track time taken
    start_time = time.time()
    best_loss = math.inf
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        epoch_start_time = time.time()

        trained_model, train_loss = train(model, train_iter, optimizer, criterion)
        valid_loss = evaluate(trained_model, valid_iter, criterion)
        scheduler.step(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        logging.info(f'| Epoch: {epoch + 1:02} '
                     f'| Train Loss: {train_loss:.3f} '
                     f'| Val. Loss: {valid_loss:.3f} '
                     f'| Time taken: {time.time() - epoch_start_time:.2f}s'
                     f'| Time elapsed: {time.time() - start_time:.2f}s')

        if valid_loss < best_loss:
            best_loss = valid_loss
            logging.info("Saving the current best model")
            torch.save(trained_model, path)

    final_model = torch.load(path)
    return final_model, train_losses, valid_losses
