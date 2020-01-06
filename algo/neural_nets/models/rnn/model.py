import logging

import torch
import torch.nn as nn

from algo.neural_nets.models.rnn.model_config import HIDDEN_DIM, BIDIRECTIONAL, DROPOUT, N_LAYERS
from util.logginghandler import TQDMLoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[TQDMLoggingHandler()])


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pretrained_embeddings):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False

        self.rnn = nn.GRU(embedding_dim, HIDDEN_DIM, num_layers=N_LAYERS,
                          bidirectional=BIDIRECTIONAL, dropout=0 if N_LAYERS < 2 else DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM * 2 if BIDIRECTIONAL else HIDDEN_DIM, output_dim)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.embedding(x)
        embedded = self.dropout(x)
        output_gru, hidden = self.rnn(embedded)

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        return self.fc(hidden.squeeze(0))
