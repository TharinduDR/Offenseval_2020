import torch
import torch.nn as nn

from neural_nets.models.bert.model_config import HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT


class BERTGRU(nn.Module):
    def __init__(self, bert, output_dim):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          HIDDEN_DIM,
                          num_layers=N_LAYERS,
                          bidirectional=BIDIRECTIONAL,
                          batch_first=True,
                          dropout=0 if N_LAYERS < 2 else DROPOUT)

        self.out = nn.Linear(HIDDEN_DIM * 2 if BIDIRECTIONAL else HIDDEN_DIM, output_dim)

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, text):

        with torch.no_grad():
            embedded = self.bert(text)[0]

        _, hidden = self.rnn(embedded)

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        output = self.out(hidden)

        return output
