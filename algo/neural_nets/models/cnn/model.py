import torch
import torch.nn as nn

from algo.neural_nets.models.cnn.model_config import DROPOUT, KERNEL_NUM, FIXED_LENGTH, KERNEL_SIZE


class CNN(nn.Module):

    def __init__(self, padding_idx, vocab_size, embedding_dim, output_dim, pretrained_embeddings):
        super(CNN, self).__init__()
        self.dropout = nn.Dropout(DROPOUT)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.padding_idx = padding_idx
        self.embedding.weight.requires_grad = False

        self.conv = nn.ModuleList([nn.Conv2d(1, KERNEL_NUM, (i, self.embedding.embedding_dim)) for i in KERNEL_SIZE])
        self.maxpools = [nn.MaxPool2d((FIXED_LENGTH + 1 - i, 1)) for i in KERNEL_SIZE]
        self.fc = nn.Linear(len(KERNEL_SIZE) * KERNEL_NUM, output_dim)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [self.maxpools[i](torch.tanh(cov(x))).squeeze(3).squeeze(2) for i, cov in enumerate(self.conv)]  # B X Kn

        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        y = self.fc(x)
        return y
