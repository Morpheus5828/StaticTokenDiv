""" This script contains class to generate models for generate task
..moduleauthor::Marius THORRE
"""

import torch.nn as nn


class NLP_SGNS(nn.Module):
    def __init__(self, k, vocab_size, embed_dim=100):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(k*embed_dim, 256)

        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NLP_embedding(nn.Module):
    def __init__(self, k, vocab_size, embed_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(k * embed_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x