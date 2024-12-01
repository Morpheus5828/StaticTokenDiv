""" This script contains code to evaluate models
..moduleauthor::Marius THORRE
"""


import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def compute_perplexity(
        x_test,
        y_test,
        model,
        device,
        batch_size
):
    test_dataset = TensorDataset(x_test, y_test)
    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_loss = 0.0
    total_words = 0

    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            outputs = model(batch_X)

            loss = criterion(outputs, batch_Y)

            total_loss += loss.item()
            total_words += batch_Y.size(0)

    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss)
    return perplexity