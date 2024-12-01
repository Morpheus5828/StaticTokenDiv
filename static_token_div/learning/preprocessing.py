""" This script contains code to prepare data for learning model
..moduleauthor::Marius THORRE
"""

import os, sys
import torch
from sklearn.model_selection import train_test_split

module_path = os.path.abspath(os.path.join('../..'))

if module_path not in sys.path:
    sys.path.append(module_path)


def _read_corpus(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = text.strip().split()
    return tokens


def _process_data_with_corpus(
    vocab,
    tokens,
    k: int,
    blacklist: list = []
) -> tuple:
    X_data, y_data = [], []
    indices = [vocab.get_word_index(token) for token in tokens]

    blacklist_indices = {vocab.get_word_index(word) for word in blacklist if word in vocab.dico_voca}

    for i in range(k, len(indices)):
        x_indices = indices[i - k:i]

        if indices[i] in blacklist_indices:
            continue

        embeddings = [vocab.get_emb_torch(idx) for idx in x_indices]
        x_embedded = torch.cat(embeddings)
        X_data.append(x_embedded)
        y_data.append(indices[i])

    X_tensor = torch.stack(X_data)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
    return X_tensor, y_tensor


def get_training_data_for_sgns(
    vocab,
    file_path: str,
    k: int,
    blacklist: list = [],
):
    data = _read_corpus(file_path)
    print(f"\t Data length: {len(data)} tokens")

    x_data, y_data = _process_data_with_corpus(vocab, data, k=k, blacklist=blacklist)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    print(f"\t x_train shape: {x_train.shape} x_test.shape: {x_test.shape}")

    return x_train, y_train, x_test, y_test

def get_training_data():
    pass

