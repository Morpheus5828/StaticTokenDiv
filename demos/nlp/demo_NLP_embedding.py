""" This script is a demo script to run NLP using embedding layer method
..moduleauthor::Marius THORRE
"""

import torch
import torch.nn as nn
import os, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

module_path = os.path.abspath(os.path.join('../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import static_token_div.learning.learning as learning
from static_token_div.learning.models import NLP_embedding
from static_token_div.eval.eval_learning import compute_perplexity

corpus_path = os.path.join("../../resources/tlnl_tp1_data/alexandre_dumas/fusion.txt")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\t Using gpu: %s ' % torch.cuda.is_available())

def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = text.strip().split()
    return tokens


def create_vocab(tokens):
    vocab = {}
    for word in tokens:
        if word not in vocab.keys():
            vocab[word] = len(vocab)
    return vocab


def extract_data(data, k):
    all_input = []
    all_target = []
    for idx in range(len(data) - k):
        current_input = data[idx:idx + k]
        all_input.append(current_input)
        all_target.append(data[idx + k])
    return all_input, all_target

k = 5
batch_size = 128
nb_epoch = 5

if __name__ == "__main__":
    corpus = read_corpus(corpus_path)
    vocab = create_vocab(corpus)
    data = [vocab[word] for word in corpus]

    # 1. Extract input data
    print(f"\t 1. Extract input data")
    X_data, y_data = extract_data(data, k=k)
    x_train, x_test, y_train, y_test = train_test_split(torch.tensor(X_data), torch.tensor(y_data), test_size=0.2)

    # 2. Create model instance
    print(f"\t 2. Create model instance")
    model = NLP_embedding(
        k=k,
        vocab_size=len(vocab.keys())
    ).to(device)

    optimizer = torch.optim.Adam(
        lr=1e-4,
        params=model.parameters(),
        weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    # 3. Start training model
    print(f"\t 3. Start training model")
    history = learning.fit_NLP_SGNS(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        device=device,
        batch_size=batch_size,
        nb_epoch=nb_epoch
    )

    # 4. Compute perplexity
    print(f"\t 4. Compute perplexity")
    perplexity = compute_perplexity(
        x_test=x_test,
        y_test=y_test,
        model=model,
        device=device,
        batch_size=batch_size
    )

    # 5. Plot result
    print(f"\t 5. Save training loss")
    plt.plot(history["train_loss"], c="blue", label="Train loss")
    plt.plot(history["test_loss"], c="orange", label="Test loss")
    plt.ylabel("Perplexity")
    plt.xlabel("Epoch")
    plt.title(f"Training loss, perplexity: {perplexity:.2f}")
    plt.legend()
    plt.ylim(0, 5)
    plt.savefig("NLP_Embedding_perplexity_fusion.png")

    # 6. Save model
    print(f"\t 6. Save model at ./nlp_embedding_model_fusion")
    model = model.to('cpu')
    torch.save(model.state_dict(), "../../resources/nlp_embedding_model_fusion.pth")





