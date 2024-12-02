""" This script is a demo script to run NLP using SGNS embedding method
..moduleauthor::Marius THORRE
"""

import torch
import torch.nn as nn
import os, sys
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

import static_token_div.learning.preprocessing as preprocessing
import static_token_div.learning.learning as learning
from static_token_div.learning.models import NLP_SGNS
from static_token_div.eval.eval_learning import compute_perplexity
import static_token_div.tools.vocab_tools as vocab_tools


corpus_path = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/fusion.txt")
embedding_path = os.path.join(project_path, "demos/nlp/fusion_emb_filename.txt")
vocab = vocab_tools.Vocab(emb_filename=embedding_path)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\t Using gpu: %s ' % torch.cuda.is_available())

k = 5
batch_size = 128
nb_epoch = 10

if __name__ == "__main__":

    # 1. Extract input data
    print(f"\t 1. Extract input data")
    x_train, y_train, x_test, y_test = preprocessing.get_training_data_for_sgns(
        vocab=vocab,
        file_path=corpus_path,
        k=k
    )

    # 2. Create model instance
    print(f"\t 2. Create model instance")
    model = NLP_SGNS(
        k=k,
        vocab_size=vocab.vocab_size + 1
    ).to(device)

    optimizer = torch.optim.Adam(
        lr=1e-4,
        params=model.parameters(),
        weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    # 3 Start training model
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

    # 4 Compute perplexity
    print(f"\t 4. Compute perplexity")
    perplexity = compute_perplexity(
        x_test=x_test,
        y_test=y_test,
        model=model,
        device=device,
        batch_size=batch_size
    )

    # 5 Plot result
    print(f"\t 5. Save training loss")
    plt.plot(history["train_loss"], c="blue", label="Train loss")
    plt.plot(history["test_loss"], c="orange", label="Test loss")
    plt.ylabel("Perplexity")
    plt.xlabel("Epoch")
    plt.title(f"Training loss, perplexity: {perplexity:.2f}")
    plt.legend()
    plt.ylim(0, 10)
    #plt.savefig("NLP_SGNS_perplexity_CMC.png")

    # 6 Save model
    print(f"\t 6. Save model at ./nlp_SGNS_model")
    model = model.to('cpu')
    torch.save(model.state_dict(), "nlp_sgns_fusion.pth")





