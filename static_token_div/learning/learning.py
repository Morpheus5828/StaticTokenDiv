""" This script contains code to fit model
..moduleauthor::Marius THORRE
"""

import torch
from torch.utils.data import TensorDataset, DataLoader


def fit_NLP_SGNS(
        model,
        criterion,
        optimizer,
        x_train,
        y_train,
        x_test,
        y_test,
        device,
        batch_size=64,
        nb_epoch=10
) -> dict:

    history = {
        "train_loss": [],
        "test_loss": []
    }

    num_samples = x_train.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(nb_epoch):
        model.train()
        total_loss = 0.0

        for batch_X, batch_Y in train_dataloader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)

                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        history["test_loss"].append(avg_test_loss)

        print(f'\t Epoch [{epoch+1}/{nb_epoch}], train loss: {avg_train_loss:.4f}, test loss: {avg_test_loss:.4f}')

    return history