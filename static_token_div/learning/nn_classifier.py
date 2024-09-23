"""This script is a classification learning program using neural network,
the goal is to predict if word context is positive or negative like word2vec
.module author::Marius THORRE
"""
import time
import torch
import torch.nn as nn
from torch.nn.functional import relu, sigmoid, softmax
from torch.utils.data import random_split, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from static_token_div.learning.preprocessing import extract_context, create_learning_data
import warnings

start = time.time()

print(f"\n\tStarting learning process")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\tUsing gpu: %s ' % torch.cuda.is_available())

train_context_path = "train.txt"
test_context_path = "test.txt"

positive_context, negative_context = extract_context(context_file_path=train_context_path)
X_train, y_train = create_learning_data(positive_context, negative_context)

positive_context, negative_context = extract_context(context_file_path=test_context_path)
X_test, y_test = create_learning_data(positive_context, negative_context)

print(f"\tTraining input shape: {X_train.shape}, training target shape: {y_train.shape}")
print(f"\tTraining input shape: {X_train.shape}, training target shape: {y_train.shape}")


class BinaryClassifier(nn.Module):
    def __init__(self, input_size, embed_size):
        super(BinaryClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, embed_size)
        self.linear2 = torch.nn.Linear(embed_size, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        return x


def loss(output, target_pos, negative_samples):
    positive_score = torch.matmul(output, target_pos.t())

    negative_score = torch.bmm(negative_samples, output.unsqueeze(2)).squeeze()

    pos_loss = torch.log(sigmoid(positive_score))
    neg_loss = torch.sum(torch.log(sigmoid(-negative_score)), dim=1)

    total_loss = -(pos_loss + neg_loss).mean()
    return total_loss

num_epochs = 10
batch_size = 32
learning_rate = 0.01

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).long()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).long()

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

embedding_size = 5
model = BinaryClassifier(input_size=2, embed_size=embedding_size)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.to(device)

loss_fn = nn.CrossEntropyLoss()

epochs = 300

loss_values = []
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for batch, (context, target) in enumerate(train_loader):
        context = context.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pred = model(context)
        loss = loss_fn(pred, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss/len(train_loader)
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}")

    loss_values.append(epoch_loss)

print("Training complete.")

end = time.time()
print(f"\n\tCreation process time: {end - start:.2f} s")
