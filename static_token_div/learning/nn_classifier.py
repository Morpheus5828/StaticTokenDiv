"""This script is a classification learning program using neural network,
the goal is to predict if word context is positive or negative
.module author::Marius THORRE
"""
# https://otmaneboughaba.com/posts/Word2Vec-in-Pytorch/
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class EmbeddingDataset(Dataset):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def __getitem__(self, index):
        context = train_data[:][0]
        target = train_data[:][1]

        return context, target


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, m, c_pos, c_neg):
        loss = -(
                torch.log(torch.sigmoid(torch.dot(m, c_pos))) +
                torch.sum(torch.log(torch.sigmoid(torch.dot(-m, c_neg))))
        )
        return loss


class ClassifierWord2Vec(nn.Module):
    def __init__(self, input_dim):
        super(ClassifierWord2Vec, self).__init__()
        self.l1 = nn.Linear(input_dim, 100)
        self.l2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


if __name__ == "__main__":
    #TODO train data, test data
    #TODO a configurer
    # train_data = torch.tensor(0)
    # test_data = torch.tensor(0)
    # input_dim = 0
    #
    # dataset = EmbeddingDataset(train_data=train_data, test_data=test_data)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    #
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # embedding_size = 5
    # model = ClassifierWord2Vec(input_dim=input_dim)
    # model.to(device)
    #
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # loss = Loss()
    #
    # epochs = 10
    #
    # for epoch in range(epochs):
    #     running_loss = 0
    #     for batch,

    pass
