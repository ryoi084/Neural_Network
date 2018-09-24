import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# ニューラルネットワークを作成
# 活性化関数にsigmoidを使用
# ニューラル数は 2 -> 1 となっている
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(2,1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x


# トレーニングデータを作成
# AND回路(AND CIRCUIT)を再現したい
train_X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
train_Y = torch.tensor([[0.],[0.],[0.],[1.]])

test_X = train_X
test_Y = train_Y

# ニューラルネットワークのインスタンスを作成
network = Network()

# コスト関数は平均二乗誤差
criterion = nn.MSELoss()
# 最適化には確立勾配法を使用
# optimizer = optim.SGD(network.parameters(), lr = 10.)
optimizer = optim.Adam(network.parameters(), lr = 1.)

# 学習開始
for epoch in range(1000):
    running_loss = 0.0
    for data in zip(train_X,train_Y):
        inputs, labels = Variable(data[0]), Variable(data[1])

        # 勾配を0に初期化
        optimizer.zero_grad()

        # ラベルデータに対する誤差関数を計算
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

    print(f'Epoch: {epoch+1}, {running_loss*25:.4f}%')



for i, j in test_X:
    output = network.forward(torch.tensor([i,j]))
    print(f"{i} {j} -> {output.detach()}")


# 作ったニューラルネットワークのパラメータを保存
torch.save(network.state_dict(),'And_Circuit.pth')
