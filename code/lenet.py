import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.c5 = nn.Linear(16 * 5 * 5, 120)
        self.f6 = nn.Linear(120, 84)
        self.output_layer = nn.Linear(84, 10)

    def forward(self, x):
        temp = F.relu(self.c1(x))               # C1层 卷积层
        temp = F.max_pool2d(temp, 2)            # S2层 池化层（下采样层）
        temp = F.relu(self.c3(temp))            # C3层 卷积层
        temp = F.max_pool2d(temp, 2)            # S4层 池化层（下采样层）
        temp = F.relu(self.c5(temp))            # C5层 卷积层
        temp = F.relu(self.f6(temp))            # F6层 全连接层
        out = self.output_layer(temp)           # output层 全连接层
        return out

