from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Critic
import torch
import numpy as np
import torch.nn as nn
class criticnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(112,128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128,64)
        self.linear3 = nn.Linear(64,32)
        self.linear4 = nn.Linear(32,1)
        self.func = nn.Sequential(
            self.linear1,
            self.relu,
            self.linear2,
            self.relu,
            self.linear3,
            self.relu,
            self.linear4
        )
    def forward(self,input):
        data = torch.as_tensor(input,dtype=torch.float32,device='cuda:0')
        data = self.func(data)
        return data

# net = criticnet()
# data = torch.ones(128,112)
# print(net(data))