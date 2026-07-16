import torch
import torch.nn as nn

num_channels = 1

class GeneralConv(nn.Module):
    pass

class ResnetBlock(nn.Module):
    pass

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential()

    def forward(self, x):
        x = self.nn(x)
        return x