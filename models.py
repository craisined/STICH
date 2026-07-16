import torch
import torch.nn as nn


NUM_CHANNELS = 1

class GeneralConv1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()
        self.pad = nn.ReflectionPad1d(1)
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size)

    def forward(self, x):
        pad = self.pad(x)
        conv = self.conv(pad)
        return conv

class GeneralConv2D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size)

    def forward(self, x):
        pad = self.pad(x)
        conv = self.conv(pad)
        return conv

class ResnetBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv_1 = GeneralConv1D(num_features, num_features)
        self.conv_2 = GeneralConv1D(num_features, num_features)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        return conv_2 + x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential()

    def forward(self, x):
        x = self.nn(x)
        return x
    
class Discriminator(nn.Module):

    num_features = 64

    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            GeneralConv1D(NUM_CHANNELS, self.num_features),
            GeneralConv1D(self.num_features, self.num_features * 2),
            GeneralConv1D(self.num_features * 2, self.num_features * 4),
            GeneralConv1D(self.num_features * 4, self.num_features * 8),
            GeneralConv1D(self.num_features * 8, 1) # decision
        )

    def forward(self, x):
        x = self.nn(x)
        return x