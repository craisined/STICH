import torch
import torch.nn as nn


NUM_CHANNELS = 1

class GeneralConv1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()
        self.pad = nn.ReflectionPad1D(1)
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=3)

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

    initial_features = 32

    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            # encoding
            GeneralConv1D(NUM_CHANNELS, self.initial_features),
            GeneralConv1D(self.initial_features, self.initial_features * 2),
            GeneralConv1D(self.initial_features * 2, self.initial_features * 4),

            # transformation
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),

            # decoding
        )

    def forward(self, x):
        x = self.nn(x)
        return x