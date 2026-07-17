import torch
import torch.nn as nn

NUM_CHANNELS = 1

class GeneralConv1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, padding_mode="reflect")

    def forward(self, x):
        conv = self.conv(x)
        return conv

class GeneralConv2D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding_mode="reflect")

    def forward(self, x):
        conv = self.conv(x)
        return conv

class GeneralDeconv1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_features, out_features, kernel_size=kernel_size, padding_mode="reflect")

    def forward(self, x):
        deconv = self.deconv(x)
        return deconv

class GeneralDeconv2D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, padding_mode="reflect")

    def forward(self, x):
        deconv = self.deconv(x)
        return deconv

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
        self.encoder = nn.Sequential(
            GeneralConv1D(NUM_CHANNELS, self.initial_features),
            GeneralConv1D(self.initial_features, self.initial_features * 2),
            GeneralConv1D(self.initial_features * 2, self.initial_features * 4))
        
        self.transformer = nn.Sequential(
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4))
        
        self.decoder = nn.Sequential(
            GeneralDeconv1D(self.initial_features * 4, self.initial_features * 2),
            GeneralDeconv1D(self.initial_features * 2, self.initial_features),
            GeneralDeconv1D(self.initial_features, 1))

    def forward(self, x):
        return self.decoder(self.transformer(self.encoder(x)))
    
class Discriminator(nn.Module):

    initial_features = 64
    relu_factor = .2

    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            GeneralConv1D(NUM_CHANNELS, self.initial_features),
            nn.LeakyReLU(self.relu_factor),
            GeneralConv1D(self.initial_features, self.initial_features * 2),
            nn.LeakyReLU(self.relu_factor),
            GeneralConv1D(self.initial_features * 2, self.initial_features * 4),
            nn.LeakyReLU(self.relu_factor),
            GeneralConv1D(self.initial_features * 4, self.initial_features * 8),
            nn.LeakyReLU(self.relu_factor),
            GeneralConv1D(self.initial_features * 8, 1)
        )

    def forward(self, x):
        x = self.nn(x)
        return x
