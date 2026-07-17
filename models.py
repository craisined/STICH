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
        self.resnet = nn.Sequential(
            GeneralConv1D(num_features, num_features),
            nn.InstanceNorm1d(num_features),
            nn.ReLU(),

            GeneralConv1D(num_features, num_features),
            nn.InstanceNorm1d(num_features))

    def forward(self, x):
        return self.resnet(x) + x

class Generator(nn.Module):

    initial_features = 32

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            GeneralConv1D(NUM_CHANNELS, self.initial_features),
            nn.ReLU(),

            GeneralConv1D(self.initial_features, self.initial_features * 2),
            nn.InstanceNorm1d(self.initial_features * 2),
            nn.ReLU(),
            
            GeneralConv1D(self.initial_features * 2, self.initial_features * 4),
            nn.InstanceNorm1d(self.initial_features * 4),
            nn.ReLU())
        
        self.transformer = nn.Sequential(
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4))
        
        self.decoder = nn.Sequential(
            GeneralDeconv1D(self.initial_features * 4, self.initial_features * 2),
            nn.InstanceNorm1d(self.initial_features * 2),
            nn.ReLU(),

            GeneralDeconv1D(self.initial_features * 2, self.initial_features),
            nn.InstanceNorm1d(self.initial_features),
            nn.ReLU(),

            GeneralConv1D(self.initial_features, 1),
            nn.InstanceNorm1d(1),
            nn.Tanh())

    def forward(self, x):
        return self.decoder(self.transformer(self.encoder(x)))

class GeneratorLoss(nn.Module):

    def __init__(self, discriminator, opposing_generator, cycle_consistency_factor=10):
        super().__init__()
        self.discriminator = discriminator
        self.opposing_generator = opposing_generator
        self.cycle_consistency_factor = cycle_consistency_factor
    
    def forward(self, x):
        gan_loss = torch.log(1 - self.discriminator(x))
        cycle_consistency_loss = self.cycle_consistency_factor * \
            torch.linalg.vector_norm(self.opposing_generator(self(x)) - x)
        return gan_loss + cycle_consistency_loss

    
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
