import torch
import torch.nn as nn

NUM_CHANNELS = 1

class GeneralConv1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=25, stride=1, padding=None):
        super().__init__()
        kernel_size = kernel_size + (kernel_size - stride) % 2
        if padding is None:
            padding = (kernel_size - stride) // 2
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="zeros")

    def forward(self, x):
        conv = self.conv(x)
        return conv

class GeneralConv2D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding_mode="reflect") # TODO: padding for 2D

    def forward(self, x):
        conv = self.conv(x)
        return conv

class GeneralDeconv1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=25, stride=1, padding=None):
        super().__init__()
        kernel_size = kernel_size + (kernel_size - stride) % 2
        if padding is None:
            padding = (kernel_size - stride) // 2
        self.deconv = nn.ConvTranspose1d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="zeros") # TODO: Reflect?

    def forward(self, x):
        deconv = self.deconv(x)
        return deconv

class GeneralDeconv2D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding_mode="reflect") # TODO: padding for 2D

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
            nn.InstanceNorm1d(num_features)
        )

    def forward(self, x):
        resnet_result = self.resnet(x)
        return resnet_result + x

class Generator(nn.Module):

    initial_features = 32

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            GeneralConv1D(NUM_CHANNELS, self.initial_features, stride=2),
            nn.ReLU(),

            GeneralConv1D(self.initial_features, self.initial_features * 2, stride=2),
            nn.InstanceNorm1d(self.initial_features * 2),
            nn.ReLU(),
            
            GeneralConv1D(self.initial_features * 2, self.initial_features * 4),
            nn.InstanceNorm1d(self.initial_features * 4),
            nn.ReLU()
        )
        
        self.transformer = nn.Sequential(
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4),
            ResnetBlock(self.initial_features * 4))
        
        self.decoder = nn.Sequential(
            GeneralDeconv1D(self.initial_features * 4, self.initial_features * 2, stride=2),
            nn.InstanceNorm1d(self.initial_features * 2),
            nn.ReLU(),

            GeneralDeconv1D(self.initial_features * 2, self.initial_features, stride=2),
            nn.InstanceNorm1d(self.initial_features),
            nn.ReLU(),

            GeneralConv1D(self.initial_features, 1),
            # nn.InstanceNorm1d(1), this could be an issue for audio
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.transformer(self.encoder(x)))

class GeneratorLoss(nn.Module):

    def __init__(self, discriminator, opposing_generator, cycle_consistency_factor=10):
        super().__init__()
        self.discriminator = discriminator
        self.opposing_generator = opposing_generator
        self.cycle_consistency_factor = cycle_consistency_factor

        # self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.l1Loss = nn.L1Loss()
    
    def forward(self, x, original):
        disc_logits = self.discriminator(x)
        # gan_loss = self.bce(disc_logits, torch.ones_like(disc_logits))
        gan_loss = self.mse(disc_logits, torch.ones_like(disc_logits))
        cycle_consistency_loss = self.l1Loss(self.opposing_generator(x), original)
        return gan_loss + self.cycle_consistency_factor * cycle_consistency_loss

class Discriminator(nn.Module):

    initial_features = 64
    relu_factor = .2

    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            GeneralConv1D(NUM_CHANNELS, self.initial_features, kernel_size=25, stride=4),
            nn.LeakyReLU(self.relu_factor),
            
            GeneralConv1D(self.initial_features, self.initial_features * 2, kernel_size=25, stride=4),
            nn.InstanceNorm1d(self.initial_features * 2, affine=True),
            nn.LeakyReLU(self.relu_factor),
            
            GeneralConv1D(self.initial_features * 2, self.initial_features * 4, kernel_size=25, stride=4),
            nn.InstanceNorm1d(self.initial_features * 4, affine=True),
            nn.LeakyReLU(self.relu_factor),
            
            GeneralConv1D(self.initial_features * 4, self.initial_features * 8, kernel_size=25, stride=4, padding=0),
            nn.InstanceNorm1d(self.initial_features * 8, affine=True),
            nn.LeakyReLU(self.relu_factor),
            
            GeneralConv1D(self.initial_features * 8, 1, kernel_size=25, stride=1, padding=0)
        )

    def forward(self, x):
        return self.nn(x)

class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super().__init__()
        # self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
    
    def forward(self, x, original):
        # return self.bce(x, original)
        return self.mse(x, original)
