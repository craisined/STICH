import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(padding="same", kernel=25, in_channels=64, out_channels=64),
            nn.ReLU(),
            nn.Conv1d(padding="same", kernel=25, in_channels=64, out_channels=64)
        )

    def forward(self, inp):
        return self.network(inp) + inp


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            ResNet(),
            nn.ReLU(),
            ResNet(),
            nn.ReLU(),
            ResNet(),
            nn.ReLU(),
            ResNet(),
            nn.ReLU(),
            ResNet(),
            nn.ReLU(),
        )

    def forward(self, inp):
        return self.network(inp)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 243),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)