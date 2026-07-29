class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(),
            nn.ReLU(),
            nn.Conv1d()
        )

    def forward(self, inp):
        return self.network(inp) + inp


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            ResNet(),
            ResNet(),
            ResNet(),
            ResNet(),
            ResNet(),
        )

    def forward(self):
        pass
