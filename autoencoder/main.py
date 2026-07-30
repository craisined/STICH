import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 1. Unpaired Dataset Class (Combines both domain sources)
class UnpairedMusicDataset(Dataset):
    def __init__(self, path_humming, path_classical):
        self.humming_data = torch.from_numpy(np.load(path_humming)).float()
        self.classical_data = torch.from_numpy(np.load(path_classical)).float()

        self.len_humming = len(self.humming_data)
        self.len_classical = len(self.classical_data)

    def __len__(self):
        # Epoch size matches the larger dataset
        return max(self.len_humming, self.len_classical)

    def __getitem__(self, idx):
        # Cycle through humming data using modulo if index exceeds length
        humming_sample = self.humming_data[idx % self.len_humming]

        # Draw a random classical sample to ensure unaligned/unpaired matching
        random_classical_idx = torch.randint(0, self.len_classical, (1,)).item()
        classical_sample = self.classical_data[random_classical_idx]

        return humming_sample, classical_sample


# 2. Generator Architecture
class ResNet(nn.Module):
    def __init__(self, channels=64, kernel_size=3, dilation=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding="same", dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding="same", dilation=dilation),
            nn.GroupNorm(8, channels),
        )

    def forward(self, inp):
        return self.network(inp) + inp


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=7, padding="same"),
            ResNet(dilation=1),
            nn.ReLU(),
            ResNet(dilation=2),
            nn.ReLU(),
            ResNet(dilation=4),
            nn.ReLU(),
            ResNet(dilation=1),
            nn.ReLU(),
            ResNet(dilation=2),
            nn.ReLU(),
            ResNet(dilation=4),
            nn.Conv1d(64, 16, kernel_size=7, padding="same"),
        )

    def forward(self, inp):
        return self.network(inp)

"""
# 3. Discriminator Architecture
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.network = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.pool(x).squeeze(-1)  # (Batch, 16)
        return self.network(x)
"""
# 3. Patch1D Discriminator (Preserves temporal structural cues)
class Discriminator(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# 4. Loss Functions
class GeneratorLoss(nn.Module):
    def __init__(self, cycle_consistency_factor=10):
        super().__init__()
        self.cycle_consistency_factor = cycle_consistency_factor
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def forward(self, generated_embedding, original, inverse_generator, discriminator):
        disc_result = discriminator(generated_embedding)
        gan_loss = self.bce(disc_result, torch.ones_like(disc_result))
        
        reconstructed = inverse_generator(generated_embedding)
        cycle_consistency = self.l1(reconstructed, original)
        
        return gan_loss + (self.cycle_consistency_factor * cycle_consistency)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, disc_result, expected):
        return self.bce(disc_result, expected)


# ---------------------------------------------------------------
# Setup & Training Loop
# ---------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Models
gen_humming_to_classical = Generator().to(device)
gen_classical_to_humming = Generator().to(device)

disc_classical = Discriminator().to(device)
disc_humming = Discriminator().to(device)

# Initialize Loss Functions
criterion_G = GeneratorLoss(cycle_consistency_factor=5).to(device)
criterion_D = DiscriminatorLoss().to(device)

# Initialize Optimizers
optimizer_G = optim.Adam(
    list(gen_humming_to_classical.parameters()) + list(gen_classical_to_humming.parameters()),
    lr=1e-4,
    betas=(0.5, 0.999)
)
optimizer_D = optim.Adam(
    list(disc_classical.parameters()) + list(disc_humming.parameters()),
    lr=1e-4,
    betas=(0.5, 0.999)
)

# Unified Dataset & DataLoader instantiation
dataset = UnpairedMusicDataset("humming.npy", "classical.npy")

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    pin_memory=True if device.type == "cuda" else False,
    num_workers=2
)

def main():
    epochs = 20

    for epoch in range(epochs):
        gen_humming_to_classical.train()
        gen_classical_to_humming.train()
        disc_classical.train()
        disc_humming.train()

        for i, (humming, classical) in enumerate(train_loader):
            humming = humming.to(device)
            classical = classical.to(device)

            # -----------------------------------------------------------
            # 1. Train Discriminators
            # -----------------------------------------------------------
            optimizer_D.zero_grad()

            fake_classical = gen_humming_to_classical(humming)
            fake_humming = gen_classical_to_humming(classical)

            # Classical Discriminator Loss
            real_disc_classical = disc_classical(classical)
            fake_disc_classical = disc_classical(fake_classical.detach())
            loss_D_classical_real = criterion_D(real_disc_classical, torch.ones_like(real_disc_classical))
            loss_D_classical_fake = criterion_D(fake_disc_classical, torch.zeros_like(fake_disc_classical))
            loss_D_classical = (loss_D_classical_real + loss_D_classical_fake) / 2

            # Humming Discriminator Loss
            real_disc_humming = disc_humming(humming)
            fake_disc_humming = disc_humming(fake_humming.detach())
            loss_D_humming_real = criterion_D(real_disc_humming, torch.ones_like(real_disc_humming))
            loss_D_humming_fake = criterion_D(fake_disc_humming, torch.zeros_like(fake_disc_humming))
            loss_D_humming = (loss_D_humming_real + loss_D_humming_fake) / 2

            # Total Discriminator Loss
            total_loss_D = loss_D_classical + loss_D_humming
            total_loss_D.backward()
            optimizer_D.step()

            # -----------------------------------------------------------
            # 2. Train Generators
            # -----------------------------------------------------------
            optimizer_G.zero_grad()

            loss_G_humming_to_classical = criterion_G(
                generated_embedding=fake_classical,
                original=humming,
                inverse_generator=gen_classical_to_humming,
                discriminator=disc_classical
            )

            loss_G_classical_to_humming = criterion_G(
                generated_embedding=fake_humming,
                original=classical,
                inverse_generator=gen_humming_to_classical,
                discriminator=disc_humming
            )

            total_loss_G = loss_G_humming_to_classical + loss_G_classical_to_humming
            total_loss_G.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch + 1}/{epochs}] | Loss D: {total_loss_D.item():.4f} | Loss G: {total_loss_G.item():.4f}")

    torch.save(gen_humming_to_classical.state_dict(), "gen_humming_to_classical.pth")
    torch.save(gen_classical_to_humming.state_dict(), "gen_classical_to_humming.pth")

    print("Generators saved successfully!")

if __name__=="__main__":
    main()
