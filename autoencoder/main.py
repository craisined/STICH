import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
import random

class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class UnpairedMusicDataset(Dataset):
    def __init__(self, path_humming, path_classical):
        self.humming_data = torch.from_numpy(np.load(path_humming)).float()
        self.classical_data = torch.from_numpy(np.load(path_classical)).float()

        self.len_humming = len(self.humming_data)
        self.len_classical = len(self.classical_data)

    def __len__(self):
        return max(self.len_humming, self.len_classical)

    def __getitem__(self, idx):
        humming_sample = self.humming_data[idx % self.len_humming]
        random_classical_idx = torch.randint(0, self.len_classical, (1,)).item()
        classical_sample = self.classical_data[random_classical_idx]
        return humming_sample, classical_sample


# ---------------------------------------------------------------
# FIX 2: InstanceNorm instead of GroupNorm for Style Transfer
# ---------------------------------------------------------------
class ResNet(nn.Module):
    def __init__(self, channels=64, kernel_size=3, dilation=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding="same", dilation=dilation),
            nn.InstanceNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding="same", dilation=dilation),
            nn.InstanceNorm1d(channels),
        )

    def forward(self, inp):
        return self.network(inp) + inp


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=7, padding="same"),
            ResNet(dilation=1),
            ResNet(dilation=2),
            ResNet(dilation=4),
            ResNet(dilation=8),
            ResNet(dilation=16),
            ResNet(dilation=32),
            nn.Conv1d(64, 16, kernel_size=7, padding="same"),
        )

    def forward(self, inp):
        return self.network(inp)


class Discriminator(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv1d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(128, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(self, x):
        return self.net(x)

class GeneratorLoss(nn.Module):
    def __init__(self, cycle_consistency_factor=7.0, identity_factor=4.0):
        super().__init__()
        self.cycle_consistency_factor = cycle_consistency_factor
        self.identity_factor = identity_factor
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def forward(
        self, generated_embedding, original, target, generator, inverse_generator, discriminator
    ):
        disc_result = discriminator(generated_embedding)
        gan_loss = self.bce(disc_result, torch.ones_like(disc_result))

        reconstructed = inverse_generator(generated_embedding)
        cycle_consistency = self.l1(reconstructed, original)

        identity = generator(target)
        identity_consistency = self.l1(identity, target)

        return (
            gan_loss
            + (self.cycle_consistency_factor * cycle_consistency)
            + (self.identity_factor * identity_consistency)
        )


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, real_result, fake_result):
        loss_real = self.bce(real_result, torch.ones_like(real_result))
        loss_fake = self.bce(fake_result, torch.zeros_like(fake_result))
        return (loss_real + loss_fake) * 0.5


# ---------------------------------------------------------------
# Setup & Training Loop
# ---------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_humming_to_classical = Generator().to(device)
    gen_classical_to_humming = Generator().to(device)
    disc_classical = Discriminator().to(device)
    disc_humming = Discriminator().to(device)

    criterion_G = GeneratorLoss().to(device)
    criterion_D = DiscriminatorLoss().to(device)

    optimizer_G = optim.Adam(
        list(gen_humming_to_classical.parameters())
        + list(gen_classical_to_humming.parameters()),
        lr=1e-4,
        betas=(0.5, 0.999),
    )
    optimizer_D = optim.Adam(
        list(disc_classical.parameters()) + list(disc_humming.parameters()),
        lr=1e-4,
        betas=(0.5, 0.999),
    )

    dataset = UnpairedMusicDataset("humming.npy", "classical.npy")
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        pin_memory=True if device.type == "cuda" else False,
        num_workers=2,
    )

    epochs = 30
    
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs)

    buffer_classical = ReplayBuffer()
    buffer_humming = ReplayBuffer()

    for epoch in range(epochs):
        gen_humming_to_classical.train()
        gen_classical_to_humming.train()
        disc_classical.train()
        disc_humming.train()

        for i, (humming, classical) in enumerate(train_loader):
            humming = humming.to(device)
            classical = classical.to(device)

            # -----------------------------------------------------------
            # 1. Train Generators (Train G FIRST in CycleGAN)
            # -----------------------------------------------------------
            optimizer_G.zero_grad()

            fake_classical = gen_humming_to_classical(humming)
            fake_humming = gen_classical_to_humming(classical)

            loss_G_humming_to_classical = criterion_G(
                generated_embedding=fake_classical,
                original=humming,
                target=classical,
                generator=gen_humming_to_classical,
                inverse_generator=gen_classical_to_humming,
                discriminator=disc_classical,
            )

            loss_G_classical_to_humming = criterion_G(
                generated_embedding=fake_humming,
                original=classical,
                target=humming,
                generator=gen_classical_to_humming,
                inverse_generator=gen_humming_to_classical,
                discriminator=disc_humming,
            )

            total_loss_G = loss_G_humming_to_classical + loss_G_classical_to_humming
            total_loss_G.backward()
            optimizer_G.step()

            # -----------------------------------------------------------
            # 2. Train Discriminators
            # -----------------------------------------------------------
            optimizer_D.zero_grad()

            # Push fakes to buffer and get a mix of current and old fakes
            buffered_fake_classical = buffer_classical.push_and_pop(fake_classical.detach())
            buffered_fake_humming = buffer_humming.push_and_pop(fake_humming.detach())

            loss_D_classical = criterion_D(
                disc_classical(classical), disc_classical(buffered_fake_classical)
            )
            loss_D_humming = criterion_D(
                disc_humming(humming), disc_humming(buffered_fake_humming)
            )

            total_loss_D = loss_D_classical + loss_D_humming
            total_loss_D.backward()
            optimizer_D.step()

        # Step the schedulers at the end of each epoch
        scheduler_G.step()
        scheduler_D.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}] | Loss D: {total_loss_D.item():.4f} | Loss G: {total_loss_G.item():.4f} | LR: {scheduler_G.get_last_lr()[0]:.6f}"
        )

    torch.save(gen_humming_to_classical.state_dict(), "gen_humming_to_classical.pth")
    torch.save(gen_classical_to_humming.state_dict(), "gen_classical_to_humming.pth")
    print("Generators saved successfully!")


if __name__ == "__main__":
    main()