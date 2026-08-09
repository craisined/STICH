import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import weight_norm, spectral_norm
from website.main import *

# ---------------------------------------------------------------
# Setup & Training Loop
# ---------------------------------------------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Models
    gen_humming_to_classical = Generator().to(device)
    gen_classical_to_humming = Generator().to(device)

    disc_classical = Discriminator().to(device)
    disc_humming = Discriminator().to(device)

    # Initialize Loss Functions
    criterion_G = GeneratorLoss().to(device)
    criterion_D = DiscriminatorLoss().to(device)

    # Initialize Optimizers
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

    # Unified Dataset & DataLoader instantiation
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
            loss_D_classical_real = criterion_D(
                real_disc_classical, torch.ones_like(real_disc_classical)
            )
            loss_D_classical_fake = criterion_D(
                fake_disc_classical, torch.zeros_like(fake_disc_classical)
            )
            loss_D_classical = (loss_D_classical_real + loss_D_classical_fake) / 2

            # Humming Discriminator Loss
            real_disc_humming = disc_humming(humming)
            fake_disc_humming = disc_humming(fake_humming.detach())
            loss_D_humming_real = criterion_D(
                real_disc_humming, torch.ones_like(real_disc_humming)
            )
            loss_D_humming_fake = criterion_D(
                fake_disc_humming, torch.zeros_like(fake_disc_humming)
            )
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

        print(
            f"Epoch [{epoch + 1}/{epochs}] | Loss D: {total_loss_D.item():.4f} | Loss G: {total_loss_G.item():.4f}"
        )

    torch.save(gen_humming_to_classical.state_dict(), "gen_humming_to_classical.pth")
    torch.save(gen_classical_to_humming.state_dict(), "gen_classical_to_humming.pth")

    print("Generators saved successfully!")


if __name__ == "__main__":
    main()
