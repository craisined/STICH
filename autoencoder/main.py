import torch
import torch.nn as nn
import torch.optim as optim


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(padding="same", kernel=3, in_channels=64, out_channels=64),
            nn.ReLU(),
            nn.Conv2d(padding="same", kernel=3, in_channels=64, out_channels=64),
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


class GeneratorLoss:
    def __init__(self, cycle_consistency_factor=10):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def forward(self, generated_embedding, original, inverse_generator, discriminator):
        disc_result = discriminator(generated_embedding)
        gan_loss = self.bce(disc_result, torch.ones_like(disc_result))
        cycle_consistency = self.l1(
            self.inverse_generator(generated_embedding), original
        )
        return gan_loss + self.cycle_consistency_factor * cycle_consistency


class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, disc_result, expected):
        return self.bce(disc_result, expected)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CycleGAN-style generators and discriminators
gen_humming_to_classical = Generator().to(device)
gen_classical_to_humming = Generator().to(device)

disc_classical = Discriminator().to(device)
disc_humming = Discriminator().to(device)

# Initialize Loss Functions
criterion_G = GeneratorLoss(cycle_consistency_factor=10).to(device)
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

# Placeholder DataLoaders (Replace with your PyTorch DataLoaders)
humming_dataloader = []
classical_dataloader = []

epochs = 10

for epoch in range(epochs):
    gen_humming_to_classical.train()
    gen_classical_to_humming.train()
    disc_classical.train()
    disc_humming.train()

    for i, (humming, classical) in enumerate(zip(humming_dataloader, classical_dataloader)):
        humming = humming.to(device)
        classical = classical.to(device)

        # -----------------------------------------------------------
        # 1. Train Discriminators
        # -----------------------------------------------------------
        optimizer_D.zero_grad()

        # Generate fake embeddings
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

        # Generator loss: Humming -> Classical -> Humming
        loss_G_humming_to_classical = criterion_G(
            generated_embedding=fake_classical,
            original=humming,
            inverse_generator=gen_classical_to_humming,
            discriminator=disc_classical
        )

        # Generator loss: Classical -> Humming -> Classical
        loss_G_classical_to_humming = criterion_G(
            generated_embedding=fake_humming,
            original=classical,
            inverse_generator=gen_humming_to_classical,
            discriminator=disc_humming
        )

        # Total Generator Loss
        total_loss_G = loss_G_humming_to_classical + loss_G_classical_to_humming
        total_loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch + 1}/{epochs}] | Loss D: {total_loss_D.item():.4f} | Loss G: {total_loss_G.item():.4f}")