from dataloader import DataLoader
import logging
from models import Generator, Discriminator, GeneratorLoss
from pathlib import Path
import sys
import torch
from torch import nn

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = 1
lr = 0.01

disc_dataloader = DataLoader(Path("data") / "humtrans_processed", Path("data") / "musicnet_processed", device)
gen_dataloader = DataLoader(Path("data") / "humtrans_processed", Path("data") / "musicnet_processed", device)

classical_to_humming_gen = Generator()
humming_to_classical_gen = Generator()
classical_disc = Discriminator()
humming_disc = Discriminator()

classical_to_humming_gen.to(device)
humming_to_classical_gen.to(device)
classical_disc.to(device)
humming_disc.to(device)

classical_to_humming_loss = GeneratorLoss(humming_disc, humming_to_classical_gen)
humming_to_classical_loss = GeneratorLoss(classical_disc, classical_to_humming_gen)
# classical_disc_loss = nn.BCEWithLogitsLoss()
# humming_disc_loss = nn.BCEWithLogitsLoss()
classical_disc_loss = nn.MSELoss()
humming_disc_loss = nn.MSELoss()

classical_to_humming_optim = torch.optim.SGD(classical_to_humming_gen.parameters(), lr=lr)
humming_to_classical_optim = torch.optim.SGD(humming_to_classical_gen.parameters(), lr=lr)
classical_disc_optim = torch.optim.SGD(classical_disc.parameters(), lr=lr)
humming_disc_optim = torch.optim.SGD(humming_disc.parameters(), lr=lr)

epochs = 10
dataset_length = 14000
for epoch in range(epochs):

    disc_dataloader.reset()
    gen_dataloader.reset()

    for train_loop in range(dataset_length):
        for disc_updates in range(k):
            
            classical_disc_optim.zero_grad()
            humming_disc_optim.zero_grad()

            humming_data, classical_data = disc_dataloader.pop()

            # Train discriminator with fake data
            classical_output = humming_to_classical_gen(humming_data)
            classical_output.detach()
            classical_logits = classical_disc(classical_output)
            classical_loss_val = classical_disc_loss(classical_logits, torch.zeros_like(classical_logits))
            classical_loss_val.backward()

            humming_output = classical_to_humming_gen(classical_data)
            humming_output.detach()
            humming_logits = humming_disc(humming_output)
            humming_loss_val = humming_disc_loss(humming_logits, torch.zeros_like(humming_logits))
            humming_loss_val.backward()

            logger.info(f"Loss for classical discriminator (fake): {classical_loss_val}")
            logger.info(f"Loss for humming discriminator (fake): {humming_loss_val}")

            # Train discriminator with real data
            classical_logits = classical_disc(classical_data)
            classical_loss_val = classical_disc_loss(classical_logits, torch.ones_like(classical_logits))
            classical_loss_val.backward()

            humming_logits = humming_disc(humming_data)
            humming_loss_val = humming_disc_loss(humming_logits, torch.ones_like(humming_logits))
            humming_loss_val.backward()

            classical_disc_optim.step()
            humming_disc_optim.step()

            logger.info(f"Loss for classical discriminator (real): {classical_loss_val}")
            logger.info(f"Loss for humming discriminator (real): {humming_loss_val}")

        # Train generators
        classical_to_humming_optim.zero_grad()
        humming_to_classical_optim.zero_grad()

        humming_data, classical_data = gen_dataloader.pop()

        classical_output = humming_to_classical_gen(humming_data)
        classical_loss_val = humming_to_classical_loss(classical_output, humming_data)
        classical_loss_val.backward()

        humming_output = classical_to_humming_gen(classical_data)
        humming_loss_val = classical_to_humming_loss(humming_output, classical_data)
        humming_loss_val.backward()

        classical_to_humming_optim.step()
        humming_to_classical_optim.step()

        logger.info(f"Loss for classical generator: {classical_loss_val}")
        logger.info(f"Loss for humming generator: {humming_loss_val}")