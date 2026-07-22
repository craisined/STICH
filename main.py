from dataloader import DataLoader
import logging
from models import DiscriminatorLoss, Generator, Discriminator, GeneratorLoss
from pathlib import Path
import sys
import torch
from torch import nn
from plotter import Plotter

log_handler = logging.FileHandler("model.log", mode="a", encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[log_handler]
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = 1
lr = 2e-4
betas = (0.5, 0.999)

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
classical_disc_loss = DiscriminatorLoss()
humming_disc_loss = DiscriminatorLoss()

classical_to_humming_optim = torch.optim.Adam(classical_to_humming_gen.parameters(), lr=lr, betas=betas)
humming_to_classical_optim = torch.optim.Adam(humming_to_classical_gen.parameters(), lr=lr, betas=betas)
classical_disc_optim = torch.optim.Adam(classical_disc.parameters(), lr=lr, betas=betas)
humming_disc_optim = torch.optim.Adam(humming_disc.parameters(), lr=lr, betas=betas)

plotter = Plotter()
classical_disc_loss_history = []
humming_disc_loss_history = []
humming_to_classical_gen_loss_history = []
classical_to_humming_gen_loss_history = []

epochs = 10
dataset_length = 14000
for epoch in range(epochs):

    disc_dataloader.reset()
    gen_dataloader.reset()
    
    epoch_classical_disc_loss_history = []
    epoch_humming_disc_loss_history = []
    epoch_humming_to_classical_gen_loss_history = []
    epoch_classical_to_humming_gen_loss_history = []

    for train_loop in range(dataset_length):
        for disc_updates in range(k):
            
            classical_disc_optim.zero_grad()
            humming_disc_optim.zero_grad()

            humming_data, classical_data = disc_dataloader.pop()

            # Train discriminator with fake data
            classical_output = humming_to_classical_gen(humming_data).detach()
            classical_logits = classical_disc(classical_output)
            classical_loss_val = classical_disc_loss(classical_logits, torch.zeros_like(classical_logits))
            logger.info(f"Loss for classical discriminator (fake): {classical_loss_val}")
            classical_loss_val.backward()

            humming_output = classical_to_humming_gen(classical_data).detach()
            humming_logits = humming_disc(humming_output)
            humming_loss_val = humming_disc_loss(humming_logits, torch.zeros_like(humming_logits))
            logger.info(f"Loss for humming discriminator (fake): {humming_loss_val}")
            humming_loss_val.backward()

            # Train discriminator with real data
            classical_logits = classical_disc(classical_data)
            classical_loss_val = classical_disc_loss(classical_logits, torch.ones_like(classical_logits))
            logger.info(f"Loss for classical discriminator (real): {classical_loss_val}")
            epoch_classical_disc_loss_history.append(classical_loss_val.item())
            classical_loss_val.backward()

            humming_logits = humming_disc(humming_data)
            humming_loss_val = humming_disc_loss(humming_logits, torch.ones_like(humming_logits))
            logger.info(f"Loss for humming discriminator (real): {humming_loss_val}")
            epoch_humming_disc_loss_history.append(humming_loss_val.item())
            humming_loss_val.backward()

            classical_disc_optim.step()
            humming_disc_optim.step()

        # Train generators
        classical_to_humming_optim.zero_grad()
        humming_to_classical_optim.zero_grad()

        humming_data, classical_data = gen_dataloader.pop()

        classical_output = humming_to_classical_gen(humming_data)
        classical_loss_val = humming_to_classical_loss(classical_output, humming_data)
        logger.info(f"Loss for classical generator: {classical_loss_val}")
        epoch_humming_to_classical_gen_loss_history.append(classical_loss_val.item())
        classical_loss_val.backward()

        humming_output = classical_to_humming_gen(classical_data)
        humming_loss_val = classical_to_humming_loss(humming_output, classical_data)
        logger.info(f"Loss for humming generator: {humming_loss_val}")
        epoch_classical_to_humming_gen_loss_history.append(humming_loss_val.item())
        humming_loss_val.backward()

        classical_to_humming_optim.step()
        humming_to_classical_optim.step()
        
    plotter.plotEpochLoss(
        epoch + 1, 
        epoch_classical_disc_loss_history, 
        epoch_humming_disc_loss_history, 
        epoch_humming_to_classical_gen_loss_history, 
        epoch_classical_to_humming_gen_loss_history
    )
    
    classical_disc_loss_history.append(sum(epoch_classical_disc_loss_history) / len(epoch_classical_disc_loss_history))
    humming_disc_loss_history.append(sum(epoch_humming_disc_loss_history) / len(epoch_humming_disc_loss_history))
    humming_to_classical_gen_loss_history.append(sum(epoch_humming_to_classical_gen_loss_history) / len(epoch_humming_to_classical_gen_loss_history))
    classical_to_humming_gen_loss_history.append(sum(epoch_classical_to_humming_gen_loss_history) / len(epoch_classical_to_humming_gen_loss_history))
    
plotter.plotFullLoss(
    classical_disc_loss_history,
    humming_disc_loss_history,
    humming_to_classical_gen_loss_history,
    classical_to_humming_gen_loss_history
)