from dataloader import DataLoaderLegacy, HummingClassicalDataset
import logging
from models import DiscriminatorLoss, Generator, Discriminator, GeneratorLoss
import os
from pathlib import Path
from plotter import Plotter

import torch
from torch.amp import autocast, GradScaler
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# GPU & Threads
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Logging
if local_rank == 0:
    logger = logging.getLogger(__name__)
    log_handler = logging.FileHandler("model.log", mode="a", encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[log_handler]
    )

# Datasets
music_dataset = HummingClassicalDataset(
    humming_dir=Path("data") / "humtrans_processed",
    classical_dir=Path("data") / "musicnet_processed"
)

disc_sampler = DistributedSampler(music_dataset, shuffle=True)
disc_dataloader = DataLoader(
    music_dataset, 
    batch_size=4,
    pin_memory=True, 
    sampler=disc_sampler,
    num_workers=4
)
gen_sampler = DistributedSampler(music_dataset, shuffle=True)
gen_dataloader = DataLoader(
    music_dataset, 
    batch_size=4,
    pin_memory=True, 
    sampler=gen_sampler,
    num_workers=4
)

# Models
classical_to_humming_gen = Generator().to(local_rank)
humming_to_classical_gen = Generator().to(local_rank)
classical_disc = Discriminator().to(local_rank)
humming_disc = Discriminator().to(local_rank)

classical_to_humming_gen = DDP(classical_to_humming_gen, device_ids=[local_rank])
humming_to_classical_gen = DDP(humming_to_classical_gen, device_ids=[local_rank])
classical_disc = DDP(classical_disc, device_ids=[local_rank])
humming_disc = DDP(humming_disc, device_ids=[local_rank])

# Loss and optimizers
lr = 2e-4
betas = (0.5, 0.999)

classical_to_humming_loss, humming_to_classical_loss = GeneratorLoss(humming_disc, humming_to_classical_gen), GeneratorLoss(classical_disc, classical_to_humming_gen)
classical_disc_loss, humming_disc_loss = DiscriminatorLoss(), DiscriminatorLoss()
classical_to_humming_optim = optim.Adam(
    classical_to_humming_gen.parameters(),
    lr=lr,
    betas=betas
)
humming_to_classical_optim = optim.Adam(
    humming_to_classical_gen.parameters(),
    lr=lr,
    betas=betas
)
classical_disc_optim = optim.Adam(
    classical_disc.parameters(),
    lr=lr,
    betas=betas
)
humming_disc_optim = optim.Adam(
    humming_disc.parameters(),
    lr=lr,
    betas=betas
)

# Plotting
if local_rank == 0:
    plotter = Plotter()
classical_disc_loss_history = []
humming_disc_loss_history = []
humming_to_classical_gen_loss_history = []
classical_to_humming_gen_loss_history = []

# AMP: one shared scaler for all optimizers (loss-scaling keeps FP16 grads from underflowing)
scaler = GradScaler("cuda")

epochs = 10
for epoch in range(epochs):

    disc_sampler.set_epoch(epoch)
    gen_sampler.set_epoch(epoch)

    epoch_classical_disc_loss_history = []
    epoch_humming_disc_loss_history = []
    epoch_humming_to_classical_gen_loss_history = []
    epoch_classical_to_humming_gen_loss_history = []

    for ((humming_disc_data, classical_disc_data), (humming_gen_data, classical_gen_data)) in zip(disc_dataloader, gen_dataloader): 

        humming_disc_data = humming_disc_data.to(local_rank)
        classical_disc_data = classical_disc_data.to(local_rank)
        humming_gen_data = humming_gen_data.to(local_rank)
        classical_gen_data = classical_gen_data.to(local_rank)  

        classical_disc_optim.zero_grad()
        humming_disc_optim.zero_grad()

        # Train discriminator with fake data
        with autocast("cuda"):
            classical_output = humming_to_classical_gen(humming_disc_data).detach()
            classical_probs = classical_disc(classical_output)
            classical_loss_val = classical_disc_loss(classical_probs, torch.zeros_like(classical_probs))
        scaler.scale(classical_loss_val).backward()

        with autocast("cuda"):
            humming_output = classical_to_humming_gen(classical_disc_data).detach()
            humming_probs = humming_disc(humming_output)
            humming_loss_val = humming_disc_loss(humming_probs, torch.zeros_like(humming_probs))
        scaler.scale(humming_loss_val).backward()

        if local_rank == 0 and iteration % 50 == 0:
            logger.info(f"Loss for discriminators (fake): {classical_loss_val} (classical) | {humming_loss_val} (humming)")

        # Train discriminator with real data
        with autocast("cuda"):
            classical_probs = classical_disc(classical_disc_data)
            classical_loss_val = classical_disc_loss(classical_probs, torch.ones_like(classical_probs))
        epoch_classical_disc_loss_history.append(classical_loss_val.item())
        scaler.scale(classical_loss_val).backward()

        with autocast("cuda"):
            humming_probs = humming_disc(humming_disc_data)
            humming_loss_val = humming_disc_loss(humming_probs, torch.ones_like(humming_probs))
        epoch_humming_disc_loss_history.append(humming_loss_val.item())
        scaler.scale(humming_loss_val).backward()

        if local_rank == 0 and iteration % 50 == 0:
            logger.info(f"Loss for discriminators (real): {classical_loss_val} (classical) | {humming_loss_val} (humming)")

        scaler.step(classical_disc_optim)
        scaler.step(humming_disc_optim)

        # Train generators
        classical_to_humming_optim.zero_grad()
        humming_to_classical_optim.zero_grad()

        with autocast("cuda"):
            classical_output = humming_to_classical_gen(humming_gen_data)
            classical_loss_val = humming_to_classical_loss(classical_output, humming_gen_data)
        epoch_humming_to_classical_gen_loss_history.append(classical_loss_val.item())
        scaler.scale(classical_loss_val).backward()

        with autocast("cuda"):
            humming_output = classical_to_humming_gen(classical_gen_data)
            humming_loss_val = classical_to_humming_loss(humming_output, classical_gen_data)
        epoch_classical_to_humming_gen_loss_history.append(humming_loss_val.item())
        scaler.scale(humming_loss_val).backward()

        if local_rank == 0 and iteration % 50 == 0:
            logger.info(f"Loss for generators: {classical_loss_val} (classical) | {humming_loss_val} (humming)")

        scaler.step(classical_to_humming_optim)
        scaler.step(humming_to_classical_optim)
        scaler.update()
    
    if local_rank == 0:
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

if local_rank == 0:
    plotter.plotFullLoss(
        classical_disc_loss_history,
        humming_disc_loss_history,
        humming_to_classical_gen_loss_history,
        classical_to_humming_gen_loss_history
    )

dist.destroy_process_group()