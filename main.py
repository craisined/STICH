from dataloader import DataLoaderLegacy
import logging
from models import DiscriminatorLoss, Generator, Discriminator, GeneratorLoss
from pathlib import Path
import sys
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from plotter import Plotter

# Logging
log_handler = logging.FileHandler("model.log", mode="a", encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[log_handler]
)
logger = logging.getLogger(__name__)

# GPU
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Datasets
music_dataset = HummingClassicalDataset(
    humming_dir=Path("data") / "humtrans_processed",
    classical_dir=Path("data") / "musicnet_processed"
)
disc_sampler = DistributedSampler(music_dataset, shuffle=True)
disc_dataloader = DataLoader(
    music_dataset, 
    batch_size=4 * torch.cuda.device_count(),
    pin_memory=True, 
    sampler=sampler
)
gen_sampler = DistributedSampler(music_dataset, shuffle=True)
gen_dataloader = DataLoader(
    music_dataset, 
    batch_size=4 * torch.cuda.device_count(),
    pin_memory=True, 
    sampler=sampler
)

# Models
classical_to_humming_gen, humming_to_classical_gen = Generator(), Generator()
classical_disc, humming_disc = Discriminator(), Discriminator()
classical_to_humming_gen.to(local_rank)
humming_to_classical_gen.to(local_rank)
classical_disc.to(local_rank)
humming_disc.to(local_rank)

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
plotter = Plotter()
classical_disc_loss_history = []
humming_disc_loss_history = []
humming_to_classical_gen_loss_history = []
classical_to_humming_gen_loss_history = []

epochs = 10
for epoch in range(epochs):

    epoch_classical_disc_loss_history = []
    epoch_humming_disc_loss_history = []
    epoch_humming_to_classical_gen_loss_history = []
    epoch_classical_to_humming_gen_loss_history = []

    for ((humming_disc_data, classical_disc_data), (humming_gen_data, classical_gen_data)) in zip(disc_dataloader, gen_dataloader):   

        classical_disc_optim.zero_grad()
        humming_disc_optim.zero_grad()

        # Train discriminator with fake data
        classical_output = humming_to_classical_gen(humming_disc_data).detach()
        classical_probs = classical_disc(classical_output)
        classical_loss_val = classical_disc_loss(classical_probs, torch.zeros_like(classical_probs))
        logger.info(f"Loss for classical discriminator (fake): {classical_loss_val}")
        classical_loss_val.backward()

        humming_output = classical_to_humming_gen(classical_disc_data).detach()
        humming_probs = humming_disc(humming_output)
        humming_loss_val = humming_disc_loss(humming_probs, torch.zeros_like(humming_probs))
        logger.info(f"Loss for humming discriminator (fake): {humming_loss_val}")
        humming_loss_val.backward()

        # Train discriminator with real data
        classical_probs = classical_disc(classical_disc_data)
        classical_loss_val = classical_disc_loss(classical_probs, torch.ones_like(classical_probs))
        logger.info(f"Loss for classical discriminator (real): {classical_loss_val}")
        epoch_classical_disc_loss_history.append(classical_loss_val.item())
        classical_loss_val.backward()

        humming_probs = humming_disc(humming_disc_data)
        humming_loss_val = humming_disc_loss(humming_probs, torch.ones_like(humming_probs))
        logger.info(f"Loss for humming discriminator (real): {humming_loss_val}")
        epoch_humming_disc_loss_history.append(humming_loss_val.item())
        humming_loss_val.backward()

        classical_disc_optim.step()
        humming_disc_optim.step()

        # Train generators
        classical_to_humming_optim.zero_grad()
        humming_to_classical_optim.zero_grad()

        classical_output = humming_to_classical_gen(humming_gen_data)
        classical_loss_val = humming_to_classical_loss(classical_output, humming_gen_data)
        logger.info(f"Loss for classical generator: {classical_loss_val}")
        epoch_humming_to_classical_gen_loss_history.append(classical_loss_val.item())
        classical_loss_val.backward()

        humming_output = classical_to_humming_gen(classical_gen_data)
        humming_loss_val = classical_to_humming_loss(humming_output, classical_gen_data)
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

dist.destroy_process_group()