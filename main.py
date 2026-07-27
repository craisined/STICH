import argparse
from dataloader import HummingClassicalDataset
import logging
from models import DiscriminatorLoss, Generator, Discriminator, GeneratorLoss
import os
from pathlib import Path
from plotter import Plotter

import torch
from torch.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cycle-factor", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iters-per-log", type=int, default=10)
    parser.add_argument("--humming-folder", type=int, default="data/humtrans_processed")
    parser.add_argument("--classical-folder", type=int, default="data/musicnet_processed")
    return parser.parse_args()


def main():

    args = parse_args()

    # GPU & Threads
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    process_id = os.getppid()

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Logging
    if local_rank == 0:
        logger = logging.getLogger(__name__)
        log_handler = logging.FileHandler(
            f"logs/model_{process_id}.log", mode="w", encoding="utf-8"
        )
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[log_handler],
        )

    # Datasets
    music_dataset = HummingClassicalDataset(
        humming_dir=Path(args.humming_folder),
        classical_dir=Path(args.classical_folder),
    )

    sampler = DistributedSampler(music_dataset, shuffle=True)
    dataloader = DataLoader(
        music_dataset, batch_size=args.batch_size, pin_memory=True, sampler=sampler, num_workers=1
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
    lr = args.lr
    betas = (0.5, 0.999)

    classical_to_humming_loss = GeneratorLoss(
        humming_disc,
        humming_to_classical_gen,
        cycle_consistency_factor=args.cycle_factor,
    )
    humming_to_classical_loss = GeneratorLoss(
        classical_disc,
        classical_to_humming_gen,
        cycle_consistency_factor=args.cycle_factor,
    )
    classical_disc_loss, humming_disc_loss = DiscriminatorLoss(), DiscriminatorLoss()
    classical_to_humming_optim = optim.Adam(
        classical_to_humming_gen.parameters(), lr=lr, betas=betas
    )
    humming_to_classical_optim = optim.Adam(
        humming_to_classical_gen.parameters(), lr=lr, betas=betas
    )
    classical_disc_optim = optim.Adam(classical_disc.parameters(), lr=lr, betas=betas)
    humming_disc_optim = optim.Adam(humming_disc.parameters(), lr=lr, betas=betas)

    # Plotting
    if local_rank == 0:
        plotter = Plotter(process_id)
    classical_disc_loss_history = []
    humming_disc_loss_history = []
    humming_to_classical_gen_loss_history = []
    classical_to_humming_gen_loss_history = []

    # AMP: one shared scaler for all optimizers (loss-scaling keeps FP16 grads from underflowing)
    scaler = GradScaler("cuda")

    epochs = args.epochs
    iters_per_log = args.iters_per_log
    for epoch in range(epochs):

        sampler.set_epoch(epoch)

        epoch_classical_disc_loss_history = []
        epoch_humming_disc_loss_history = []
        epoch_humming_to_classical_gen_loss_history = []
        epoch_classical_to_humming_gen_loss_history = []

        for iteration, (humming_data, classical_data) in enumerate(dataloader):

            humming_data = humming_data.to(local_rank)
            classical_data = classical_data.to(local_rank)

            classical_disc_optim.zero_grad()
            humming_disc_optim.zero_grad()

            # Train discriminator with fake data
            with autocast("cuda"):
                classical_output = humming_to_classical_gen(humming_data).detach()
                classical_probs = classical_disc(classical_output)
                classical_loss_val = classical_disc_loss(
                    classical_probs, torch.zeros_like(classical_probs)
                )
                classical_fake_disc_loss = classical_loss_val.item()
            scaler.scale(classical_loss_val).backward()

            with autocast("cuda"):
                humming_output = classical_to_humming_gen(classical_data).detach()
                humming_probs = humming_disc(humming_output)
                humming_loss_val = humming_disc_loss(
                    humming_probs, torch.zeros_like(humming_probs)
                )
                humming_fake_disc_loss = humming_loss_val.item()
            scaler.scale(humming_loss_val).backward()

            # Train discriminator with real data
            with autocast("cuda"):
                classical_probs = classical_disc(classical_data)
                classical_loss_val = classical_disc_loss(
                    classical_probs, torch.ones_like(classical_probs)
                )
                classical_real_disc_loss = classical_loss_val.item()
            epoch_classical_disc_loss_history.append(
                (classical_real_disc_loss + classical_fake_disc_loss) / 2
            )
            scaler.scale(classical_loss_val).backward()

            with autocast("cuda"):
                humming_probs = humming_disc(humming_data)
                humming_loss_val = humming_disc_loss(
                    humming_probs, torch.ones_like(humming_probs)
                )
                humming_real_disc_loss = humming_loss_val.item()
            epoch_humming_disc_loss_history.append(
                (humming_real_disc_loss + humming_fake_disc_loss) / 2
            )
            scaler.scale(humming_loss_val).backward()

            scaler.step(classical_disc_optim)
            scaler.step(humming_disc_optim)

            # Train generators
            classical_to_humming_optim.zero_grad()
            humming_to_classical_optim.zero_grad()

            with autocast("cuda"):
                classical_output = humming_to_classical_gen(humming_data)
                classical_loss_val = humming_to_classical_loss(
                    classical_output, humming_data
                )
                classical_gen_loss = classical_loss_val.item()
            epoch_humming_to_classical_gen_loss_history.append(classical_gen_loss)
            scaler.scale(classical_loss_val).backward()

            with autocast("cuda"):
                humming_output = classical_to_humming_gen(classical_data)
                humming_loss_val = classical_to_humming_loss(
                    humming_output, classical_data
                )
                humming_gen_loss = humming_loss_val.item()
            epoch_classical_to_humming_gen_loss_history.append(humming_gen_loss)
            scaler.scale(humming_loss_val).backward()

            if local_rank == 0 and iteration % iters_per_log == 0:
                logger.info(
                    f"Loss for discriminators (fake): {classical_fake_disc_loss} (classical) | {humming_fake_disc_loss} (humming)"
                )
                logger.info(
                    f"Loss for discriminators (real): {classical_real_disc_loss} (classical) | {humming_real_disc_loss} (humming)"
                )
                logger.info(
                    f"Loss for generators: {classical_gen_loss} (classical) | {humming_gen_loss} (humming)"
                )

            scaler.step(classical_to_humming_optim)
            scaler.step(humming_to_classical_optim)
            scaler.update()

        classical_disc_loss_avg = sum(epoch_classical_disc_loss_history) / len(
            epoch_classical_disc_loss_history
        )
        humming_disc_loss_avg = sum(epoch_humming_disc_loss_history) / len(
            epoch_humming_disc_loss_history
        )
        humming_to_classical_gen_loss_avg = sum(
            epoch_humming_to_classical_gen_loss_history
        ) / len(epoch_humming_to_classical_gen_loss_history)
        classical_to_humming_gen_loss_avg = sum(
            epoch_classical_to_humming_gen_loss_history
        ) / len(epoch_classical_to_humming_gen_loss_history)

        if local_rank == 0:
            plotter.plotEpochLoss(
                epoch + 1,
                epoch_classical_disc_loss_history,
                epoch_humming_disc_loss_history,
                epoch_humming_to_classical_gen_loss_history,
                epoch_classical_to_humming_gen_loss_history,
            )
            logger.info(
                f"EPOCH {epoch}: {classical_disc_loss_avg} (classical disc) | {humming_disc_loss_avg} (humming disc) | {humming_to_classical_gen_loss_avg} (h -> c) | {classical_to_humming_gen_loss_avg} (c -> h)"
            )

            torch.save(
                {
                    "classical_to_humming": classical_to_humming_gen.module.state_dict(),
                    "humming_to_classical": humming_to_classical_gen.module.state_dict(),
                },
                f"models/cyclegan_{process_id}_{epoch}.pt",
            )
            
            try:
                
                
                cpu_generator = Generator()
                cpu_generator.load_state_dict(humming_to_classical_gen.module.state_dict())
                cpu_generator.eval()
                torch.onnx.export(
                    cpu_generator, torch.randn(1, 1, 160_000), "models/humming_to_classical.onnx",
                    input_names=["audio"], output_names=["audio_out"],
                    dynamic_axes={"audio": {2: "samples"}, "audio_out": {2: "samples"}},
                    opset_version=17,
                    external_data=False,
                )
                
                cpu_generator = Generator()
                cpu_generator.load_state_dict(classical_to_humming_gen.module.state_dict())
                cpu_generator.eval()
                torch.onnx.export(
                    cpu_generator, torch.randn(1, 1, 160_000), "models/classical_to_humming.onnx",
                    input_names=["audio"], output_names=["audio_out"],
                    dynamic_axes={"audio": {2: "samples"}, "audio_out": {2: "samples"}},
                    opset_version=17,
                    external_data=False,
                )
            except Exception:
                logger.exception(f"ONNX export failed after epoch {epoch}")

        classical_disc_loss_history.append(classical_disc_loss_avg)
        humming_disc_loss_history.append(humming_disc_loss_avg)
        humming_to_classical_gen_loss_history.append(humming_to_classical_gen_loss_avg)
        classical_to_humming_gen_loss_history.append(classical_to_humming_gen_loss_avg)

    if local_rank == 0:
        plotter.plotFullLoss(
            classical_disc_loss_history,
            humming_disc_loss_history,
            humming_to_classical_gen_loss_history,
            classical_to_humming_gen_loss_history,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
