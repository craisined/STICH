import argparse
from dataloader import HummingClassicalDataset, SAMPLE_RATE
import logging
from models import (
    DiscriminatorLoss,
    Generator,
    GeneratorLoss,
    MultiScaleDiscriminator,
    ReplayPool,
)
import os
from pathlib import Path
from plotter import Plotter

import torch
from torch.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    # Spectral cycle loss sits around 8 at init where waveform L1 sat around
    # 0.3, so this is not comparable to the old default. Sweep 2/5/10.
    parser.add_argument("--cycle-factor", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-seconds", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--iters-per-log", type=int, default=10)
    parser.add_argument("--disc-scales", type=int, default=3,
                        help="discriminators per domain, at 1x/2x/4x downsampling")
    parser.add_argument("--pool-size", type=int, default=50,
                        help="past fakes kept for the discriminators; 0 disables")
    # Safety net against a single pathological batch, not a regularizer.
    # Measured generator grad norms sit around 1200-1500, so this only engages
    # on genuine outliers; set it near the typical norm and you have quietly
    # cut the learning rate instead.
    parser.add_argument("--grad-clip", type=float, default=5000.0)
    # The *_1d folders hold raw waveforms. The plain *_processed folders hold
    # 2-D spectrograms on some branches, which this model does not take.
    parser.add_argument(
        "--humming-folder", type=str, default="data/humtrans_processed_1d"
    )
    parser.add_argument(
        "--classical-folder", type=str, default="data/musicnet_processed_1d"
    )
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
        crop_length=args.crop_seconds * SAMPLE_RATE,
    )

    sampler = DistributedSampler(music_dataset, shuffle=True)
    dataloader = DataLoader(
        music_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=sampler,
        num_workers=args.num_workers,
    )

    # Models
    classical_to_humming_gen = Generator().to(local_rank)
    humming_to_classical_gen = Generator().to(local_rank)
    classical_disc = MultiScaleDiscriminator(args.disc_scales).to(local_rank)
    humming_disc = MultiScaleDiscriminator(args.disc_scales).to(local_rank)

    classical_to_humming_gen = DDP(classical_to_humming_gen, device_ids=[local_rank])
    humming_to_classical_gen = DDP(humming_to_classical_gen, device_ids=[local_rank])
    classical_disc = DDP(classical_disc, device_ids=[local_rank])
    humming_disc = DDP(humming_disc, device_ids=[local_rank])

    # Loss and optimizers
    lr = args.lr
    betas = (0.5, 0.999)

    # Do NOT call .to() on these. GeneratorLoss registers the discriminator and
    # the opposing generator as submodules, and both are already DDP-wrapped --
    # moving a DDP module after construction invalidates its gradient buckets
    # and segfaults. The STFT window follows the audio's device instead, see
    # STFTLoss.magnitude().
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

    classical_pool = ReplayPool(args.pool_size)
    humming_pool = ReplayPool(args.pool_size)

    # Hold the learning rate for the first half of training, then decay it
    # linearly to zero. Standard CycleGAN, and it matters for the final result:
    # a constant rate keeps both players moving and nothing ever settles.
    optimizers = [
        classical_to_humming_optim,
        humming_to_classical_optim,
        classical_disc_optim,
        humming_disc_optim,
    ]
    decay_start = args.epochs // 2

    def lr_scale(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / max(1, args.epochs - decay_start))

    schedulers = [optim.lr_scheduler.LambdaLR(o, lr_scale) for o in optimizers]

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

            # ---- Discriminators ----
            classical_disc_optim.zero_grad()
            humming_disc_optim.zero_grad()

            with autocast("cuda"):
                classical_output = humming_to_classical_gen(humming_data).detach()
                humming_output = classical_to_humming_gen(classical_data).detach()

                # Judge a mix of current and past fakes, not just the newest.
                classical_fakes = classical_pool.query(classical_output)
                humming_fakes = humming_pool.query(humming_output)

                classical_fake_loss = classical_disc_loss(
                    classical_disc(classical_fakes), is_real=False
                )
                classical_real_loss = classical_disc_loss(
                    classical_disc(classical_data), is_real=True
                )
                humming_fake_loss = humming_disc_loss(
                    humming_disc(humming_fakes), is_real=False
                )
                humming_real_loss = humming_disc_loss(
                    humming_disc(humming_data), is_real=True
                )
                disc_loss = (
                    classical_fake_loss + classical_real_loss
                    + humming_fake_loss + humming_real_loss
                )

                classical_fake_disc_loss = classical_fake_loss.item()
                classical_real_disc_loss = classical_real_loss.item()
                humming_fake_disc_loss = humming_fake_loss.item()
                humming_real_disc_loss = humming_real_loss.item()

            epoch_classical_disc_loss_history.append(
                (classical_real_disc_loss + classical_fake_disc_loss) / 2
            )
            epoch_humming_disc_loss_history.append(
                (humming_real_disc_loss + humming_fake_disc_loss) / 2
            )

            scaler.scale(disc_loss).backward()
            scaler.unscale_(classical_disc_optim)
            scaler.unscale_(humming_disc_optim)
            clip_grad_norm_(classical_disc.parameters(), args.grad_clip)
            clip_grad_norm_(humming_disc.parameters(), args.grad_clip)
            scaler.step(classical_disc_optim)
            scaler.step(humming_disc_optim)

            # ---- Generators ----
            # The generator step only needs the discriminators' verdict, so
            # freeze them and skip computing gradients we would throw away.
            classical_disc.requires_grad_(False)
            humming_disc.requires_grad_(False)

            classical_to_humming_optim.zero_grad()
            humming_to_classical_optim.zero_grad()

            # One backward for both generators. Each cycle term already flows
            # into the opposing generator, so splitting it into separate
            # backwards only gave DDP more chances to mis-sync.
            with autocast("cuda"):
                classical_output = humming_to_classical_gen(humming_data)
                humming_output = classical_to_humming_gen(classical_data)

                classical_loss_val = humming_to_classical_loss(
                    classical_output, humming_data
                )
                humming_loss_val = classical_to_humming_loss(
                    humming_output, classical_data
                )
                gen_loss = classical_loss_val + humming_loss_val

                classical_gen_loss = classical_loss_val.item()
                humming_gen_loss = humming_loss_val.item()

            epoch_humming_to_classical_gen_loss_history.append(classical_gen_loss)
            epoch_classical_to_humming_gen_loss_history.append(humming_gen_loss)

            scaler.scale(gen_loss).backward()
            scaler.unscale_(classical_to_humming_optim)
            scaler.unscale_(humming_to_classical_optim)
            clip_grad_norm_(classical_to_humming_gen.parameters(), args.grad_clip)
            clip_grad_norm_(humming_to_classical_gen.parameters(), args.grad_clip)
            scaler.step(classical_to_humming_optim)
            scaler.step(humming_to_classical_optim)
            scaler.update()

            classical_disc.requires_grad_(True)
            humming_disc.requires_grad_(True)

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

        for scheduler in schedulers:
            scheduler.step()

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
            # Under LSGAN a discriminator loss near 0 means it has won outright
            # and the generator is getting almost no gradient. Worth seeing at a
            # glance rather than inferring from the raw numbers.
            for name, value in (
                ("classical", classical_disc_loss_avg),
                ("humming", humming_disc_loss_avg),
            ):
                if value < 0.01:
                    logger.warning(
                        f"EPOCH {epoch}: {name} discriminator loss {value:.5f} is near "
                        f"zero -- it has outrun the generator (lr {schedulers[0].get_last_lr()[0]:.2e})"
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
                cpu_generator.load_state_dict(
                    humming_to_classical_gen.module.state_dict()
                )
                cpu_generator.eval()
                torch.onnx.export(
                    cpu_generator,
                    torch.randn(1, 1, 160_000),
                    "models/humming_to_classical.onnx",
                    input_names=["audio"],
                    output_names=["audio_out"],
                    dynamic_axes={"audio": {2: "samples"}, "audio_out": {2: "samples"}},
                    opset_version=17,
                    external_data=False,
                )

                cpu_generator = Generator()
                cpu_generator.load_state_dict(
                    classical_to_humming_gen.module.state_dict()
                )
                cpu_generator.eval()
                torch.onnx.export(
                    cpu_generator,
                    torch.randn(1, 1, 160_000),
                    "models/classical_to_humming.onnx",
                    input_names=["audio"],
                    output_names=["audio_out"],
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
