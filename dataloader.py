from pathlib import Path
import random
import numpy as np
import logging
import torch
from torch.utils.data import Dataset

from models import TOTAL_STRIDE

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CROP_SECONDS = 2
TARGET_RMS = 0.1
PEAK_CEILING = 0.99


class HummingClassicalDataset(Dataset):

    def __init__(self, humming_dir, classical_dir, crop_length=CROP_SECONDS * SAMPLE_RATE):
        self.crop_length = crop_length

        # Wrap in torch.stack() to collapse the list of tensors into one massive tensor.
        # This uses exactly 2 file descriptors for shared memory instead of 28,000.
        self.humming, self.humming_lengths = self._load_dir(humming_dir, "humming")
        self.classical, self.classical_lengths = self._load_dir(classical_dir, "classical")
        logger.info("All audio data successfully loaded and stacked into RAM.")

        self.humming_files_len = len(self.humming_lengths)
        self.classical_files_len = len(self.classical_lengths)
        assert self.classical_files_len > 0

    def _load_dir(self, directory, name):
        paths = sorted(Path(directory).glob("*.npy"))
        assert paths, f"No .npy files found in {directory}"

        logger.info(f"Loading {name} data into RAM. This may take a few minutes...")
        clips, lengths = [], []
        for path in paths:
            array = np.load(path).astype(np.float32)
            length = content_length(array)
            # Anything too short to fill one crop would have to be zero-padded,
            # and padding is the tell we are trying to remove.
            if length < self.crop_length:
                continue
            clips.append(torch.from_numpy(array).reshape(1, -1))
            lengths.append(length)

        skipped = len(paths) - len(clips)
        if skipped:
            logger.info(f"Skipped {skipped}/{len(paths)} {name} clips shorter than "
                        f"{self.crop_length} samples.")
        assert clips, f"No {name} clips long enough for a {self.crop_length}-sample crop"

        return torch.stack(clips).share_memory_(), torch.tensor(lengths)

    def __len__(self):
        return self.humming_files_len

    def __getitem__(self, idx):
        # random.choice() acts unpredictably on stacked PyTorch tensors.
        # Generate a random integer index instead.
        classical_idx = random.randint(0, self.classical_files_len - 1)
        humming = self._random_crop(self.humming[idx], self.humming_lengths[idx])
        classical = self._random_crop(self.classical[classical_idx],
                                      self.classical_lengths[classical_idx])
        return normalize(humming)[0], normalize(classical)[0]

    def _random_crop(self, clip, length):
        # Crop only from the region with real content, never from the padding.
        latest_start = int(length) - self.crop_length
        start = random.randint(0, latest_start) if latest_start > 0 else 0
        return clip[:, start:start + self.crop_length]


def content_length(array):
    """Samples up to and including the last non-zero one.

    process_humtrans.py zero-pads every clip out to a fixed 10 s. Those runs of
    exact 0.0 are something a tanh generator can never produce, so a
    discriminator can spot real humming without listening to the music at all.
    """
    nonzero = np.nonzero(array)[0]
    return int(nonzero[-1]) + 1 if nonzero.size else 0


def crop(sample, multiple=TOTAL_STRIDE):
    length = sample.shape[-1]
    return sample[..., : length - length % multiple]


def normalize(sample, target_rms=TARGET_RMS):
    """Scale to a fixed RMS, backing off if that would clip.

    Min-max normalization gave every real clip an exact -1 and an exact +1,
    which no tanh output ever reproduces, and it let one spike sample set the
    gain for a whole clip.
    """
    rms = sample.pow(2).mean(dim=-1, keepdim=True).sqrt()
    peak = sample.abs().amax(dim=-1, keepdim=True)

    scale = target_rms / rms.clamp(min=1e-8)
    scale = torch.minimum(scale, PEAK_CEILING / peak.clamp(min=1e-8))
    scale = torch.where(rms > 1e-6, scale, torch.zeros_like(scale))
    return sample * scale, scale


def denormalize(sample, scale):
    return sample / scale.clamp(min=1e-8)
