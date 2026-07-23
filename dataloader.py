from pathlib import Path
import random
import numpy as np
import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class DataLoaderLegacy:

    def __init__(self, humming_dir, classical_dir, device=None):

        self.device = device
        self.humming_files = list(Path(humming_dir).glob("*.npy"))
        self.classical_files = list(Path(classical_dir).glob("*.npy"))
        self.humming_files_len = len(self.humming_files)
        self.classical_files_len = len(self.classical_files)

        assert self.humming_files_len > self.classical_files_len # not great practice, but who's gonna stop me!
        assert self.classical_files_len > 0

        self.reset()

    def _load(self, path):
        array = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(array) # TODO: check if shape is correct
        tensor = tensor.reshape((1, 1, -1))
        return tensor.to(self.device)
    
    def reset(self):
        random.shuffle(self.humming_files)
        random.shuffle(self.classical_files)
        self.humming_pos = 0 
        self.classical_pos = 0

    def pop(self): 
        humming = self.crop(self._load(self.humming_files[self.humming_pos]))
        classical = self.crop(self._load(self.classical_files[self.classical_pos]))

        self.humming_pos += 1
        self.classical_pos += 1

        if self.humming_pos >= self.humming_files_len:
            self.reset()
        elif self.classical_pos >= self.classical_files_len:
            random.shuffle(self.classical_files)
            self.classical_pos = 0

        return humming, classical
    
    def crop(self, sample):
        original_length = sample.shape[2]
        remainder = original_length % 4
        cropped_length = original_length - remainder
        cropped_sample = sample[:, :, :cropped_length]
        
        return cropped_sample

class HummingClassicalDataset(Dataset):

    def __init__(self, humming_dir, classical_dir):

        humming_files = list(Path(humming_dir).glob("*.npy"))
        classical_files = list(Path(classical_dir).glob("*.npy"))
        self.humming_files_len = len(humming_files)
        self.classical_files_len = len(classical_files)

        assert self.humming_files_len > self.classical_files_len # not great practice, but who's gonna stop me!
        assert self.classical_files_len > 0

        # Wrap in torch.stack() to collapse the list of tensors into one massive tensor.
        # This uses exactly 2 file descriptors for shared memory instead of 28,000.

        logger.info("Loading humming data into RAM. This may take a few minutes...")
        self.humming = torch.stack([self._load(path) for path in humming_files])
        logger.info("Loading classical data into RAM. This may take a few minutes...")
        self.classical = torch.stack([self._load(path) for path in classical_files])
        logger.info("All audio data successfully loaded and stacked into RAM.")

    def __len__(self):
        return self.humming_files_len

    def __getitem__(self, idx):
        # random.choice() acts unpredictably on stacked PyTorch tensors. 
        # Generate a random integer index instead.
        classical_idx = random.randint(0, self.classical_files_len - 1)
        return self.humming[idx], self.classical[classical_idx]

    def _load(self, path):
        array = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(array).reshape(1, -1)  
        return self._crop(tensor).contiguous()

    def _crop(self, sample):
        length = sample.shape[-1]
        remainder = length % 4  
        return sample[:, : length - remainder]
