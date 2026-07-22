from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class HummingClassialDataset(Dataset):

    def __init__(self, humming_dir, classical_dir, device=None):

        self.device = device
        self.humming_files = list(Path(humming_dir).glob("*.npy"))
        self.classical_files = list(Path(classical_dir).glob("*.npy"))
        self.humming_files_len = len(self.humming_files)
        self.classical_files_len = len(self.classical_files)

        assert self.humming_files_len > self.classical_files_len # not great practice, but who's gonna stop me!
        assert self.classical_files_len > 0

        self.loader = DataLoader(self, batch_size=1, shuffle=True)

        self.reset()

    def __len__(self):
        return self.humming_files_len

    def __getitem__(self, idx):
        humming = self._load(self.humming_files[idx])
        classical = self._load(random.choice(self.classical_files))
        return self._crop(humming), self._crop(classical)

    def _load(self, path):
        array = np.load(path).astype(np.float32)
        return torch.from_numpy(array).reshape(1, -1) 

    def _crop(self, sample):
        length = sample.shape[-1]
        remainder = length % 4  
        return sample[:, : length - remainder]

    def reset(self):
        # a fresh iterator reshuffles the humming order (shuffle=True)
        self._iterator = iter(self.loader)

    def pop(self):
        try:
            humming, classical = next(self._iterator)
        except StopIteration:
            self.reset()
            humming, classical = next(self._iterator)

        return humming.to(self.device), classical.to(self.device)
