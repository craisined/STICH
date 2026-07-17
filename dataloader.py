from pathlib import Path
import random

import numpy as np
import torch


class DataLoader:

    def __init__(self, humming_dir, classical_dir):
        
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
        return tensor
    
    def reset(self):
        random.shuffle(self.humming_files)
        random.shuffle(self.classical_files)
        self.humming_pos = 0 
        self.classical_pos = 0

    def pop(self): 
        humming = self._load(self.humming_files[self.humming_pos])
        classical = self._load(self.classical_files[self.classical_pos])

        self.humming_pos += 1
        self.classical_pos += 1

        if self.humming_pos >= self.humming_files_len:
            self.reset()
        elif self.classical_pos >= self.classical_files_len:
            random.shuffle(self.classical_files_len)
            self.classical_pos = 0

        return humming, classical
