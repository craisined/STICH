from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class DataLoader:

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