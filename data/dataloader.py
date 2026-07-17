from pathlib import Path
import random

import numpy as np
import torch


class DataLoader:
   

    def __init__(self, dir_a, dir_b, shuffle=True): # put in the directories
        self.files_a = sorted(Path(dir_a).glob("*.npy"))
        self.files_b = sorted(Path(dir_b).glob("*.npy"))
        if not self.files_a:
            raise FileNotFoundError(f"No .npy files found in {dir_a}")
        if not self.files_b:
            raise FileNotFoundError(f"No .npy files found in {dir_b}")
        self.shuffle = shuffle
        

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))

    def load(self, path): # loads a file into a tensor
        array = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(array)
        return tensor.view(1, 1, -1) 

    def __iter__(self): # loads one pair per iter: this is what is used in the training loop
        order_a = list(range(len(self.files_a)))
        order_b = list(range(len(self.files_b)))
        if self.shuffle:
            random.shuffle(order_a)
            random.shuffle(order_b)

        for step in range(len(self)):
            # just use modulus on the numbers to loop back (fancy heh)
            # when the smaller dataset finishes, it will loop back to index 0
            a = self._load(self.files_a[order_a[step % len(order_a)]])
            b = self._load(self.files_b[order_b[step % len(order_b)]])
            yield a, b
