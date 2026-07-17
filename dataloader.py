from pathlib import Path
import random

import numpy as np
import torch


class DataLoader:

    def __init__(self, dir_a, dir_b): # put in the directories
        self.files_a = sorted(Path(dir_a).glob("*.npy"))
        self.files_b = sorted(Path(dir_b).glob("*.npy"))
        if not self.files_a:
            raise FileNotFoundError(f"No .npy files found in {dir_a}")
        if not self.files_b:
            raise FileNotFoundError(f"No .npy files found in {dir_b}")

        self.a_is_bigger = len(self.files_a) >= len(self.files_b)

        self.order_a = list(range(len(self.files_a)))
        self.order_b = list(range(len(self.files_b)))
        random.shuffle(self.order_a)
        random.shuffle(self.order_b)
        # keep track of indices
        self.pos_a = 0 
        self.pos_b = 0

    def _load(self, path): # loads a file into a tensor
        array = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(array)
        return tensor.view(1, 1, -1)
    
    def reset(self):
        random.shuffle(self.order_a)
        random.shuffle(self.order_b)
        self.pos_a = 0 
        self.pos_b = 0

    def pop(self): 
        a = self._load(self.files_a[self.order_a[self.pos_a]])
        b = self._load(self.files_b[self.order_b[self.pos_b]])

        # add to indices
        self.pos_a += 1
        self.pos_b += 1

        a_done = self.pos_a >= len(self.order_a)
        b_done = self.pos_b >= len(self.order_b)

        if (a_done and self.a_is_bigger) or (b_done and not self.a_is_bigger):
            # the bigger one finished: reshuffle both and start from the beginning
            random.shuffle(self.order_a)
            random.shuffle(self.order_b)
            self.pos_a = 0
            self.pos_b = 0
        else:
            # only the smaller one ran out: reshuffle only the smaller one, keep the bigger one going
            if a_done:
                random.shuffle(self.order_a)
                self.pos_a = 0
            if b_done:
                random.shuffle(self.order_b)
                self.pos_b = 0

        return a, b
