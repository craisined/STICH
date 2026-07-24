
from pathlib import Path
import numpy as np
from tqdm import tqdm

humming_dict = Path("data/humtrans_processed")
classical_dict = Path("data/musicnet_processed")


def compute_stats(input_dict):
    paths = sorted(input_dict.glob("*.npy"))
    assert len(paths) > 0, f"No .npy files found in {input_dict}"

    
    total = 0.0
    total_squared = 0.0
    count = 0

    for path in tqdm(paths, desc=f"Reading {input_dict.name}", unit=" files"):
        data = np.load(path).astype(np.float64)
        total += data.sum()
        total_squared += np.square(data).sum()
        count += data.size

    mean = total / count
    variance = total_squared / count - mean ** 2
    std = np.sqrt(max(variance, 0.0))  

    return mean, std, len(paths), count


for input_dict in (humming_dict, classical_dict):
    mean, std, file_count, sample_count = compute_stats(input_dict)
    print(f"{input_dict.name}: {file_count} files, {sample_count} samples")
    print(f"  mean = {mean:.8f}")
    print(f"  std  = {std:.8f}")
