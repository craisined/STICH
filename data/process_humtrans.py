# Download and store https://huggingface.co/datasets/dadinghh2/HumTrans/blob/main/all_wav.zip in input_dict

from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm

input_dict = Path("humtrans")
output_dict = Path("humtrans_processed")
output_dict.mkdir(parents=True, exist_ok=True)

sampling_rate = 16000
file_number = 0
for path in tqdm(input_dict.iterdir(), desc="Creating numpy arrays", unit=" files"):
    if not path.is_file() or not path.suffix == ".wav":
        continue
    data, sr = librosa.load(path, sr=sampling_rate)
    np.save(output_dict / f"sample_{file_number}.npy", data)
    file_number += 1