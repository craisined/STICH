# Download and store https://huggingface.co/datasets/dadinghh2/HumTrans/blob/main/all_wav.zip in input_dict

from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm

input_dict = Path("data/humtrans")
output_dict = Path("data/humtrans_processed")
output_dict.mkdir(parents=True, exist_ok=True)

sampling_rate = 16000
file_number = 0
clip_seconds = 10
top_db = 40

for path in tqdm(input_dict.iterdir(), desc="Creating numpy arrays", unit=" files"):
    if not path.is_file() or not path.suffix == ".wav":
        continue
    data, sr = librosa.load(path, sr=sampling_rate)
    start_index = librosa.effects.trim(data, top_db=top_db)[1][0]
    data = data[start_index:sampling_rate * clip_seconds]
    data = np.pad(data, (0, sampling_rate * clip_seconds - len(data)), mode='constant', constant_values=0)
    np.save(output_dict / f"sample_{file_number}.npy", data)
    file_number += 1