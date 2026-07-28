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
min_seconds = 2  # must fill one training crop; see dataloader.CROP_SECONDS

def process_humming(path, sr):
    data, sr = librosa.load(path, sr=sr)
    # trim() returns (trimmed_audio, (start, end)). Using only start meant the
    # end bound stayed on the original timeline, so trimming the head shortened
    # the clip and the padding below made up the difference in digital silence.
    _, (start_index, end_index) = librosa.effects.trim(data, top_db=top_db)
    data = data[start_index:end_index][: sampling_rate * clip_seconds]

    if len(data) < sampling_rate * min_seconds:
        return None

    # Still padded to a fixed length so the dataset can stack into one tensor.
    # The dataloader tracks where the real content ends and crops only there.
    return np.pad(data, (0, sampling_rate * clip_seconds - len(data)),
                  mode='constant', constant_values=0)


for path in tqdm(input_dict.iterdir(), desc="Creating numpy arrays", unit=" files"):
    if not path.is_file() or not path.suffix == ".wav":
        continue

    data = process_humming(path, sampling_rate)
    if data is None:
        continue

    np.save(output_dict / f"sample_{file_number}.npy", data)
    file_number += 1