# Download and store https://www.kaggle.com/datasets/imsparsh/musicnet-dataset/musicnet.npz

from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm

input_path = Path("data/musicnet/musicnet.npz")
output_dict = Path("data/musicnet_processed")
output_dict.mkdir(parents=True, exist_ok=True)

musicnet_sr = 44100
target_sr = 16000
clip_seconds = 20
clip_length = clip_seconds * target_sr    # samples per ~20s clip

dataset = np.load(input_path, allow_pickle=True, encoding="latin1")

file_number = 0
for recording_id in tqdm(dataset.files, desc="Creating numpy arrays", unit=" recordings"):
    audio, _labels = dataset[recording_id] # just not using labels, we could change this
    audio = librosa.resample(
        audio.astype(np.float32), orig_sr=musicnet_sr, target_sr=target_sr
    )
    
    num_clips = len(audio) // clip_length
    for i in range(num_clips):
        clip = audio[i * clip_length : (i + 1) * clip_length]
        np.save(output_dict / f"sample_{file_number}.npy", clip)
        file_number += 1
