import torch
import torchaudio
import soundfile as sf
import random

from pathlib import Path
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure output directory exists
Path("output").mkdir(exist_ok=True)

model_path = "encoder.ts"
model = torch.jit.load(model_path, map_location=device)
model.eval()

audio_sr = 16000
model_sr = 48000

def resample(arr):
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=48000)
    return resampler(arr)

def generate_embedding(audio_data):
    if isinstance(audio_data, np.ndarray):
        audio_data = torch.from_numpy(audio_data)
    
    audio_data = audio_data.reshape(1, -1)
    audio_data = resample(audio_data)
    audio_data = audio_data.reshape(1, 1, -1).to(device)
    
    with torch.no_grad():
        return model.encode(audio_data).cpu().numpy()

embedings = np.array([])
count = 0
for file in Path("data/musicnet_processed").iterdir():
    count += 1
    if count % 50 == 0:
        print(count)
    embedings = np.append(embedings, generate_embedding(np.load(file)))
np.save("cembeddings.npy", embedings)