import torch
import torchaudio
import soundfile as sf

from pathlib import Path
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "musicnet.ts"
model = torch.jit.load(model_path, map_location=device)
model.eval()

audio_sr = 16000
model_sr = 48000

def generate_embedding(folder):
    vector_sum = 0
    count = 0
    for file in folder.iterdir():
        audio_data = torch.from_numpy(np.load(file)).reshape(1, -1)
        audio_data = resample(audio_data)
        audio_data = audio_data.reshape(1, 1, -1)
        with torch.no_grad():
            vector_sum += model.encode(audio_data)
        count += 1
    return vector_sum / count

humming_embedding = generate_embedding("data/humtrans_processed")
classical_embedding = generate_embedding("data/musicnet_processed")

def humming_to_classical_embedding(audio_embedding):
    return audio_embedding - humming_embedding + classical_embedding

def classical_to_humming_embedding(audio_embedding):
    return audio_embedding - classical_embedding + humming_embedding

def resample(arr):
    resampler = torchaudio.transforms.Resample(arr, orig_freq=16000, new_freq=48000)
    return resampler(arr)

def numpy_to_wav(arr):
    sf.write(f"output/original.wav", arr, audio_sr)
