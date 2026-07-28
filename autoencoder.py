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

def resample(arr):
    resampler = torchaudio.transforms.Resample(arr, orig_freq=16000, new_freq=48000)
    return resampler(arr)

def generate_embedding(audio_data):
    return model.encode(resample(audio_data))

def generate_class_embedding(folder):
    vector_sum = 0
    count = 0
    for file in folder.iterdir():
        audio_data = torch.from_numpy(np.load(file)).reshape(1, -1)
        audio_data = resample(audio_data)
        audio_data = audio_data.reshape(1, 1, -1)
        with torch.no_grad():
            vector_sum += model.encode(resample(audio_data))
        count += 1
    return vector_sum / count

humming_embedding = generate_class_embedding(Path("data/humtrans_processed"))
classical_embedding = generate_class_embedding(Path("data/musicnet_processed"))

def humming_to_classical_embedding(audio_embedding):
    return audio_embedding - humming_embedding + classical_embedding

def classical_to_humming_embedding(audio_embedding):
    return audio_embedding - classical_embedding + humming_embedding

def wav_to_wav(audio_data, to_humming=True):
    sf.write(f"output/original.wav", audio_data, 16000)
    original_embedding = generate_embedding(audio_data)
    output_embedding = classical_to_humming_embedding(original_embedding) if to_humming else humming_to_classical_embedding(original_embedding)
    output_numpy = model.decode(output_embedding).numpy().reshape(-1)
    sf.write(f"output/output.wav", output_numpy, 48000)

wav_to_wav("data/humtrans_processed/sample_0.npy")