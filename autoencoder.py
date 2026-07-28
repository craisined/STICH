import torch
import torchaudio
from pathlib import Path
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "musicnet.ts"
model = torch.jit.load(model_path, map_location=device)
model.eval()

def generate_embedding(folder):
    vector_sum = 0
    count = 0
    for file in folder.iterdir():
        audio_data = torch.from_numpy(np.load(file)).reshape(1, 1, -1)
        with torch.no_grad():
            vector_sum += model.encode(audio_data)
        count += 1
    return vector_sum / count

humming_embedding = generate_embedding()
classical_embedding = generate_embedding()

def humming_to_classical_embedding(audio_embedding):
    return audio_embedding - humming_embedding + classical_embedding

def classical_to_humming_embedding(audio_embedding):
    return audio_embedding - classical_embedding + humming_embedding

