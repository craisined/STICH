import random
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

from config import device, model, humming_embedding, classical_embedding, MODEL_SR
from audio_utils import resample, process_audio

def generate_embedding(audio_data):
    if isinstance(audio_data, np.ndarray):
        audio_data = torch.from_numpy(audio_data)
    
    audio_data = audio_data.reshape(1, -1)
    audio_data = resample(audio_data)
    audio_data = audio_data.reshape(1, 1, -1).to(device)
    
    with torch.no_grad():
        return model.encode(audio_data)

def generate_class_embedding(folder, num_files=2000):
    vector_sum = 0
    count = 0
    files = [file for file in folder.iterdir()]
    random.shuffle(files)

    if num_files is None:
        num_files = len(files)

    for file in files[:num_files]:
        if count % 100 == 0:
            print(count)
        audio_data = torch.from_numpy(np.load(file)).reshape(1, -1)
        audio_data = resample(audio_data)
        audio_data = audio_data.reshape(1, 1, -1).to(device)
        with torch.no_grad():
            vector_sum += model.encode(audio_data)
        count += 1
    return vector_sum / count

def humming_to_classical_embedding(audio_embedding):
    return audio_embedding - humming_embedding + classical_embedding

def classical_to_humming_embedding(audio_embedding):
    return audio_embedding - classical_embedding + humming_embedding

def wav_to_wav(audio_data, to_humming=True):
    sf.write("output/original.wav", audio_data, 16000)
    original_embedding = generate_embedding(audio_data)
    original_norm = torch.linalg.norm(original_embedding)
    original_embedding /= original_norm
    
    output_embedding = (
        classical_to_humming_embedding(original_embedding) 
        if to_humming 
        else humming_to_classical_embedding(original_embedding)
    )
    
    output_embedding = original_norm * output_embedding / torch.linalg.norm(output_embedding)
    output_numpy = model.decode(output_embedding).detach().cpu().numpy().reshape(-1)
    
    output_numpy = process_audio(output_numpy, sample_rate=MODEL_SR)
    sf.write("output/output.wav", output_numpy, MODEL_SR)
    
    return output_numpy

def save_embeddings():
    hum_emb = generate_class_embedding(Path("data/humtrans_processed")).cpu()
    class_emb = generate_class_embedding(Path("data/musicnet_processed")).cpu()
    np.save("humming_embedding.npy", hum_emb)
    np.save("classical_embedding.npy", class_emb)