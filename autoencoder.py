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
        return model.encode(audio_data)

def generate_class_embedding(folder):
    vector_sum = 0
    count = 0
    files = [file for file in folder.iterdir()]
    random.shuffle(files)

    for file in files[:2000]:
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

def process_audio(audio_np, sample_rate=48000, fade_duration=0.015):
    """Normalizes peak volume and applies fade-in/fade-out to stop clicks and silence."""
    # Prevent division by zero and scale values safely to [-1.0, 1.0]
    peak = np.max(np.abs(audio_np))
    if peak > 0:
        audio_np = audio_np / peak * 0.95  # leave a tiny headroom margin
        
    fade_samples = int(sample_rate * fade_duration)
    if len(audio_np) >= 2 * fade_samples:
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        audio_np[:fade_samples] *= fade_in
        audio_np[-fade_samples:] *= fade_out
        
    return audio_np

def wav_to_wav(audio_data, to_humming=True):
    sf.write("output/original.wav", audio_data, 16000)
    original_embedding = generate_embedding(audio_data)
    
    output_embedding = (
        classical_to_humming_embedding(original_embedding) 
        if to_humming 
        else humming_to_classical_embedding(original_embedding)
    )
    
    output_numpy = model.decode(output_embedding).detach().cpu().numpy().reshape(-1)
    
    # Normalize and smooth the output waveform
    output_numpy = process_audio(output_numpy, sample_rate=model_sr)
    
    sf.write("output/output.wav", output_numpy, 48000)
    
    return output_numpy

def save_embeddings():
    humming_embedding = generate_class_embedding(Path("data/humtrans_processed")).cpu()
    classical_embedding = generate_class_embedding(Path("data/musicnet_processed")).cpu()
    np.save("humming_embedding.npy", humming_embedding)
    np.save("classical_embedding.npy", classical_embedding)

#save_embeddings()

humming_embedding = torch.from_numpy(np.load("humming_embedding.npy")).to(device)
humming_embedding = humming_embedding / humming_embedding.norm(dim=-1, keepdim=True)

classical_embedding = torch.from_numpy(np.load("classical_embedding.npy")).to(device)
classical_embedding = classical_embedding / classical_embedding.norm(dim=-1, keepdim=True)

num_tests = 10
num_files = 12000

input_folder = "data/humtrans_processed"
input_is_humming = True

for i in range(num_tests):
    input_file = Path(input_folder) / f"sample_{random.randint(0, num_files)}.npy"
    input_numpy = np.load(input_file)

    output_numpy = wav_to_wav(input_numpy, not input_is_humming)