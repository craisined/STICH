import torch
from pathlib import Path

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure output directory exists
Path("output").mkdir(exist_ok=True)

# Audio sample rates
AUDIO_SR = 16000
MODEL_SR = 48000

# Load TorchScript model
MODEL_PATH = "encoder.ts"
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()

# Load and normalize pre-computed style embeddings
humming_embedding = torch.from_numpy(np.load("humming_embedding.npy")).to(device)
humming_embedding = humming_embedding / humming_embedding.norm(dim=-1, keepdim=True)

classical_embedding = torch.from_numpy(np.load("classical_embedding.npy")).to(device)
classical_embedding = classical_embedding / classical_embedding.norm(dim=-1, keepdim=True)