from models import Generator
from pathlib import Path

import torch
import torch.multiprocessing
import random
import dataloader
import soundfile as sf
import numpy as np

num_humming_samples = 14613
sr = 16000

humming_to_classical_gen = Generator()

weights_path = "model.pt"
weights = torch.load(weights_path, weights_only=True, map_location=torch.device('cpu'))
humming_to_classical_gen.load_state_dict(weights["humming_to_classical"])

for i in range(0, 10):
    input_file = Path(f"data/humtrans_processed/sample_{random.randint(0,  num_humming_samples)}.npy")
    humming_data = dataloader.load(input_file)

    humming_to_classical_gen.eval()
    output_data = humming_to_classical_gen(humming_data).detach().numpy()


    humming_np = np.load(input_file).astype(np.float32).reshape(-1, 1)
    output_np = output_data.reshape(-1, 1)

    sf.write(f"output/original_{i}.wav", humming_np, sr)
    sf.write(f"output/reconstructed_{i}.wav", output_np, sr)