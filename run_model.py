import argparse
from models import Generator
from pathlib import Path
from data import spectrogram

import torch
import torch.multiprocessing
import random
import dataloader
import soundfile as sf
import numpy as np

Path("output").mkdir(exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sr", type=int, default=16_000)
    parser.add_argument("--input-folder", default="data/musicnet_processed")
    parser.add_argument("--output-class", default="humming")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--model-path", default="model.pt")
    return parser.parse_args()

args = parse_args()

num_samples = sum(1 for _ in Path(args.input_folder).iterdir())
sr = args.sr

model = Generator()

weights_path = args.model_path
name_to_class = {
    "classical": "humming_to_classical",
    "humming": "classical_to_humming"
}
weights = torch.load(weights_path, weights_only=True, map_location=torch.device('cpu'))
model.load_state_dict(weights[name_to_class[args.output_class]])

for i in range(args.num_samples):
    input_file = Path(args.input_folder) / f"sample_{random.randint(0,  num_samples)}.npy"
    raw = torch.from_numpy(np.load(input_file).astype(np.float32)).reshape(1, -1)

    raw = dataloader.crop(raw).numpy()
    input_data, lo, hi = dataloader.normalize(raw)

    model.eval()
    output_data = model(input_data).detach()


    original_np = spectrogram.invert_spectrogram(np.load(input_file))
    output_np = dataloader.denormalize(spectrogram.invert_spectrogram(output_data), lo, hi)

    sf.write(f"output/original_{i}.wav", original_np, sr)
    sf.write(f"output/reconstructed_{i}.wav", output_np, sr)