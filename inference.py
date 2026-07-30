import numpy as np
import torch
import torch.nn as nn
import torchaudio
from train import Generator, Discriminator
import soundfile as sf
from pathlib import Path
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "encoder.ts"
model = torch.jit.load(model_path, map_location=device)
model.eval()
Path("output").mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 2. Inference Function
# ---------------------------------------------------------------
def resample(arr):
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=48000)
    return resampler(arr)

def run_inference(
    input_path: str,
    weights_path: str,
    device: torch.device
):
    # Load model weights
    generator = Generator().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    generator.load_state_dict(state_dict)
    
    # Set model to evaluation mode
    generator.eval()

    # Load input numpy file
    input_array = np.load(input_path)  # Expected shape: (16, sequence_length)
    input_tensor = torch.from_numpy(input_array).float().reshape(1, 1, -1)

    input_tensor = resample(input_tensor)
    input_tensor = model.encode(input_tensor.to(device))

    # Perform forward pass without computing gradients
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # Post-process tensor back to NumPy array
    output_array = output_tensor.squeeze(0).cpu().numpy()

    # Save translated feature output
    return output_array


# ---------------------------------------------------------------
# 3. Main Execution
# ---------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = [file for file in Path("data/humtrans_processed").iterdir()]
    for i in range(5):
        inp = random.choice(files)
        sf.write(f"output/original_humming_{i}.wav", np.load(inp), 16000)

        # Example: Translate a humming sample into classical music
        generated_humming = run_inference(
            input_path=inp,
            weights_path="gen_humming_to_classical.pth",
            device=device
        )

        generated_humming = model.decode(torch.from_numpy(generated_humming).to(device).reshape(1, 16, 234)).cpu().detach().numpy().reshape(-1)
        sf.write(f"output/generated_classical_{i}.wav", generated_humming, 48000)
    
    files = [file for file in Path("data/musicnet_processed").iterdir()]
    for i in range(0):
        inp = random.choice(files)
        sf.write(f"output/original_classical_{i}.wav", np.load(inp), 16000)

        # Example: Translate a classical sample into humming
        generated_humming = run_inference(
            input_path=inp,
            weights_path="gen_classical_to_humming.pth",
            device=device
        )
        generated_humming = model.decode(torch.from_numpy(generated_humming).to(device).reshape(1, 16, 234)).cpu().detach().numpy().reshape(-1)
        sf.write(f"output/generated_humming_{i}.wav", generated_humming, 48000)