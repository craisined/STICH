import numpy as np
import torch
import torch.nn as nn
from main import Generator, Discriminator
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "encoder.ts"
model = torch.jit.load(model_path, map_location=device)
model.eval()

# ---------------------------------------------------------------
# 2. Inference Function
# ---------------------------------------------------------------
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
    input_tensor = torch.from_numpy(input_array).float()

    # Add batch dimension if missing -> shape: (1, 16, sequence_length)
    if input_tensor.ndim == 2:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

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

    # Example: Translate a humming sample into classical music
    generated_humming = run_inference(
        input_path="sample_humming.npy",
        weights_path="gen_humming_to_classical.pth",
        device=device
    )

    """# Example: Translate a classical sample into humming
    generated_humming = run_inference(
        input_path="sample_classical.npy",
        weights_path="gen_classical_to_humming.pth",
        device=device
    )"""

    generated_humming = model.decode(torch.from_numpy(generated_humming).to(device).reshape(1, 16, 234)).cpu().detach().numpy().reshape(-1)
    sf.write("output/output.wav", generated_humming, 48000)
