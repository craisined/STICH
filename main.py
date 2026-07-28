import random
import numpy as np
from pathlib import Path

from model_pipeline import wav_to_wav, save_embeddings

if __name__ == "__main__":
    # Uncomment below to re-generate embeddings if needed
    # save_embeddings()

    num_tests = 10
    num_files = 12000

    input_folder = "data/humtrans_processed"
    input_is_humming = True

    for i in range(num_tests):
        input_file = Path(input_folder) / f"sample_{random.randint(0, num_files)}.npy"
        input_numpy = np.load(input_file)

        output_numpy = wav_to_wav(input_numpy, not input_is_humming)
        print(f"Processed test {i + 1}/{num_tests}")