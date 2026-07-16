from pathlib import Path
import librosa

input_dict = "humtrans"
output_dict = "humtrans_processed"

sampling_rate = 16000
file_number = 0
for category, path in input_dict:
    if not path.is_file() or not path.suffix == ".wav":
        continue
    data, sr = librosa.load(path, sr=sampling_rate)
    data.save(f"sample_{file_number}", output_dict)
    file_number += 1