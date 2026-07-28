import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import librosa.display
import matplotlib as plt

def create_spectrogram(y, sr=16000, n_fft=2048, hop_length=128, n_mels=128, device=None):
    # Automatically select GPU if available and no device is specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Convert numpy arrays to tensors if necessary and move to device
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    y = y.to(device)

    # Initialize transforms and move them to the target device
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    ).to(device)
    
    db_transform = T.AmplitudeToDB(stype='power').to(device)

    # Apply transforms
    mel_spec = mel_transform(y)
    mel_spec_db = db_transform(mel_spec)
    
    # Detach from graph (if needed), move to CPU, and convert to numpy array
    return mel_spec_db.detach().cpu().numpy()

def invert_spectrogram(mel_spec_db, sr=16000, n_fft=2048, hop_length=128, n_mels=128, n_iter=128, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if not isinstance(mel_spec_db, torch.Tensor):
        mel_spec_db = torch.tensor(mel_spec_db, dtype=torch.float32)
    mel_spec_db = mel_spec_db.to(device)

    # 1. Convert dB back to power scale
    mel_spec_power = F.DB_to_amplitude(mel_spec_db, ref=1.0, power=1.0)
    
    # 2. Invert the Mel Scale back to a linear STFT power spectrogram
    inverse_mel = T.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sr
    ).to(device)
    
    linear_spec_power = inverse_mel(mel_spec_power)
    
    # 3. Apply Griffin-Lim to estimate the phase and recover the waveform
    griffin_lim = T.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter,
        power=2.0
    ).to(device)
    
    y_reconstructed = griffin_lim(linear_spec_power)
    
    # Detach from graph (if needed), move to CPU, and convert to numpy array
    return y_reconstructed.detach().cpu().numpy()

def create_spectrogram_image(mel_spec_db, file_name, sr=16000, hop_length=128):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec_db, 
        sr=sr, 
        hop_length=hop_length,
        cmap='viridis'
    )
    plt.axis('off')
    plt.margins(0)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()