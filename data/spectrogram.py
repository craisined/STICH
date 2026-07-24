import librosa
import librosa.display
import matplotlib.pyplot as plt

def create_spectrogram(y, sr=16000, n_fft=2048, hop_length=128, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec_db = librosa.power_to_db(mel_spec)
    return mel_spec_db

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

def invert_spectrogram(mel_spec_db, sr=16000, n_fft=2048, hop_length=128, n_mels=128, n_iter=128):
    mel_spec_inverted = librosa.db_to_power(mel_spec_db)
    y_reconstructed = librosa.feature.inverse.mel_to_audio(
        mel_spec_inverted,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter
    )
    return y_reconstructed