import numpy as np
import torchaudio

from config import AUDIO_SR, MODEL_SR

def resample(arr):
    resampler = torchaudio.transforms.Resample(orig_freq=AUDIO_SR, new_freq=MODEL_SR)
    return resampler(arr)

def process_audio(audio_np, sample_rate=MODEL_SR, fade_duration=0.015):
    """Normalizes peak volume and applies fade-in/fade-out to stop clicks and silence."""
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