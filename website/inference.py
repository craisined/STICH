"""
Inference for the STICH demo.

Runs the TorchScript autoencoder with style-embedding arithmetic -- the same
path as ../model_pipeline.py: encode the clip, then in embedding space step
away from the humming centroid and toward the classical one (or the reverse),
and decode the result.

Per request: decode -> redo the training preprocessing -> encode -> shift ->
decode -> normalize and fade -> WAV bytes.

Deliberately self-contained: importing ../config.py would load the model
relative to the process's working directory and create ./output at import
time, neither of which a server should depend on. The constants below mirror
../config.py and ../audio_utils.py -- keep them in step.

Needs encoder.ts plus humming_embedding.npy and classical_embedding.npy. The
centroids only mean anything in one encoder's latent space, so re-run
model_pipeline.save_embeddings() whenever the encoder is retrained.
"""
import io
from pathlib import Path
import threading
import wave

import librosa
import numpy as np
import torch

AUDIO_SR = 16000  # the rate every training clip was processed at
MODEL_SR = 48000  # the rate the autoencoder itself runs at
CLIP_SAMPLES = 10 * AUDIO_SR  # every training clip was exactly this long
TRIM_TOP_DB = 40  # matches data/process_humtrans.py
OUTPUT_PEAK = 0.95  # matches audio_utils.process_audio
FADE_SECONDS = 0.015  # ditto -- long enough to kill the edge clicks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE = Path(__file__).parent
# main.py leaves the encoder and the .npy centroids at the repo root; the
# copies alongside app.py are a fallback so the site still runs from a
# checkout with no training tree.
MODEL_DIRS = (BASE.parent, BASE.parent / "models", BASE)

# Which centroid to subtract and which to add, per direction the UI offers.
STYLE_SHIFT = {
    "humming_to_classical": ("humming", "classical"),
    "classical_to_humming": ("classical", "humming"),
}

_loaded = None
_load_lock = threading.Lock()


def convert(audio_bytes, direction="humming_to_classical", filename=""):
    """Translate an uploaded clip to the other domain. Returns WAV bytes."""
    model, centroids = _load()
    audio = _preprocess(audio_bytes, direction)
    source, target = STYLE_SHIFT[direction]

    with torch.no_grad():
        embedding = model.encode(audio)
        # The centroids are averages of unit-length encodings, so the shift
        # only means anything against an embedding scaled the same way. The
        # original norm goes back on afterwards to preserve the level.
        norm = torch.linalg.norm(embedding)
        shifted = embedding / norm - centroids[source] + centroids[target]
        shifted = norm * shifted / torch.linalg.norm(shifted)
        generated = model.decode(shifted)

    return to_wav_bytes(_postprocess(generated.cpu().numpy()))


def _load():
    """Load (once) the TorchScript encoder and both normalized centroids."""
    global _loaded
    with _load_lock:
        if _loaded is None:
            model = torch.jit.load(str(_find("encoder.ts")), map_location=DEVICE)
            model.eval()
            centroids = {
                name: _centroid(_find(f"{name}_embedding.npy"))
                for name in ("humming", "classical")
            }
            _loaded = (model, centroids)
    return _loaded


def _find(name):
    for directory in MODEL_DIRS:
        path = directory / name
        if path.exists():
            return path
    searched = " or ".join(str(d) for d in MODEL_DIRS)
    raise FileNotFoundError(
        f"{name} not found in {searched}. Export the encoder there, then run "
        f"model_pipeline.save_embeddings() for the two .npy centroids."
    )


def _centroid(path):
    embedding = torch.from_numpy(np.load(path)).to(DEVICE)
    return embedding / embedding.norm(dim=-1, keepdim=True)


def _preprocess(audio_bytes, direction):
    """Decode to the exact shape the encoder saw in training: (1, 1, N) at MODEL_SR."""
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=AUDIO_SR, mono=True)
    except Exception:
        # librosa raises a handful of backend-specific errors; none of them
        # say anything useful to someone who just picked the wrong file.
        raise ValueError("That file could not be read as audio. Try a .wav or .mp3.")
    if audio.size == 0:
        raise ValueError("That file contains no audio.")

    if direction == "humming_to_classical":
        # process_humtrans.py drops everything before the first sound, then
        # slices to CLIP_SAMPLES of the *original* timeline -- so a late start
        # means a shorter clip, not a shifted window. Kept identical here.
        start = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)[1][0]
        audio = audio[start:CLIP_SAMPLES]
    else:
        # process_musicnet.py takes plain 10 s slices, no trimming.
        audio = audio[:CLIP_SAMPLES]

    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))

    # Pad first, resample second: the centroids were averaged over encodings of
    # exactly CLIP_SAMPLES at AUDIO_SR, so the length has to match before the
    # rate change. audio_utils.py uses torchaudio for this step; librosa's soxr
    # is transparent at a clean 3x ratio and keeps torchaudio off the server.
    audio = librosa.resample(audio, orig_sr=AUDIO_SR, target_sr=MODEL_SR)
    return torch.from_numpy(audio.astype(np.float32)).reshape(1, 1, -1).to(DEVICE)


def _postprocess(generated):
    """Flatten to mono, scale into WAV range, fade the edges (audio_utils.process_audio)."""
    audio = np.nan_to_num(np.asarray(generated).reshape(-1))
    # The decoder's output is not confined to [-1, 1] -- normalize rather than
    # let the 16-bit encoder clip the waveform flat.
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (OUTPUT_PEAK / peak)

    # Decoded clips start and stop abruptly; without the fades the WAV clicks.
    fade = int(MODEL_SR * FADE_SECONDS)
    if len(audio) >= 2 * fade:
        audio[:fade] *= np.linspace(0.0, 1.0, fade)
        audio[-fade:] *= np.linspace(1.0, 0.0, fade)
    return audio


def to_wav_bytes(samples, sr=MODEL_SR):
    """Mono 16-bit PCM WAV. Public: gen_examples.py encodes source clips with it."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
