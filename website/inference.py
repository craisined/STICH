"""
Inference for the STICH demo.

Runs the exported ONNX generators with onnxruntime. Deliberately no torch and
no import of ../models.py: the .onnx files carry the graph and the weights, so
the server keeps working even as the training code changes shape.

Per request: decode -> redo the training preprocessing -> generator ->
normalize -> WAV bytes.

The .onnx files come from `main.py` (written to ../models/ at the end of
training) or from `export_onnx.py` run against a checkpoint. Re-export
whenever you retrain -- these weights are baked in.
"""
import io
from pathlib import Path
import wave

import librosa
import numpy as np
import onnxruntime

SAMPLE_RATE = 16000
CLIP_SAMPLES = 10 * SAMPLE_RATE  # every training clip was exactly this long
TRIM_TOP_DB = 40  # matches data/process_humtrans.py
OUTPUT_PEAK = 0.9  # target peak amplitude, leaving a little headroom

BASE = Path(__file__).parent
# Fresh exports land in <repo>/models; the copies alongside app.py are a
# fallback so the site still runs from a checkout with no training tree.
MODEL_DIRS = (BASE.parent / "models", BASE)

_sessions = {}


def convert(audio_bytes, direction="humming_to_classical", filename=""):
    """Translate an uploaded clip to the other domain. Returns WAV bytes."""
    session = _session(direction)
    audio = _preprocess(audio_bytes, direction)
    generated = session.run(None, {"audio": audio})[0]
    return _to_wav_bytes(_postprocess(generated))


def _session(direction):
    """Load (once) and return the onnxruntime session for one direction."""
    if direction not in _sessions:
        for directory in MODEL_DIRS:
            path = directory / f"{direction}.onnx"
            if path.exists():
                break
        else:
            searched = " or ".join(str(d) for d in MODEL_DIRS)
            raise FileNotFoundError(
                f"{direction}.onnx not found in {searched}. Train with main.py, "
                f"or run export_onnx.py against a checkpoint."
            )
        _sessions[direction] = onnxruntime.InferenceSession(str(path))
    return _sessions[direction]


def _preprocess(audio_bytes, direction):
    """Decode to the exact shape the generator saw during training: (1, 1, N)."""
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
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

    return audio.astype(np.float32).reshape(1, 1, -1)


def _postprocess(generated):
    """Flatten to mono and scale into WAV range."""
    audio = np.nan_to_num(np.asarray(generated).reshape(-1))
    # The generator ends in InstanceNorm1d (models.py) with no bounding
    # nonlinearity, so its output is not confined to [-1, 1] -- normalize
    # rather than let the 16-bit encoder clip the waveform flat.
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (OUTPUT_PEAK / peak)
    return audio


def _to_wav_bytes(samples, sr=SAMPLE_RATE):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
