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
CLIP_SAMPLES = 10 * SAMPLE_RATE  # longest clip we feed the generator
MIN_SAMPLES = SAMPLE_RATE  # below ~1 s there is not enough context to convert
TRIM_TOP_DB = 40  # matches data/process_humtrans.py
TARGET_RMS = 0.1  # matches dataloader.TARGET_RMS
PEAK_CEILING = 0.99  # matches dataloader.PEAK_CEILING
# models.TOTAL_STRIDE. Duplicated rather than imported: this module stays free
# of torch so the demo survives training-code changes (see module docstring).
TOTAL_STRIDE = 8
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
        # process_humtrans.py trims silence off both ends. No zero-padding here
        # either: the model never saw digital silence during training.
        _, (start, end) = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
        audio = audio[start:end]

    audio = audio[:CLIP_SAMPLES]

    # The encoder downsamples by TOTAL_STRIDE and the decoder undoes it exactly,
    # so a length off the grid comes back the wrong size.
    audio = audio[: len(audio) - len(audio) % TOTAL_STRIDE]

    if len(audio) < MIN_SAMPLES:
        raise ValueError("That clip is too short. Try at least a second of audio.")

    return _normalize(audio).astype(np.float32).reshape(1, 1, -1)


def _normalize(audio):
    """RMS normalize with a peak guard -- matches dataloader.normalize."""
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.abs(audio).max())
    if rms <= 1e-6:
        raise ValueError("That clip is silent.")
    return audio * min(TARGET_RMS / max(rms, 1e-8), PEAK_CEILING / max(peak, 1e-8))


def _postprocess(generated):
    """Flatten to mono and scale into WAV range."""
    audio = np.nan_to_num(np.asarray(generated).reshape(-1))
    # The generator ends in Tanh, so output is already inside [-1, 1] but
    # typically well short of full scale. Rescale so the demo is audible.
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
