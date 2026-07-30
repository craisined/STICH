"""
Inference for the STICH demo.

Runs the TorchScript autoencoder with a CycleGAN generator in between: encode
the clip to its latent embedding, translate that embedding with the generator
trained for the requested direction (see main.py), and decode the result.

Per request: decode -> redo the training preprocessing -> encode -> translate
-> decode -> normalize and fade -> WAV bytes.

Deliberately self-contained: importing ../config.py would load the model
relative to the process's working directory and create ./output at import
time, neither of which a server should depend on. The constants below mirror
../config.py and ../audio_utils.py -- keep them in step.

Needs encoder.ts plus the two generator checkpoints, gen_humming_to_classical
.pth and gen_classical_to_humming.pth. The generators only mean anything in
one encoder's latent space, so retrain them whenever the encoder is retrained.
"""
import importlib.util
import io
from pathlib import Path
import threading
import warnings
import wave

import librosa
import numpy as np
import torch
import torch.nn as nn

AUDIO_SR = 16000  # the rate every training clip was processed at
MODEL_SR = 48000  # the rate the autoencoder itself runs at
CLIP_SAMPLES = 10 * AUDIO_SR  # every training clip was exactly this long
TRIM_TOP_DB = 40  # matches data/process_humtrans.py
OUTPUT_PEAK = 0.95  # matches audio_utils.process_audio
FADE_SECONDS = 0.015  # ditto -- long enough to kill the edge clicks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE = Path(__file__).parent
# main.py leaves the encoder and the .pth checkpoints at the repo root; the
# copies alongside app.py are a fallback so the site still runs from a
# checkout with no training tree.
MODEL_DIRS = (BASE.parent, BASE.parent / "models", BASE)

# Which generator translates which way. Both were trained as one CycleGAN, so
# they are inverses of each other and the round trip in gen_examples.py works.
GENERATORS = {
    "humming_to_classical": "gen_humming_to_classical.pth",
    "classical_to_humming": "gen_classical_to_humming.pth",
}

_loaded = None
_load_lock = threading.Lock()


def convert(audio_bytes, direction="humming_to_classical", filename=""):
    """Translate an uploaded clip to the other domain. Returns WAV bytes."""
    model, generators = _load()
    audio = _preprocess(audio_bytes, direction)

    with torch.no_grad():
        # The generators were trained on raw encoder output, so the embedding
        # goes across unscaled -- no normalizing on the way in or out.
        embedding = model.encode(audio)
        translated = generators[direction](embedding)
        generated = model.decode(translated)

    return to_wav_bytes(_postprocess(generated.cpu().numpy()))


def _load():
    """Load (once) the TorchScript encoder and both trained generators."""
    global _loaded
    with _load_lock:
        if _loaded is None:
            model = torch.jit.load(str(_find("encoder.ts")), map_location=DEVICE)
            model.eval()
            generators = {
                direction: _generator(_find(checkpoint))
                for direction, checkpoint in GENERATORS.items()
            }
            _loaded = (model, generators)
    return _loaded


def _generator(path):
    """The generator a checkpoint was saved from, weights loaded, ready to run."""
    state = torch.load(str(path), map_location=DEVICE)
    generator = _build(state).to(DEVICE)
    try:
        generator.load_state_dict(state)
    except RuntimeError as exc:
        # A checkpoint from a training run whose Generator no longer matches
        # either architecture below. Loading it would be meaningless even if
        # it were forced, so say which file is stale rather than 500.
        raise RuntimeError(
            f"{path.name} does not match any known Generator -- it is from an "
            f"older or newer training run. Re-export it from the run that "
            f"produced the other checkpoint, or update main.py to match it."
        ) from exc
    generator.eval()
    return generator


def _build(state):
    """Pick the architecture a checkpoint's keys describe.

    The two directions are retrained often and land one file at a time, so the
    pair is regularly mid-swap. Choosing per checkpoint rather than assuming
    one architecture keeps the direction that is already current working while
    the other is still being re-exported.
    """
    if any(key.endswith("weight_g") for key in state):
        return WeightNormGenerator()
    return _architecture().Generator()


def _architecture():
    """main.py -- the training script the batch-norm checkpoints were saved from.

    Loaded by path rather than `import main`: the repo root holds a different
    main.py, and importing that one instead would both give the wrong module
    and load the encoder as a side effect of its imports.
    """
    spec = importlib.util.spec_from_file_location("stich_gan", BASE / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _conv(in_channels, out_channels, kernel_size=25):
    """Weight-normalized conv. The checkpoints store weight_g/weight_v, which is
    the pre-parametrization spelling, so the deprecated call is the correct one
    here -- the replacement writes different key names and would not load."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        )


class WeightNormResNet(nn.Module):
    """Pre-activation residual block: weight norm in place of main.py's batch norm."""

    def __init__(self, channels=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.LeakyReLU(),
            _conv(channels, channels),
            nn.LeakyReLU(),
            _conv(channels, channels),
        )

    def forward(self, inp):
        return self.network(inp) + inp


class WeightNormGenerator(nn.Module):
    """The newer generator: same 5-block layout as main.py's, weight-normalized.

    Reconstructed from the checkpoint's keys, which pin every shape but not the
    activations -- those carry no weights. LeakyReLU at the unparameterized
    slots matches main.py. Fold this back into main.py once the training script
    is synced, and delete it here.
    """

    def __init__(self, channels=64, latent=16):
        super().__init__()
        self.network = nn.Sequential(
            _conv(latent, channels),
            WeightNormResNet(channels),
            nn.LeakyReLU(),
            WeightNormResNet(channels),
            nn.LeakyReLU(),
            WeightNormResNet(channels),
            nn.LeakyReLU(),
            WeightNormResNet(channels),
            nn.LeakyReLU(),
            WeightNormResNet(channels),
            _conv(channels, latent),
        )

    def forward(self, inp):
        return self.network(inp)


def _find(name):
    for directory in MODEL_DIRS:
        path = directory / name
        if path.exists():
            return path
    searched = " or ".join(str(d) for d in MODEL_DIRS)
    raise FileNotFoundError(
        f"{name} not found in {searched}. Export the encoder there, then train "
        f"the generators with main.py for the two .pth checkpoints."
    )


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

    # Pad first, resample second: the generators were trained on encodings of
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
