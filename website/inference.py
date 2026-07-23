"""
Inference for the STICH demo.

>>> STATUS: DUMMY <<<
convert() currently ignores the uploaded audio and the trained model
entirely, and returns a synthesized placeholder clip so the whole
upload -> server -> download round-trip works end to end.

--- How to plug in the real model later ---
Replace the body of convert() with:

    1. Decode `audio_bytes` -> mono waveform at 16 kHz in [-1, 1]
       (librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)).
    2. Trim/pad to 10 s (160_000 samples) and crop to a multiple of 4,
       matching dataloader.py `_crop`.
    3. Reshape to (1, 1, N) and run through the appropriate generator:
         humming_to_classical -> humming_to_classical_gen
         classical_to_humming -> classical_to_humming_gen
       Load Generator() from ../models.py with a saved checkpoint,
       call .eval(), run on CPU (no DDP / no torch.distributed).
    4. Encode the output waveform (already in [-1, 1] via Tanh) back to
       WAV bytes (soundfile.write(buf, wav, 16000, format="WAV")).

NOTE: main.py does not yet save a checkpoint — that must be added first.
Only stdlib is used below, so the dummy stage needs no numpy/torch.
"""
import io
import math
import struct
import time
import wave

SAMPLE_RATE = 16000

# A short melody per direction so the two directions sound different and the
# result feels responsive. (freq_hz, seconds) pairs; freq 0 == a brief rest.
_C = 261.63  # helper note frequencies (C major-ish)
_MELODIES = {
    # "classical"-ish: an arpeggio with warm harmonics
    "humming_to_classical": [
        (_C, 0.32), (_C * 1.26, 0.32), (_C * 1.5, 0.32), (_C * 2, 0.5),
        (_C * 1.5, 0.32), (_C * 1.26, 0.32), (_C, 0.6),
    ],
    # "humming"-ish: a plain, wandering single voice with vibrato
    "classical_to_humming": [
        (_C * 0.75, 0.6), (_C * 0.84, 0.5), (_C * 0.75, 0.7), (_C * 0.67, 0.8),
    ],
}


def convert(audio_bytes, direction="humming_to_classical", filename=""):
    """Return WAV bytes. DUMMY: input is ignored, output is synthesized."""
    time.sleep(1.0)  # simulate model latency so the UI spinner is visible
    rich = direction == "humming_to_classical"
    harmonics = (1.0, 0.5, 0.33, 0.2) if rich else (1.0, 0.15)
    vibrato = 0.0 if rich else 0.02
    samples = []
    for freq, dur in _MELODIES.get(direction, _MELODIES["humming_to_classical"]):
        samples.extend(_note(freq, dur, harmonics=harmonics, vibrato=vibrato))
    return _to_wav_bytes(samples)


def _note(freq, dur, harmonics=(1.0,), vibrato=0.0, amp=0.32):
    """Synthesize one note as a list of float samples in [-1, 1]."""
    n = int(dur * SAMPLE_RATE)
    out = []
    if freq <= 0:  # rest
        return [0.0] * n
    norm = sum(harmonics)
    for i in range(n):
        t = i / SAMPLE_RATE
        f = freq * (1.0 + vibrato * math.sin(2 * math.pi * 5.0 * t))
        val = sum(a * math.sin(2 * math.pi * f * k * t)
                  for k, a in enumerate(harmonics, start=1)) / norm
        # simple attack/release envelope to avoid clicks
        env = min(1.0, t / 0.02) * min(1.0, (dur - t) / 0.06)
        out.append(amp * env * val)
    return out


def _to_wav_bytes(samples, sr=SAMPLE_RATE):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for s in samples:
            s = -1.0 if s < -1.0 else 1.0 if s > 1.0 else s
            frames += struct.pack("<h", int(s * 32767))
        w.writeframes(bytes(frames))
    return buf.getvalue()
