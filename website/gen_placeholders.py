"""
Generate placeholder example clips for the gallery (stdlib only).

These are synthesized stand-ins so the "Hear it" gallery has working audio
players before the real model exists. Replace the files in static/examples/
with real model output when a checkpoint is ready.

Run:
    ./.venv/bin/python gen_placeholders.py
"""
import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 16000
OUT_DIR = Path(__file__).parent / "static" / "examples"

C = 261.63  # C4


def note(freq, dur, harmonics=(1.0,), vibrato=0.0, amp=0.32):
    n = int(dur * SAMPLE_RATE)
    if freq <= 0:
        return [0.0] * n
    norm = sum(harmonics)
    out = []
    for i in range(n):
        t = i / SAMPLE_RATE
        f = freq * (1.0 + vibrato * math.sin(2 * math.pi * 5.0 * t))
        val = sum(a * math.sin(2 * math.pi * f * k * t)
                  for k, a in enumerate(harmonics, start=1)) / norm
        env = min(1.0, t / 0.02) * min(1.0, (dur - t) / 0.06)
        out.append(amp * env * val)
    return out


def humming(melody):
    """Plain single voice with vibrato — stands in for a hummed tune."""
    s = []
    for freq, dur in melody:
        s.extend(note(freq, dur, harmonics=(1.0, 0.15), vibrato=0.02))
    return s


def classical(melody):
    """Warmer, harmonically richer voice — stands in for the 'classical' side."""
    s = []
    for freq, dur in melody:
        s.extend(note(freq, dur, harmonics=(1.0, 0.5, 0.33, 0.2), vibrato=0.0))
    return s


def write_wav(name, samples):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        frames = bytearray()
        for x in samples:
            x = -1.0 if x < -1.0 else 1.0 if x > 1.0 else x
            frames += struct.pack("<h", int(x * 32767))
        w.writeframes(bytes(frames))
    print("wrote", path.relative_to(OUT_DIR.parent.parent))


# A few short melodies (freq, seconds). freq 0 == rest.
MELODIES = {
    1: [(C, 0.4), (C * 1.12, 0.4), (C * 1.26, 0.4), (C * 1.12, 0.4), (C, 0.6)],
    2: [(C * 1.5, 0.3), (C * 1.68, 0.3), (C * 2, 0.3), (C * 1.68, 0.3), (C * 1.5, 0.5)],
    3: [(C * 0.75, 0.6), (C * 0.84, 0.5), (C, 0.6), (C * 0.84, 0.7)],
    4: [(C, 0.35), (C * 1.26, 0.35), (C * 1.5, 0.35), (C * 2, 0.55)],
    5: [(C * 1.12, 0.45), (C, 0.45), (C * 0.9, 0.45), (C * 0.84, 0.6)],
}


def main():
    # Humming -> Classical pairs
    for i in (1, 2, 3):
        write_wav(f"humming_{i}.wav", humming(MELODIES[i]))
        write_wav(f"classical_{i}.wav", classical(MELODIES[i]))

    # Classical -> Humming pairs
    for i in (4, 5):
        write_wav(f"classical_{i}.wav", classical(MELODIES[i]))
        write_wav(f"humming_{i}.wav", humming(MELODIES[i]))

    # Cycle demo: hum -> classical -> hum (the "translate it back" idea)
    write_wav("cycle_hum_in.wav", humming(MELODIES[1]))
    write_wav("cycle_classical.wav", classical(MELODIES[1]))
    write_wav("cycle_hum_out.wav", humming(MELODIES[1]))

    print("\nDone. Placeholder examples are in static/examples/")


if __name__ == "__main__":
    main()
