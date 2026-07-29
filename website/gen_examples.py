"""
Generate the "Hear it" gallery clips from real model output.

Picks source clips out of the processed datasets, runs them through the same
`inference.convert()` the site serves to visitors, and writes the before/after
pairs `index.html` references into static/examples/. What the gallery plays is
then exactly what the "Try it yourself" box would produce for those clips.

Run (from the repo root, using the repo venv):
    ./.venv/bin/python website/gen_examples.py

Needs what inference.py needs -- encoder.ts plus the two centroid .npy files --
and the processed datasets in ../data/. Overwrites static/examples/ in place,
so listen to the results before committing them: these pairs are the first
thing a visitor hears, and a bad draw makes the model look worse than it is.
Re-roll the selection with --seed until a set sounds right, then record the
seed in the commit message.
"""
import argparse
import random
from pathlib import Path

import numpy as np

import inference

BASE = Path(__file__).parent
OUT_DIR = BASE / "static" / "examples"
HUMMING_DIR = BASE.parent / "data" / "humtrans_processed"
CLASSICAL_DIR = BASE.parent / "data" / "musicnet_processed"

# The slots the gallery in index.html expects: three humming -> classical
# pairs, two classical -> humming pairs, and a hum -> classical -> hum round
# trip. Changing these means changing the data-src attributes to match.
HUM_TO_CLS_SLOTS = (1, 2, 3)
CLS_TO_HUM_SLOTS = (4, 5)


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild static/examples/ from real model output."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="which source clips to draw; change it to re-roll the gallery",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    # One extra hum beyond the three pairs -- the round trip needs its own.
    hums = _pick(HUMMING_DIR, len(HUM_TO_CLS_SLOTS) + 1, rng)
    classicals = _pick(CLASSICAL_DIR, len(CLS_TO_HUM_SLOTS), rng)

    # Nothing touches disk until every clip has rendered. A missing model or a
    # short dataset part-way through would otherwise leave the gallery half
    # old and half new, which is harder to notice than an outright failure.
    clips = {}

    def add(name, wav_bytes, source):
        clips[name] = wav_bytes
        print(f"  {name:<20} <- {source.name}")

    print("humming -> classical")
    for slot, source in zip(HUM_TO_CLS_SLOTS, hums):
        clip = _as_upload(source)
        add(f"humming_{slot}.wav", clip, source)
        add(f"classical_{slot}.wav",
            inference.convert(clip, "humming_to_classical"), source)

    print("classical -> humming")
    for slot, source in zip(CLS_TO_HUM_SLOTS, classicals):
        clip = _as_upload(source)
        add(f"classical_{slot}.wav", clip, source)
        add(f"humming_{slot}.wav",
            inference.convert(clip, "classical_to_humming"), source)

    print("round trip")
    source = hums[-1]
    clip = _as_upload(source)
    # Feed step 2 back in as if it were an upload, so the return leg goes
    # through decoding and preprocessing the same way a visitor's would.
    there = inference.convert(clip, "humming_to_classical")
    back = inference.convert(there, "classical_to_humming")
    add("cycle_hum_in.wav", clip, source)
    add("cycle_classical.wav", there, source)
    add("cycle_hum_out.wav", back, source)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, wav_bytes in clips.items():
        (OUT_DIR / name).write_bytes(wav_bytes)

    print(f"\nWrote {len(clips)} clips (seed {args.seed}). Listen before committing.")
    print("If they hold up, drop the two 'placeholder' banners from index.html.")


def _pick(folder, count, rng):
    files = sorted(folder.glob("sample_*.npy"))
    if len(files) < count:
        raise FileNotFoundError(
            f"need {count} clips in {folder}, found {len(files)}. Run "
            f"data/process_humtrans.py and data/process_musicnet.py first."
        )
    return rng.sample(files, count)


def _as_upload(npy_path):
    """A processed clip encoded as a WAV -- what an upload of it would look like."""
    audio = np.load(npy_path)
    # Match the peak inference.py normalizes its output to, so comparing the
    # two players is about timbre rather than which one is louder.
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (inference.OUTPUT_PEAK / peak)
    return inference.to_wav_bytes(audio, sr=inference.AUDIO_SR)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        # Missing model or missing dataset -- both are setup problems, and the
        # messages already say which. A traceback would only bury them.
        raise SystemExit(f"\n{exc}")
