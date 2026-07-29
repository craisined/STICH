# Download and store https://www.kaggle.com/datasets/imsparsh/musicnet-dataset/musicnet.npz
#
# musicnet_metadata.csv ships with the same Kaggle dataset but is NOT inside the
# .npz -- download it separately to use --ensembles.

import argparse
import csv
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm

musicnet_sr = 44100
target_sr = 16000
clip_seconds = 10
clip_length = clip_seconds * target_sr    # samples per ~10s clip


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", type=Path, default=Path("data/musicnet/musicnet.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/musicnet_processed"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/musicnet/musicnet_metadata.csv")
    )
    parser.add_argument(
        "--ensembles",
        nargs="*",
        default=None,
        help="Case-insensitive substrings matched against the metadata 'ensemble' "
        "column, e.g. --ensembles 'Solo Violin' 'Solo Cello' 'Accompanied Violin'. "
        "Omit to keep every recording. Narrowing to one instrument family gives "
        "the discriminator a single coherent timbre to model instead of an "
        "average over piano, strings and winds.",
    )
    return parser.parse_args()


def selected_ids(metadata_path, ensembles):
    """Recording ids whose ensemble matches any of the requested substrings."""
    if not ensembles:
        return None

    if not metadata_path.exists():
        raise SystemExit(
            f"{metadata_path} not found. It is a separate file in the Kaggle "
            f"musicnet dataset, not part of musicnet.npz. Download it, or drop "
            f"--ensembles to process every recording."
        )

    wanted = [e.lower() for e in ensembles]
    keep, seen = set(), set()
    with open(metadata_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ensemble = (row.get("ensemble") or "").strip()
            seen.add(ensemble)
            if any(w in ensemble.lower() for w in wanted):
                keep.add(str(row["id"]).strip())

    if not keep:
        raise SystemExit(
            f"No recordings matched {ensembles}. Available ensembles:\n  "
            + "\n  ".join(sorted(seen))
        )
    print(f"Matched {len(keep)} recordings across ensembles {ensembles}")
    return keep


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = np.load(args.input, allow_pickle=True, encoding="latin1")
    keep = selected_ids(args.metadata, args.ensembles)

    recordings = [r for r in dataset.files if keep is None or str(r).strip() in keep]
    if not recordings:
        raise SystemExit("No recordings left to process after filtering.")

    file_number = 0
    for recording_id in tqdm(recordings, desc="Creating numpy arrays", unit=" recordings"):
        audio, _labels = dataset[recording_id]  # just not using labels, we could change this
        audio = librosa.resample(
            audio.astype(np.float32), orig_sr=musicnet_sr, target_sr=target_sr
        )

        num_clips = len(audio) // clip_length
        for i in range(num_clips):
            clip = audio[i * clip_length : (i + 1) * clip_length]
            np.save(args.output / f"sample_{file_number}.npy", clip)
            file_number += 1

    print(f"Wrote {file_number} clips from {len(recordings)} recordings to {args.output}")


if __name__ == "__main__":
    main()
