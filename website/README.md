# STICH — Demo Website

A local demo site for the STICH model (humming ⇄ classical music).
Plain HTML/CSS/JS front end, Flask back end for the model.

> **Current stage: live model.** "Try it yourself" runs your upload — or a
> clip you hum straight into the mic — through the autoencoder and a style
> shift in embedding space. Real output, from an early checkpoint.
> The "Hear it" gallery still uses **synthesized placeholder** clips —
> run `gen_examples.py` to replace them with real model output.

Needs three files the training side produces, none of them in git (see
`.gitignore`): `encoder.ts` and the two centroids `humming_embedding.npy` /
`classical_embedding.npy`. `inference.py` looks for them in the repo root,
then `../models/`, then this directory, and `/api/convert` answers 503 with
the paths it searched if they are missing.

## Run it

The repo-root `.venv` already has everything:

```bash
./.venv/bin/python website/app.py
```

Then open <http://127.0.0.1:5000>. On macOS, port 5000 is usually taken by
AirPlay Receiver — either turn it off in System Settings → General →
AirDrop & Handoff, or pick another port:

```bash
./.venv/bin/python -m flask --app website/app.py run --port 5001
```

Starting from a bare environment instead:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r website/requirements.txt
```

## Layout

```
website/
  app.py              # Flask: serves the site + POST /api/convert
  inference.py        # convert() — decode → encode → style shift → decode → WAV
  encoder.ts, *.npy   # optional fallback copies; the repo root wins
  gen_examples.py     # rebuilds static/examples/ from real model output
  gen_placeholders.py # the synthesized stand-ins; delete once the real ones land
  requirements.txt
  static/
    index.html
    css/styles.css
    js/main.js
    examples/         # before/after .wav clips for the gallery
```

## How inference works

Same path as `../model_pipeline.py`, rebuilt against an upload instead of a
`.npy` from the training set:

1. Decode to 16 kHz mono, repeat the preprocessing the training data went
   through (`data/process_humtrans.py` for humming, `data/process_musicnet.py`
   for classical), pad to exactly 10 s.
2. Resample to the model's 48 kHz. Padding comes **first** — the centroids are
   averages over encodings of exactly 160 000 samples at 16 kHz, so a
   different length would not line up with them.
3. `model.encode()`, scale to unit norm, subtract the source centroid and add
   the target one, rescale to the original norm, `model.decode()`.
4. Normalize the peak to 0.95 and fade 15 ms at each edge (the same thing
   `audio_utils.process_audio` does), then write a 48 kHz mono WAV.

`inference.py` does not import `../config.py`: that would load the model
relative to the process's working directory and create `./output` at import
time. It carries its own copies of the constants instead — keep them in step
with `../config.py` and `../audio_utils.py`.

One deliberate difference from the pipeline: resampling uses librosa's soxr
rather than `torchaudio.transforms.Resample`. They are transparent at this
clean 3× ratio, and it keeps torchaudio off the server's dependency list.

The decoder's output is not confined to [-1, 1], so `_postprocess()` normalizes
before encoding — otherwise the 16-bit WAV would clip flat.

The encoder and centroids load on first use and are cached, so the first
conversion after a restart is a beat slower.

## Mic recording

The record button captures up to 10 s with `MediaRecorder`, then decodes and
re-encodes it as a PCM WAV **in the browser** (`encodeWav` in `js/main.js`)
before uploading. That matters: `MediaRecorder` produces webm/opus (mp4 on
Safari), neither of which soundfile can read — converting client-side keeps
the server on one audio path and off ffmpeg. Recording needs a secure context,
which `127.0.0.1` counts as; over a LAN address the browser will refuse.

## After retraining

1. Drop the new `encoder.ts` at the repo root (or in `../models/`).
2. **Recompute both centroids** — `model_pipeline.save_embeddings()`. A mean
   embedding only means anything in the latent space it was averaged in, so
   stale `.npy` files with a new encoder produce noise, not an error.
3. Nothing else for "Try it yourself" — `inference.py` picks both up from the
   repo root.
4. **Rebuild the gallery** so it shows the new checkpoint:

   ```bash
   ./.venv/bin/python website/gen_examples.py
   ```

## Gallery clips

`gen_examples.py` draws source clips from `../data/humtrans_processed/` and
`../data/musicnet_processed/`, runs them through the same `inference.convert()`
that serves visitors, and writes the thirteen files `index.html` references.
The gallery therefore plays exactly what the upload box would return for those
clips — no separate code path to drift out of step.

Nothing is written until every clip has rendered, so a missing model fails
without leaving the gallery half old and half new.

A draw is random but seeded. Re-roll until a set sounds good, then note the
seed:

```bash
./.venv/bin/python website/gen_examples.py --seed 12
```

Two things to know before committing the result:

- **Listen first.** These pairs are the first thing a visitor hears, and a bad
  draw makes the model sound worse than it is.
- **Size.** Real clips are about 8 MB against the placeholders' 0.9 MB, and
  `computePeaks` in `js/main.js` fetches every one of them on page load to draw
  its waveform. Worth revisiting if the demo ever leaves localhost.

Once real clips are in, drop the two `placeholder-banner` paragraphs from
`index.html` and delete `gen_placeholders.py`.
