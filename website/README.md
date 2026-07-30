# STICH — Demo Website

A local demo site for the STICH model (humming → classical music).
Plain HTML/CSS/JS front end, Flask back end for the model.

> **Current stage: presentable.** Every clip on the page is real model output.
> "Try it yourself" runs your upload — or a clip you hum straight into the mic —
> through the autoencoder and a CycleGAN generator in latent space, and the
> "Hear it" gallery is rendered by `gen_examples.py` through that same code
> path. No placeholders and no synthesized stand-ins remain.

> **The site is humming → classical only.** The reverse generator exists and
> `/api/convert` still accepts `classical_to_humming`, but its output is not
> good enough to show, so nothing in the UI, the gallery, or the copy offers or
> mentions that direction. The page describes the reverse generator only as
> what the cycle-consistency loss is measured against during training — which
> is the honest answer if a visitor asks why it is a CycleGAN. Keep it that way
> unless that direction is retrained to a standard worth demoing.

Needs three files the training side produces. `encoder.ts` is not in git (see
`.gitignore`); the two generator checkpoints `gen_humming_to_classical.pth` /
`gen_classical_to_humming.pth` are. `inference.py` loads all three — both
generators come up together even though the site only calls one. It looks for
them in the repo root, then `../models/`, then this directory, and
`/api/convert` answers 503 with the paths it searched if they are missing.

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
  inference.py        # convert() — decode → encode → generator → decode → WAV
  main.py             # CycleGAN training script; inference.py takes Generator from it
  encoder.ts, *.pth   # optional fallback copies; the repo root wins
  gen_examples.py     # rebuilds static/examples/ from real model output
  requirements.txt
  static/
    index.html
    css/styles.css
    js/main.js
    examples/         # before/after .wav clips for the gallery
```

## How inference works

The autoencoder supplies the latent space; a CycleGAN generator does the
translation inside it:

1. Decode to 16 kHz mono, repeat the preprocessing the training data went
   through (`data/process_humtrans.py` for humming, `data/process_musicnet.py`
   for classical), pad to exactly 10 s.
2. Resample to the model's 48 kHz. Padding comes **first** — the generators
   were trained on encodings of exactly 160 000 samples at 16 kHz, so a
   different length would not line up with them.
3. `model.encode()` to a `(1, 16, 234)` embedding, run it through the
   generator for the requested direction, `model.decode()`. The generators saw
   raw encoder output in training, so nothing is normalized on the way in or
   out — unlike the centroid arithmetic this replaced, which needed unit norm.
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

The encoder and both generators load on first use and are cached, so the first
conversion after a restart is a beat slower.

`inference.py` pulls the `Generator` class out of `main.py` — the training
script the checkpoints were saved from, so the architecture cannot drift away
from the weights. It loads that file by path rather than with `import main`,
because the repo root holds a different `main.py` that would otherwise win.

### Two generator architectures

The checkpoints are retrained often and land one direction at a time, so the
pair is regularly mid-swap — one file from the new run, one still from the old.
`_build()` reads each checkpoint's keys and picks its architecture, so the
direction that is current keeps working while the other is re-exported:

- **batch norm** (`running_mean` in the keys) → `Generator` from `main.py`.
- **weight norm** (`weight_g` / `weight_v`) → `WeightNormGenerator` in
  `inference.py`: the same five-block layout, weight-normalized pre-activation
  residual blocks instead of batch norm.

`WeightNormGenerator` is reconstructed from the checkpoint's keys. Those pin
every shape, but **not the activations** — those carry no weights, so a wrong
guess there loads cleanly and quietly degrades the audio rather than erroring.
It assumes `nn.LeakyReLU()` at the unparameterized slots, matching `main.py`.
Once `main.py` is synced with the training run, move the class into it and
delete the copy here.

A checkpoint matching neither architecture raises with the offending filename,
and `/api/convert` answers 503 with that message.

## Mic recording

The record button captures up to 10 s with `MediaRecorder`, then decodes and
re-encodes it as a PCM WAV **in the browser** (`encodeWav` in `js/main.js`)
before uploading. That matters: `MediaRecorder` produces webm/opus (mp4 on
Safari), neither of which soundfile can read — converting client-side keeps
the server on one audio path and off ffmpeg. Recording needs a secure context,
which `127.0.0.1` counts as; over a LAN address the browser will refuse.

## After retraining

1. Drop the new `encoder.ts` at the repo root (or in `../models/`).
2. **Retrain both generators** — `main.py`. A generator only means anything in
   the latent space it was trained in, so stale `.pth` files with a new encoder
   produce noise, not an error. Same if the `Generator` architecture in
   `main.py` changes without new weights: that fails loudly at load instead.
3. Nothing else for "Try it yourself" — `inference.py` picks all three up from
   the repo root.
4. **Rebuild the gallery** so it shows the new checkpoint:

   ```bash
   ./.venv/bin/python website/gen_examples.py
   ```

## Gallery clips

`gen_examples.py` draws source hums from `../data/humtrans_processed/`, runs
them through the same `inference.convert()` that serves visitors, and writes
the six files `index.html` references. The gallery therefore plays exactly what
the upload box would return for those clips — no separate code path to drift
out of step.

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
- **Size.** The six clips total about 4 MB, and `computePeaks` in
  `js/main.js` fetches every one of them on page load to draw its waveform.
  Worth revisiting if the demo ever leaves localhost.

The clips currently committed were drawn with `--seed 0`. The card titles in
`index.html` are deliberately generic ("Hummed melody · one"), so a re-roll does
not leave the page describing clips it no longer plays.
