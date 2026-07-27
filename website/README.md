# STICH — Demo Website

A local demo site for the STICH model (humming ⇄ classical music).
Plain HTML/CSS/JS front end, Flask back end for the model.

> **Current stage: live model.** "Try it yourself" runs your upload — or a
> clip you hum straight into the mic — through the exported ONNX generators.
> Real output, from an early checkpoint.
> The "Hear it" gallery still uses **synthesized placeholder** clips;
> regenerate those from real model output when you have a checkpoint you like.

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
  inference.py        # convert() — decode → ONNX generator → WAV
  *.onnx              # fallback copies; ../models/ wins when present
  gen_placeholders.py # regenerates the example clips in static/examples/
  requirements.txt
  static/
    index.html
    css/styles.css
    js/main.js
    examples/         # placeholder before/after .wav clips
```

## How inference works

`inference.convert()` decodes the upload to 16 kHz mono, repeats the
preprocessing the training data went through (`data/process_humtrans.py` for
humming, `data/process_musicnet.py` for classical), pads to 10 s, and runs it
through the matching `.onnx` with onnxruntime. No torch, and no import of
`../models.py` — the graph and weights are baked into the export.

The generator's last layer is `InstanceNorm1d` with no bounding nonlinearity,
so its output is not in [-1, 1]; `_postprocess()` normalizes to a fixed peak
before encoding, otherwise the 16-bit WAV would clip flat.

Sessions are built on first use and cached, so the first conversion after a
restart is a beat slower. A 10 s clip takes roughly 1–2 s on CPU.

## Mic recording

The record button captures up to 10 s with `MediaRecorder`, then decodes and
re-encodes it as a PCM WAV **in the browser** (`encodeWav` in `js/main.js`)
before uploading. That matters: `MediaRecorder` produces webm/opus (mp4 on
Safari), neither of which soundfile can read — converting client-side keeps
the server on one audio path and off ffmpeg. Recording needs a secure context,
which `127.0.0.1` counts as; over a LAN address the browser will refuse.

## After retraining

1. Re-export: `main.py` writes `../models/*.onnx` at the end of a run, or run
   `export_onnx.py <checkpoint> models/` against a saved `.pt`.
2. Nothing else — `inference.py` prefers `../models/` and falls back to the
   copies in this directory.
3. **Replace the gallery clips** in `static/examples/` with real model output
   (same filenames the gallery references in `index.html`) — these are still
   synthesized placeholders from `gen_placeholders.py`.
