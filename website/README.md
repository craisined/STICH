# STICH — Demo Website

A local demo site for the STICH model (humming ⇄ classical music).
Plain HTML/CSS/JS front end, Flask back end for the model.

> **Current stage: DUMMY.** The trained model is **not** wired in yet.
> - The "Hear it" gallery uses **synthesized placeholder** clips.
> - "Try it yourself" runs the real upload → server → download round-trip,
>   but the server returns a **synthesized placeholder** clip, not a
>   translation of your upload.
> Everything else (layout, players, flow) is final.

## Run it

```bash
cd website
python3 -m venv .venv          # first time only
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python app.py
```

Then open <http://127.0.0.1:5000>.

## Layout

```
website/
  app.py              # Flask: serves the site + POST /api/convert
  inference.py        # convert() — DUMMY now; swap in the model here
  gen_placeholders.py # regenerates the example clips in static/examples/
  requirements.txt
  static/
    index.html
    css/styles.css
    js/main.js
    examples/         # placeholder before/after .wav clips
```

## Wiring in the real model (later)

1. **Save a checkpoint.** `main.py` trains but never saves one — add a
   `torch.save(model.module.state_dict(), ...)` for the generators.
2. **Fill in `inference.convert()`** — the docstring there lists the exact
   steps (decode → 16 kHz mono → pad/crop to 10 s → generator → WAV bytes).
   No changes to `app.py` or the front end are needed.
3. **Replace the gallery clips** in `static/examples/` with real model output
   (same filenames the gallery references in `index.html`).
4. Uncomment `torch` / `librosa` / `soundfile` / `numpy` in `requirements.txt`.
