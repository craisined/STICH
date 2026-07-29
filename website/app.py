"""
STICH demo — Flask server.

Serves the static demo site and exposes a single inference endpoint,
POST /api/convert, which runs the uploaded clip through the autoencoder and
shifts its embedding toward the requested direction (see inference.py).

Run (from the repo root, using the repo venv):
    ./.venv/bin/python website/app.py
then open http://127.0.0.1:5000

Port 5000 is taken by AirPlay Receiver on macOS. If it is in use:
    ./.venv/bin/python -m flask --app website/app.py run --port 5001
"""
import io
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory

import inference

BASE = Path(__file__).parent
app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="")

# Directions the UI is allowed to request.
DIRECTIONS = {"humming_to_classical", "classical_to_humming"}


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/convert", methods=["POST"])
def convert():
    if "audio" not in request.files:
        return jsonify(error="No audio file was uploaded."), 400

    upload = request.files["audio"]
    if not upload.filename:
        return jsonify(error="The uploaded file has no name."), 400

    direction = request.form.get("direction", "humming_to_classical")
    if direction not in DIRECTIONS:
        return jsonify(error=f"Unknown direction '{direction}'."), 400

    audio_bytes = upload.read()
    if not audio_bytes:
        return jsonify(error="The uploaded file is empty."), 400

    try:
        wav_bytes = inference.convert(audio_bytes, direction, filename=upload.filename)
    except FileNotFoundError as exc:
        # No encoder or centroids to run -- a setup problem, not a bad upload.
        app.logger.error("model unavailable: %s", exc)
        return jsonify(error=str(exc)), 503
    except ValueError as exc:
        # Bad upload -- the message is written for the person who chose it.
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("conversion failed")
        return jsonify(error="Something went wrong converting that file."), 500

    return send_file(
        io.BytesIO(wav_bytes),
        mimetype="audio/wav",
        as_attachment=False,
        download_name="stich_output.wav",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
