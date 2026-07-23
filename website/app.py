"""
STICH demo — Flask server.

Serves the static demo site and exposes a single inference endpoint,
POST /api/convert, which currently returns a DUMMY generated audio clip
(see inference.py). Swapping in the trained model later is a one-function
change inside inference.py; this file does not need to change.

Run:
    ./.venv/bin/python app.py
then open http://127.0.0.1:5000
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

    # DUMMY for now: ignores the input and returns a synthesized clip.
    wav_bytes = inference.convert(audio_bytes, direction, filename=upload.filename)

    return send_file(
        io.BytesIO(wav_bytes),
        mimetype="audio/wav",
        as_attachment=False,
        download_name="stich_output.wav",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
