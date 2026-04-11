"""
api_server.py — Standalone ML prediction API for Render deployment.
Exposes only /predict and /health endpoints (no UI).
"""

import os
import uuid
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from config import Config

load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "Ensemble", "accuracy": "92%+", "diseases": len(Config.DISEASES)})


@app.route('/predict', methods=['POST'])
def predict():
    file_key = 'file' if 'file' in request.files else 'audio' if 'audio' in request.files else None
    if not file_key:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files[file_key]
    ext = 'wav'
    if audio_file.filename and '.' in audio_file.filename:
        ext = audio_file.filename.rsplit('.', 1)[1].lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Invalid format. Allowed: {Config.ALLOWED_EXTENSIONS}"}), 400

    filename = f"upload_{uuid.uuid4().hex[:8]}.{ext}"
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)

    try:
        audio_file.save(filepath)

        duration = librosa.get_duration(path=filepath)
        if duration < Config.MIN_AUDIO_DURATION or duration > Config.MAX_AUDIO_DURATION:
            os.remove(filepath)
            return jsonify({"error": f"Audio must be {Config.MIN_AUDIO_DURATION}-{Config.MAX_AUDIO_DURATION}s. Got {duration:.1f}s."}), 400

        from predict import predict_respiratory
        result = predict_respiratory(filepath)

        if os.path.exists(filepath):
            os.remove(filepath)

        if "error" in result:
            return jsonify(result), 500

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
