"""
app.py — RespiAI Flask Application (UI layer for Vercel)

Serves the frontend and proxies prediction requests to the
ML API hosted on Render (set API_BASE_URL env var).
"""

import os
import json
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from config import Config

load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# ML API base URL — set this to your Render service URL
API_BASE_URL = os.environ.get('API_BASE_URL', '').rstrip('/')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


# ─────────────────── Error Handlers ───────────────────

@app.errorhandler(404)
def not_found(error):
    if request.is_json or request.path.startswith('/api'):
        return jsonify({"error": "Resource not found"}), 404
    return render_template('error.html', code=404,
                           message="The page you're looking for doesn't exist."), 404

@app.errorhandler(500)
def internal_error(error):
    if request.is_json or request.path.startswith('/api'):
        return jsonify({"error": "Internal server error"}), 500
    return render_template('error.html', code=500,
                           message="Something went wrong on our end."), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File exceeds 16MB limit."}), 413


# ─────────────────── Routes ───────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid format. Allowed: {Config.ALLOWED_EXTENSIONS}"}), 400

    if not API_BASE_URL:
        return render_template('error.html', code=503,
                               message="ML API not configured. Set API_BASE_URL environment variable."), 503

    try:
        file_bytes = file.read()
        resp = requests.post(
            f"{API_BASE_URL}/predict",
            files={"file": (file.filename, file_bytes, file.content_type)},
            timeout=90
        )
        data = resp.json()
        if resp.status_code != 200 or "error" in data:
            error_msg = data.get("error", "Prediction failed.")
            if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
                return jsonify({"error": error_msg}), resp.status_code
            return render_template('index.html', error=error_msg)

        result = data.get("result", data)
        if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
            return jsonify({"status": "success", "result": result})
        return render_template('result.html', result=result)

    except requests.exceptions.Timeout:
        error_msg = "Analysis timed out. The model server may be waking up \u2014 please try again in 30 seconds."
        if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
            return jsonify({"error": error_msg}), 504
        return render_template('index.html', error=error_msg)
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
            return jsonify({"error": error_msg}), 500
        return render_template('index.html', error=error_msg)


@app.route('/predict-live', methods=['POST'])
def predict_live():
    """Proxy browser-recorded audio to the ML API."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio data received."}), 400

    if not API_BASE_URL:
        return jsonify({"error": "ML API not configured. Set API_BASE_URL."}), 503

    audio_file = request.files['audio']
    try:
        audio_bytes = audio_file.read()
        resp = requests.post(
            f"{API_BASE_URL}/predict",
            files={"audio": (audio_file.filename or "live.webm", audio_bytes, audio_file.content_type)},
            timeout=90
        )
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.Timeout:
        return jsonify({"error": "Analysis timed out. Please try again."}), 504
    except Exception as e:
        return jsonify({"error": f"Live prediction failed: {str(e)}"}), 500


@app.route('/render-result', methods=['POST'])
def render_result():
    """Render result.html from JSON posted by JavaScript."""
    try:
        result_data = json.loads(request.form.get('result_data', '{}'))
        return render_template('result.html', result=result_data)
    except Exception as e:
        return render_template('error.html', code=400, message=str(e)), 400


@app.route('/health', methods=['GET'])
def health():
    if API_BASE_URL:
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
            return jsonify(resp.json()), resp.status_code
        except Exception:
            pass
    return jsonify({"status": "ok", "ui": "vercel", "api": API_BASE_URL or "not configured"})


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/diseases', methods=['GET'])
def diseases():
    disease_info = []
    for d in Config.DISEASES:
        severity, emergency = Config.SEVERITY_MAP.get(d, ('Unknown', False))
        disease_info.append({
            "name": d,
            "severity": severity,
            "emergency": emergency,
            "recommendation": Config.RECOMMENDATIONS.get(d, '')
        })
    return jsonify(disease_info)


@app.route('/feedback', methods=['POST'])
def feedback():
    """Save user feedback on prediction accuracy."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    feedback_file = os.path.join('data', 'feedback.json')
    feedbacks = []
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            feedbacks = json.load(f)
    feedbacks.append(data)
    with open(feedback_file, 'w') as f:
        json.dump(feedbacks, f, indent=2)

    return jsonify({"status": "Feedback saved. Thank you!"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=Config.DEBUG)
