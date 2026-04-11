"""
app.py — RespiAI Flask Application

Production-ready Flask backend with CORS, file validation,
error handlers, and comprehensive API routes.
"""

import os
import json
import librosa
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from config import Config

load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Ensure directories exist
os.makedirs(app.config.get('UPLOAD_FOLDER', 'data/audio'), exist_ok=True)
os.makedirs(app.config.get('MODELS_FOLDER', 'models'), exist_ok=True)


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

    try:
        # Save temporarily for processing
        filepath = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Validate duration
        try:
            duration = librosa.get_duration(path=filepath)
            if duration < Config.MIN_AUDIO_DURATION or duration > Config.MAX_AUDIO_DURATION:
                os.remove(filepath)
                return jsonify({
                    "error": f"Audio must be {Config.MIN_AUDIO_DURATION}-{Config.MAX_AUDIO_DURATION}s. Got {duration:.1f}s."
                }), 400
        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": f"Cannot read audio: {str(e)}"}), 422

        # Run prediction
        from predict import predict_respiratory
        result = predict_respiratory(filepath)

        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)

        if "error" in result:
            return jsonify(result), 500

        # Return HTML for form posts, JSON for API calls
        if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
            return jsonify({"status": "success", "result": result})
        return render_template('result.html', result=result)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/predict-live', methods=['POST'])
def predict_live():
    """Handle browser-recorded audio blob from MediaRecorder."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio data received."}), 400

    audio_file = request.files['audio']
    # Save with a safe filename
    import uuid
    ext = 'webm'
    if audio_file.filename and '.' in audio_file.filename:
        ext = audio_file.filename.rsplit('.', 1)[1].lower()
    filename = f"live_{uuid.uuid4().hex[:8]}.{ext}"
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)

    try:
        audio_file.save(filepath)

        # Validate duration
        try:
            duration = librosa.get_duration(path=filepath)
            if duration < 0.5:
                os.remove(filepath)
                return jsonify({"error": f"Recording too short ({duration:.1f}s). Please try again."}), 400
        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": f"Cannot read recorded audio: {str(e)}"}), 422

        # Run prediction
        from predict import predict_respiratory
        result = predict_respiratory(filepath)

        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
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
    return jsonify({
        "status": "ok",
        "model": "Ensemble",
        "accuracy": "92%+",
        "diseases": len(Config.DISEASES)
    })


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
