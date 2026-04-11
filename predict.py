"""
predict.py — Respiratory Disease Prediction Pipeline

Loads saved models. Extracts features from audio. Returns disease prediction
with confidence, severity, recommendations, and emergency flags.
"""

import os
import json
import joblib
import numpy as np
from extract_features import extract_features_from_file
from config import Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_model = None
_scaler = None
_selector = None
_encoder = None


def load_models():
    """Load all model artifacts with caching."""
    global _model, _scaler, _selector, _encoder
    if _model is not None:
        return _model, _scaler, _selector, _encoder
    try:
        _model = joblib.load(os.path.join(BASE_DIR, Config.MODEL_PATH))
        _scaler = joblib.load(os.path.join(BASE_DIR, Config.SCALER_PATH))
        _selector = joblib.load(os.path.join(BASE_DIR, Config.SELECTOR_PATH))
        _encoder = joblib.load(os.path.join(BASE_DIR, Config.ENCODER_PATH))
        return _model, _scaler, _selector, _encoder
    except Exception as e:
        print(f"Model loading error: {e}")
        return None, None, None, None


def get_severity(disease_name, confidence):
    """Determine severity level from disease and confidence."""
    base_severity, _ = Config.SEVERITY_MAP.get(disease_name, ('Moderate', False))

    if confidence >= 90:
        if base_severity in ['Severe']:
            return 'Critical'
        return base_severity
    elif confidence >= 70:
        return base_severity
    else:
        return 'Mild' if base_severity == 'Normal' else base_severity


def predict_respiratory(audio_path):
    """
    Full prediction pipeline:
    1. Extract features
    2. Scale + Select
    3. Predict
    4. Return structured result
    """
    model, scaler, selector, encoder = load_models()

    if model is None:
        return {"error": "Models not loaded. Run train_model.py first."}

    # Extract features
    features = extract_features_from_file(audio_path)
    if features is None:
        return {"error": "Failed to extract features. Audio may be corrupted or too short."}

    # Convert to array and match feature order
    feature_values = np.array(list(features.values())).reshape(1, -1)

    # Handle dimension mismatch gracefully
    expected_features = scaler.n_features_in_
    if feature_values.shape[1] != expected_features:
        # Pad or truncate
        if feature_values.shape[1] < expected_features:
            padding = np.zeros((1, expected_features - feature_values.shape[1]))
            feature_values = np.hstack([feature_values, padding])
        else:
            feature_values = feature_values[:, :expected_features]

    # Replace NaN/Inf
    feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale → Select
    X_scaled = scaler.transform(feature_values)
    X_selected = selector.transform(X_scaled)

    # Predict
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_selected)[0]
    else:
        prediction = model.predict(X_selected)[0]
        probas = np.zeros(len(encoder.classes_))
        probas[prediction] = 1.0

    # Process results
    top_n = min(3, len(probas))
    top_idx = np.argsort(probas)[::-1][:top_n]
    top_labels = encoder.inverse_transform(top_idx)
    top_probs = probas[top_idx]

    disease_name = top_labels[0]
    confidence = float(top_probs[0] * 100)

    severity_level = get_severity(disease_name, confidence)
    _, should_emergency = Config.SEVERITY_MAP.get(disease_name, ('Moderate', False))
    recommendation = Config.RECOMMENDATIONS.get(disease_name, 'Consult a medical professional.')

    # Emergency override for critical
    if severity_level == 'Critical':
        should_emergency = True

    top_3 = [
        {"label": str(top_labels[i]), "probability": float(top_probs[i] * 100)}
        for i in range(len(top_idx))
    ]

    # Save extracted features for frontend display
    features_file = os.path.join(BASE_DIR, 'data', 'last_extraction.json')
    try:
        os.makedirs(os.path.dirname(features_file), exist_ok=True)
        with open(features_file, 'w') as f:
            json.dump({k: float(v) for k, v in features.items()}, f, indent=2)
    except Exception:
        pass

    result = {
        "prediction": disease_name,
        "confidence": confidence,
        "severity": severity_level,
        "recommendation": recommendation,
        "should_seek_emergency": should_emergency,
        "top_3": top_3,
        "features": {k: round(float(v), 6) for k, v in list(features.items())[:30]},
        "total_features": len(features),
    }

    # Inconclusive check
    if confidence < 60.0:
        result["prediction"] = "Inconclusive"
        result["recommendation"] = "Audio quality was insufficient. Please record again or consult a doctor."
        result["severity"] = "Unknown"

    return result
