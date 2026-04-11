import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'respi-ai-default-secret')
    DEBUG = os.environ.get('DEBUG', 'False').lower() in ['true', '1']

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm'}

    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'audio')
    MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
    FEATURES_FOLDER = os.path.join(BASE_DIR, 'features')
    DATA_FOLDER = os.path.join(BASE_DIR, 'data')

    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/best_respiratory_model.pkl')
    SCALER_PATH = os.environ.get('SCALER_PATH', 'models/respiratory_scaler.pkl')
    SELECTOR_PATH = os.environ.get('SELECTOR_PATH', 'models/respiratory_selector.pkl')
    ENCODER_PATH = os.environ.get('ENCODER_PATH', 'models/respiratory_encoder.pkl')

    MAX_AUDIO_DURATION = int(os.environ.get('MAX_AUDIO_DURATION', 30))
    MIN_AUDIO_DURATION = int(os.environ.get('MIN_AUDIO_DURATION', 1))

    DISEASES = [
        'Normal', 'COPD', 'Pneumonia', 'Bronchiectasis',
        'Bronchiolitis', 'URTI', 'LRTI', 'Asthma'
    ]

    RECOMMENDATIONS = {
        'Normal': 'No abnormalities detected. Maintain regular checkups.',
        'COPD': 'COPD indicators found. Schedule spirometry test urgently.',
        'Pneumonia': 'Pneumonia indicators detected. Seek immediate care.',
        'Bronchiectasis': 'Consult pulmonologist for further evaluation.',
        'Bronchiolitis': 'Monitor breathing. Consult pediatrician.',
        'URTI': 'Upper respiratory infection detected. Rest and hydrate.',
        'LRTI': 'Lower respiratory infection. Consult doctor immediately.',
        'Asthma': 'Asthma indicators found. Use rescue inhaler if prescribed.'
    }

    SEVERITY_MAP = {
        'Normal': ('Normal', False),
        'URTI': ('Mild', False),
        'Bronchiolitis': ('Mild', False),
        'Asthma': ('Moderate', False),
        'COPD': ('Moderate', True),
        'Bronchiectasis': ('Moderate', False),
        'Pneumonia': ('Severe', True),
        'LRTI': ('Severe', True),
    }
