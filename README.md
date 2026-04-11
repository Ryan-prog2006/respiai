# RespiAI — Respiratory Disease Detection

An AI-powered Flask web application that detects respiratory diseases from lung sound audio recordings using an ensemble of XGBoost, LightGBM, and Random Forest models trained on the ICBHI 2017 Respiratory Sound Database.

## Diseases Detected
Normal, COPD, Pneumonia, Bronchiectasis, Bronchiolitis, URTI, LRTI, Asthma

## Tech Stack
- **Backend:** Flask, Gunicorn, Python 3.10
- **ML:** XGBoost, LightGBM, scikit-learn, librosa
- **Frontend:** Bootstrap 5, Chart.js, WaveSurfer.js
- **Dataset:** ICBHI 2017 Respiratory Sound Database

## Setup
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

## Pipeline
1. `load_dataset.py` — Download and prepare ICBHI 2017 dataset
2. `extract_features.py` — Extract 600+ audio features
3. `train_model.py` — Train ensemble model (target >92% accuracy)
4. `app.py` — Serve predictions via Flask

## Folder Structure
```
respi-ai/
├── app.py, config.py, predict.py
├── extract_features.py, train_model.py, load_dataset.py
├── models/        # Saved .pkl model files
├── data/          # Audio, annotations, cycles
├── features/      # Extracted feature CSVs
├── static/        # CSS, JS, assets
└── templates/     # HTML templates
```
