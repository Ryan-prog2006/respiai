# RespiAI: An AI-Powered Respiratory Disease Detection System Using Ensemble Machine Learning on Lung Sound Recordings

---

## Overview

RespiAI is a full-stack AI-powered clinical decision support system that detects respiratory diseases from lung sound audio recordings. It uses an ensemble of gradient boosting and tree-based machine learning models trained on the ICBHI 2017 Respiratory Sound Database. The system extracts over 600 acoustic features per audio sample and classifies them into 8 respiratory conditions with confidence scoring, severity grading, and emergency flagging. The application is deployed as a web service accessible via browser, supporting both file upload and live microphone recording.

**Live Demo:** https://respiai.vercel.app  
**ML API:** https://respiai-4oll.onrender.com/health  
**GitHub:** https://github.com/Ryan-prog2006/respiai

---

## Research Context

Respiratory diseases are among the leading causes of morbidity and mortality worldwide. Chronic Obstructive Pulmonary Disease (COPD), pneumonia, and asthma collectively affect hundreds of millions of people globally. Traditional auscultation — the manual listening of lung sounds using a stethoscope — is highly subjective and dependent on clinician experience. Automated analysis of lung sounds using machine learning offers a scalable, objective, and accessible alternative, particularly in resource-limited settings.

This work presents RespiAI, a system that:
1. Processes raw lung sound audio recordings
2. Extracts a comprehensive multi-domain acoustic feature set (600+ features)
3. Applies a soft-voting ensemble classifier trained with class balancing
4. Delivers real-time predictions via a web interface with clinical severity grading

---

## Dataset: ICBHI 2017 Respiratory Sound Database

- **Full Name:** International Conference on Biomedical and Health Informatics (ICBHI) 2017 Respiratory Sound Database
- **Source:** https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- **Total Recordings:** 920 annotated audio files
- **Total Patients:** 126 unique patients
- **Total Respiratory Cycles Extracted:** 7,138 individual cycles
- **Sampling Rate:** 22,050 Hz (resampled from original)
- **Recording Devices:** 4 different devices (see Table 1)
- **Annotation Format:** Per-cycle timestamps with crackle/wheeze labels

### Table 1 — Recording Devices in ICBHI 2017

| Device Code | Full Name | Type |
|---|---|---|
| AKGC417L | AKG C417L | Condenser Microphone |
| LittC2SE | Littmann Classic II SE | Electronic Stethoscope |
| Meditron | Meditron | Electronic Stethoscope |
| Litt3200 | 3M Littmann 3200 | Electronic Stethoscope |

### Table 2 — Disease Classes and Cycle Distribution

| Disease | Description | Severity | Emergency |
|---|---|---|---|
| Normal | Healthy lung sounds | Normal | No |
| COPD | Chronic Obstructive Pulmonary Disease | Moderate | Yes |
| Pneumonia | Bacterial/viral lung infection | Severe | Yes |
| Bronchiectasis | Permanent bronchial widening | Moderate | No |
| Bronchiolitis | Small airway inflammation (pediatric) | Mild | No |
| URTI | Upper Respiratory Tract Infection | Mild | No |
| LRTI | Lower Respiratory Tract Infection | Severe | Yes |
| Asthma | Airway narrowing and inflammation | Moderate | No |

### Chest Recording Locations

Audio was recorded from 6 chest locations:
- **Al** — Anterior Left
- **Ar** — Anterior Right
- **Pl** — Posterior Left
- **Pr** — Posterior Right
- **Ll** — Lateral Left
- **Lr** — Lateral Right
- **Tc** — Trachea

### Filename Convention

```
{PatientID}_{RecordingIndex}_{ChestLocation}_{AcquisitionMode}_{Device}.wav
Example: 101_1b1_Al_sc_Meditron.wav
```

---

## System Architecture

The system is split into two deployment layers:

```
[User Browser]
     |
     | HTTPS
     v
[Vercel — UI Layer]          [Render — ML API Layer]
  Flask (app.py)    ------>   Flask (api_server.py)
  Templates/CSS/JS            predict.py
  Proxy requests              extract_features.py
                              models/*.pkl
```

> **Figure 1:** High-level system architecture showing the two-tier deployment. The UI layer (Vercel) handles all frontend rendering and proxies audio files to the ML API layer (Render) for inference.

### Pipeline Overview

```
Audio Input (.wav/.mp3/.ogg/.webm)
        |
        v
[1. Audio Validation]
   - Duration check (1–30 seconds)
   - Format validation
   - File size limit (16 MB)
        |
        v
[2. Feature Extraction]  <-- extract_features.py
   - MFCC (320 features)
   - Spectral (18 features)
   - Energy (7 features)
   - Respiratory-specific (8 features)
   - Mel Spectrogram (256 features)
   - Chroma (24 features)
   Total: 633 features
        |
        v
[3. Preprocessing]
   - RobustScaler normalization
   - SelectKBest (top 200 features, f_classif)
        |
        v
[4. Ensemble Inference]
   - Soft-voting: XGBoost + LightGBM + Random Forest
   - Probability output for 8 classes
        |
        v
[5. Post-processing]
   - Top-3 predictions with probabilities
   - Severity grading
   - Emergency flag
   - Confidence threshold (< 60% → Inconclusive)
        |
        v
[6. Result Display]
   - Prediction + severity badge
   - Confidence gauge (SVG)
   - Top-3 bar chart (Chart.js)
   - Medical recommendation
```

> **Figure 2:** End-to-end prediction pipeline from raw audio input to clinical result output.

---

## Feature Extraction (extract_features.py)

Features are extracted using `librosa` at a fixed sample rate of 22,050 Hz with a maximum duration of 10 seconds per cycle.

### A. MFCC Features — 320 total

Mel-Frequency Cepstral Coefficients capture the spectral envelope of the audio signal, which is highly correlated with the timbre of lung sounds.

- 40 MFCC coefficients × 4 statistics (mean, std, min, max) = **160 features**
- 40 delta MFCC × 2 statistics (mean, std) = **80 features**
- 40 delta-delta MFCC × 2 statistics (mean, std) = **80 features**

### B. Spectral Features — 18 total

| Feature | Description |
|---|---|
| Spectral Centroid | Weighted mean of frequencies (brightness) |
| Spectral Bandwidth | Spread of frequencies around centroid |
| Spectral Rolloff | Frequency below which 85% of energy lies |
| Spectral Contrast (×7) | Difference between peaks and valleys per sub-band |
| Spectral Flatness | Ratio of geometric to arithmetic mean (tonality) |

Each computed as mean + std = 2 values per feature.

### C. Energy Features — 7 total

| Feature | Description |
|---|---|
| RMS Energy (mean, std) | Root mean square signal energy |
| Zero Crossing Rate (mean, std) | Rate of sign changes (noise indicator) |
| Log Energy | Log of total signal power |
| Short-Time Energy (mean, std) | Frame-level energy variation |

### D. Respiratory-Specific Features — 8 total

These features are specifically designed to capture physiological characteristics of breathing:

| Feature | Description | Clinical Relevance |
|---|---|---|
| F0 Mean | Fundamental frequency (mean) | Vocal fold vibration, airway resonance |
| F0 Std | Fundamental frequency (std) | Pitch variability |
| HNR | Harmonics-to-Noise Ratio | Signal regularity, turbulence detection |
| Jitter | Period perturbation | Cycle-to-cycle frequency variation |
| Shimmer | Amplitude perturbation | Cycle-to-cycle amplitude variation |
| Breathing Rate | Estimated breaths per minute | Tachypnea/bradypnea detection |
| I/E Ratio | Inspiratory-to-Expiratory energy ratio | Obstructive vs. restrictive patterns |

**F0 Estimation:** Uses the PYIN algorithm (probabilistic YIN) with fmin=50 Hz, fmax=500 Hz.  
**HNR:** Computed via harmonic-percussive source separation (HPSS).  
**Breathing Rate:** Estimated from the low-frequency energy envelope (0.15–0.6 Hz band, corresponding to 9–36 breaths/min).

### E. Mel Spectrogram Features — 256 total

128 Mel filter bank bands × 2 statistics (mean, std) computed on the log-power Mel spectrogram.

### F. Chroma Features — 24 total

12 chroma bands × 2 statistics (mean, std) from the short-time Fourier transform chroma representation.

### Total Feature Count: 633 features per audio sample

> **Figure 3:** Feature extraction pipeline showing the six feature categories extracted from each respiratory cycle audio segment. Arrows indicate the flow from raw waveform to the final 633-dimensional feature vector.

---

## Data Augmentation

To address class imbalance in the ICBHI dataset, minority classes (those with fewer than 70% of the majority class count) are augmented using four techniques applied during feature extraction:

| Technique | Parameters | Purpose |
|---|---|---|
| Time Stretching | Rate: 0.8×, 1.2× | Simulate different breathing speeds |
| Pitch Shifting | ±2 semitones | Simulate different patient vocal tracts |
| White Noise Addition | SNR = 20 dB | Simulate recording noise |
| Time Shifting | ±20% of duration | Simulate recording start offset |

Each augmented sample generates a new feature vector, effectively multiplying minority class samples by up to 6×.

> **Figure 4:** Data augmentation strategy applied to minority classes. Each original audio cycle produces up to 6 augmented variants (original + 2 time-stretch + 2 pitch-shift + 1 noise + 1 time-shift), increasing minority class representation before SMOTE is applied.

---

## Model Training (train_model.py)

### Preprocessing Pipeline

1. **Label Encoding:** `sklearn.LabelEncoder` maps 8 disease strings to integers 0–7
2. **Feature Scaling:** `RobustScaler` (median/IQR-based, robust to outliers)
3. **Feature Selection:** `SelectKBest` with `f_classif` (ANOVA F-statistic), selects top 200 features from 633
4. **Class Balancing:**
   - SMOTE (Synthetic Minority Over-sampling Technique) with k=5 neighbors
   - Followed by `RandomUnderSampler` to prevent majority class dominance

### Models Trained

Five classifiers are trained with `GridSearchCV` (10-fold `StratifiedKFold`, scoring=accuracy):

| Model | Hyperparameter Grid |
|---|---|
| Random Forest | n_estimators: [200, 500], max_depth: [None, 20] |
| XGBoost | n_estimators: [200, 500], learning_rate: [0.01, 0.1] |
| LightGBM | n_estimators: [200, 500], num_leaves: [31, 63] |
| SVM (RBF) | C: [1, 10, 100] |
| MLP | hidden_layer_sizes: [(512,256,128), (256,128,64)] |

### Ensemble Construction

The top 3 models by cross-validation score are combined into a **soft-voting ensemble** (`VotingClassifier`, voting='soft'). Soft voting averages the predicted class probabilities across the three models, which is more robust than hard voting for multi-class problems.

```
Ensemble = SoftVoting(Model_1, Model_2, Model_3)
P(class_k) = (P_1(k) + P_2(k) + P_3(k)) / 3
```

### Model Artifacts Saved

| File | Contents |
|---|---|
| `best_respiratory_model.pkl` | Trained VotingClassifier ensemble |
| `respiratory_scaler.pkl` | Fitted RobustScaler |
| `respiratory_selector.pkl` | Fitted SelectKBest (k=200) |
| `respiratory_encoder.pkl` | Fitted LabelEncoder |

> **Figure 5:** Model training and selection pipeline. Five classifiers are trained with GridSearchCV, ranked by cross-validation accuracy, and the top three are combined into a soft-voting ensemble. The ensemble is saved only if its accuracy meets or exceeds the best individual model.

---

## Severity and Emergency Classification

Post-prediction, the system applies a rule-based severity grading:

### Table 3 — Severity Mapping

| Disease | Base Severity | Emergency Flag | Escalation Rule |
|---|---|---|---|
| Normal | Normal | No | — |
| URTI | Mild | No | — |
| Bronchiolitis | Mild | No | — |
| Asthma | Moderate | No | → Severe if confidence ≥ 90% |
| COPD | Moderate | Yes | → Severe if confidence ≥ 90% |
| Bronchiectasis | Moderate | No | — |
| Pneumonia | Severe | Yes | → Critical if confidence ≥ 90% |
| LRTI | Severe | Yes | → Critical if confidence ≥ 90% |

**Inconclusive threshold:** If model confidence < 60%, the prediction is overridden to "Inconclusive" and the user is advised to re-record.

---

## Web Application (app.py / api_server.py)

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main dashboard (index.html) |
| `/predict` | POST | File upload prediction |
| `/predict-live` | POST | Browser microphone recording prediction |
| `/render-result` | POST | Render result.html from JSON |
| `/health` | GET | API health check |
| `/diseases` | GET | JSON list of all diseases with metadata |
| `/about` | GET | Disease encyclopedia page |
| `/feedback` | POST | Save user prediction feedback |

### Frontend Features

- Drag-and-drop audio file upload with waveform preview (Web Audio API)
- Live microphone recording (MediaRecorder API, 5-second countdown)
- Real-time waveform visualization (Web Audio AnalyserNode)
- Siri-style live transcription (Web Speech Recognition API)
- Animated confidence gauge (SVG, color-coded: red <60%, yellow 60–80%, green >80%)
- Top-3 predictions bar chart (Chart.js)
- Responsive dark medical UI (Bootstrap 5, glassmorphism design)
- Disease encyclopedia with symptoms, causes, and treatment info

> **Figure 6:** RespiAI web interface showing the main dashboard with the file upload panel (left) and live recording panel (right). The upload panel includes drag-and-drop functionality and real-time waveform preview.

> **Figure 7:** RespiAI results page showing the prediction output for a sample audio file. Components include: (a) prediction label with severity badge, (b) animated confidence gauge, (c) top-3 disease probability bar chart, (d) medical recommendation card, and (e) collapsible extracted feature table.

---

## Deployment Architecture

### Two-Tier Cloud Deployment

| Layer | Platform | Purpose | URL |
|---|---|---|---|
| UI Layer | Vercel (Serverless) | Frontend + proxy | https://respiai.vercel.app |
| ML API Layer | Render (Free Tier) | Inference + model serving | https://respiai-4oll.onrender.com |

### Why Split Deployment?

Vercel serverless functions have a 250 MB size limit. The full ML stack (librosa, scikit-learn, XGBoost, LightGBM) exceeds this. The solution is to deploy a lightweight Flask proxy on Vercel (4 dependencies, ~10 MB) that forwards audio to the Render-hosted ML API.

### Vercel Layer Dependencies
```
flask==2.3.3
flask-cors==4.0.0
python-dotenv==1.0.0
requests==2.31.0
```

### Render Layer Dependencies
```
flask, flask-cors, gunicorn
librosa==0.10.1, soundfile, audioread
scikit-learn==1.3.0, xgboost==1.7.6, lightgbm==4.0.0
numpy, pandas, scipy, joblib
imbalanced-learn, python-dotenv, requests
```

> **Figure 8:** Deployment architecture diagram. The Vercel UI layer receives HTTP requests from the browser, validates the request, and forwards the audio file to the Render ML API via an internal HTTP POST. The ML API performs feature extraction and inference, returning a JSON result that the UI layer renders as HTML.

---

## Project File Structure

```
respi-ai/
├── app.py                    # UI Flask app (Vercel) — proxy layer
├── api_server.py             # ML Flask API (Render) — inference layer
├── config.py                 # Shared configuration (diseases, paths, severity)
├── predict.py                # Prediction pipeline (load → extract → infer)
├── extract_features.py       # 633-feature audio feature extractor
├── train_model.py            # Model training with GridSearchCV + ensemble
├── load_dataset.py           # ICBHI dataset loader and cycle splitter
├── download_icbhi.py         # Dataset download utility
│
├── models/
│   ├── best_respiratory_model.pkl   # Trained VotingClassifier
│   ├── respiratory_scaler.pkl       # RobustScaler
│   ├── respiratory_selector.pkl     # SelectKBest (k=200)
│   └── respiratory_encoder.pkl      # LabelEncoder
│
├── data/
│   ├── audio/               # Raw .wav recordings (920 files)
│   ├── annotations/         # Per-cycle annotation .txt files (922 files)
│   ├── cycles/              # Segmented respiratory cycles (7,138 .wav files)
│   ├── cycles.csv           # Master CSV with cycle metadata
│   └── patient_diagnosis.txt # Patient-level diagnosis labels
│
├── features/
│   └── respiratory_features.csv  # Extracted 633-feature matrix
│
├── static/
│   ├── css/style.css        # Dark medical UI theme
│   └── js/main.js           # Frontend interactivity
│
├── templates/
│   ├── index.html           # Main dashboard
│   ├── result.html          # Prediction results page
│   ├── about.html           # Disease encyclopedia
│   └── error.html           # Error page
│
├── requirements.txt          # Vercel (slim, 4 packages)
├── requirements-full.txt     # Render (full ML stack)
├── render.yaml               # Render deployment config
├── vercel.json               # Vercel deployment config
├── Procfile                  # Gunicorn start command
└── runtime.txt               # Python version spec
```

---

## Reproducibility: Running the Full Pipeline

```bash
# 1. Clone and install
git clone https://github.com/Ryan-prog2006/respiai
cd respiai
pip install -r requirements-full.txt

# 2. Place ICBHI dataset files in data/audio/ and data/annotations/

# 3. Load dataset and split into cycles
python load_dataset.py

# 4. Extract 633 features from all cycles
python extract_features.py

# 5. Train ensemble model
python train_model.py

# 6. Run the web app locally
python app.py
# Visit http://localhost:5000
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Feature extraction library | librosa | Industry standard for audio ML, rich feature set |
| Scaling method | RobustScaler | Robust to outliers common in medical audio |
| Feature selection | SelectKBest (f_classif) | Fast, interpretable, reduces 633→200 features |
| Class balancing | SMOTE + RandomUnderSampler | Handles severe imbalance in ICBHI dataset |
| Ensemble method | Soft voting | Averages probabilities, better calibrated confidence |
| Inconclusive threshold | 60% confidence | Prevents overconfident low-quality predictions |
| Audio resampling | 22,050 Hz | Standard for speech/audio ML, balances quality/speed |
| Max cycle duration | 10 seconds | Covers full respiratory cycle, limits compute |

---

## Limitations and Future Work

1. **Dataset size:** ICBHI 2017 contains 126 patients — larger datasets would improve generalization
2. **Cold start latency:** Render free tier sleeps after 15 min inactivity (~30s wake-up)
3. **Audio quality dependency:** Background noise significantly affects feature quality
4. **Single-cycle analysis:** The system analyzes one cycle at a time; multi-cycle aggregation could improve accuracy
5. **No patient history:** The model does not incorporate patient demographics or medical history
6. **Future:** Deep learning approaches (CNN on mel spectrograms, transformer-based audio models) may outperform the handcrafted feature approach

---

## References

1. Rocha, B.M., et al. "A Respiratory Sound Database for the Development of Automated Classification Systems." *ICBHI 2017 Challenge*. Springer, 2018.
2. McFee, B., et al. "librosa: Audio and Music Signal Analysis in Python." *Proceedings of the 14th Python in Science Conference*, 2015.
3. Chen, T., Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *KDD 2016*.
4. Ke, G., et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 2017*.
5. Chawla, N.V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR*, 2002.
6. Makarenkov, A., et al. "Automatic Classification of Respiratory Sounds." *Biomedical Signal Processing and Control*, 2019.

---

## Figures Summary (for paper insertion)

| Figure | Description | Suggested Tool |
|---|---|---|
| Figure 1 | Two-tier system architecture (Vercel + Render) | Draw.io / Lucidchart |
| Figure 2 | End-to-end prediction pipeline flowchart | Draw.io |
| Figure 3 | Feature extraction categories diagram | Draw.io / Canva |
| Figure 4 | Data augmentation strategy diagram | Canva / Gemini |
| Figure 5 | Model training and ensemble selection pipeline | Draw.io |
| Figure 6 | RespiAI web UI — main dashboard screenshot | Screenshot |
| Figure 7 | RespiAI web UI — results page screenshot | Screenshot |
| Figure 8 | Cloud deployment architecture diagram | Draw.io / Gemini |
| Figure 9 | Confusion matrix heatmap | models/confusion_matrix.png |
| Figure 10 | Class distribution bar chart | data/class_distribution.png |

---

*This document is intended as a complete technical reference for generating a research paper on the RespiAI system. All figures marked with "Gemini" are recommended to be generated using Gemini AI with the figure description as the prompt.*
