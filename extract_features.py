"""
extract_features.py — Comprehensive Audio Feature Extraction for Respiratory Sounds

Extracts 600+ features: MFCC, Spectral, Energy, Respiratory-specific,
Mel Spectrogram, and Chroma features. Includes augmentation for minority classes.
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CYCLES_DIR = os.path.join(BASE_DIR, 'data', 'cycles')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
os.makedirs(FEATURES_DIR, exist_ok=True)


# ──────────────────────────── AUGMENTATION ────────────────────────────

def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def add_white_noise(y, snr_db=20):
    noise = np.random.randn(len(y))
    sig_power = np.mean(y ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    return y + np.sqrt(noise_power) * noise

def time_shift(y, shift_max=0.2):
    shift = int(len(y) * np.random.uniform(-shift_max, shift_max))
    return np.roll(y, shift)


# ──────────────────────────── FEATURE EXTRACTION ─────────────────────

def extract_mfcc_features(y, sr, n_mfcc=40):
    """A) MFCC Features: 40 coeffs * 4 stats = 160, + delta/delta-delta * 2 stats each = 320 total."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = {}
    for i in range(n_mfcc):
        features[f'mfcc{i}_mean'] = np.mean(mfcc[i])
        features[f'mfcc{i}_std'] = np.std(mfcc[i])
        features[f'mfcc{i}_min'] = np.min(mfcc[i])
        features[f'mfcc{i}_max'] = np.max(mfcc[i])
        features[f'delta_mfcc{i}_mean'] = np.mean(delta[i])
        features[f'delta_mfcc{i}_std'] = np.std(delta[i])
        features[f'delta2_mfcc{i}_mean'] = np.mean(delta2[i])
        features[f'delta2_mfcc{i}_std'] = np.std(delta2[i])
    return features


def extract_spectral_features(y, sr):
    """B) Spectral Features."""
    features = {}
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(cent)
    features['spectral_centroid_std'] = np.std(cent)

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(bw)
    features['spectral_bandwidth_std'] = np.std(bw)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(rolloff)
    features['spectral_rolloff_std'] = np.std(rolloff)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        features[f'spectral_contrast{i}_mean'] = np.mean(contrast[i])
        features[f'spectral_contrast{i}_std'] = np.std(contrast[i])

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features['spectral_flatness_mean'] = np.mean(flatness)
    features['spectral_flatness_std'] = np.std(flatness)
    return features


def extract_energy_features(y, sr):
    """C) Energy Features."""
    features = {}
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    features['log_energy'] = np.log(np.sum(y ** 2) + 1e-9)

    frame_length = 2048
    hop_length = 512
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    ste = np.sum(frames ** 2, axis=0)
    features['short_time_energy_mean'] = np.mean(ste)
    features['short_time_energy_std'] = np.std(ste)
    return features


def extract_respiratory_features(y, sr):
    """D) Respiratory-Specific Features."""
    features = {}

    # F0 using pyin
    fmax_val = min(sr / 2, 500)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=fmax_val, sr=sr)
    f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
    if len(f0_voiced) == 0:
        f0_voiced = np.array([0.0])
    features['f0_mean'] = np.mean(f0_voiced)
    features['f0_std'] = np.std(f0_voiced)

    # HNR via harmonic/percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    h_power = np.sum(y_harmonic ** 2)
    p_power = np.sum(y_percussive ** 2) + 1e-9
    features['hnr'] = 10 * np.log10(h_power / p_power) if h_power > 0 else 0.0

    # Jitter (period perturbation)
    T = 1.0 / (f0_voiced + 1e-9)
    dT = np.abs(np.diff(T))
    features['jitter'] = np.mean(dT) / (np.mean(T) + 1e-9) if len(dT) > 0 else 0.0

    # Shimmer (amplitude perturbation)
    rms = librosa.feature.rms(y=y)[0]
    dA = np.abs(np.diff(rms))
    features['shimmer'] = np.mean(dA) / (np.mean(rms) + 1e-9) if len(dA) > 0 else 0.0

    # Breathing rate estimation (low-frequency energy envelope)
    envelope = np.abs(librosa.onset.onset_strength(y=y, sr=sr))
    if len(envelope) > 2:
        fft_env = np.abs(np.fft.rfft(envelope))
        freqs = np.fft.rfftfreq(len(envelope), d=512 / sr)
        # Breathing range: 0.2 - 0.5 Hz (12-30 breaths/min)
        mask = (freqs >= 0.15) & (freqs <= 0.6)
        if np.any(mask) and np.sum(fft_env[mask]) > 0:
            features['breathing_rate'] = freqs[mask][np.argmax(fft_env[mask])] * 60
        else:
            features['breathing_rate'] = 15.0
    else:
        features['breathing_rate'] = 15.0

    # Inspiratory/Expiratory ratio
    mid = len(y) // 2
    insp_energy = np.sum(y[:mid] ** 2)
    exp_energy = np.sum(y[mid:] ** 2) + 1e-9
    features['ie_ratio'] = insp_energy / exp_energy
    return features


def extract_mel_features(y, sr, n_mels=128):
    """E) Mel Spectrogram: 128 bands * 2 stats = 256."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features = {}
    for i in range(n_mels):
        features[f'mel{i}_mean'] = np.mean(mel_db[i])
        features[f'mel{i}_std'] = np.std(mel_db[i])
    return features


def extract_chroma_features(y, sr):
    """F) Chroma Features: 12 bands * 2 stats = 24."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features = {}
    for i in range(12):
        features[f'chroma{i}_mean'] = np.mean(chroma[i])
        features[f'chroma{i}_std'] = np.std(chroma[i])
    return features


def extract_all_features(y, sr):
    """Extract all feature sets and combine."""
    features = {}
    features.update(extract_mfcc_features(y, sr))
    features.update(extract_spectral_features(y, sr))
    features.update(extract_energy_features(y, sr))
    features.update(extract_respiratory_features(y, sr))
    features.update(extract_mel_features(y, sr))
    features.update(extract_chroma_features(y, sr))
    return features


def extract_features_from_file(audio_path):
    """Extract all features from a single audio file. Returns dict or None."""
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=10)
        if len(y) < sr * 0.2:
            return None
        return extract_all_features(y, sr)
    except Exception as e:
        print(f"Error extracting from {audio_path}: {e}")
        return None


def augment_and_extract(y, sr, label, augment=True):
    """Extract features from original + augmented versions for minority balancing."""
    results = []
    feats = extract_all_features(y, sr)
    if feats:
        feats['diagnosis'] = label
        results.append(feats)

    if augment:
        # Time stretch
        for rate in [0.8, 1.2]:
            try:
                y_aug = time_stretch(y, rate)
                f = extract_all_features(y_aug, sr)
                if f:
                    f['diagnosis'] = label
                    results.append(f)
            except Exception:
                pass

        # Pitch shift
        for n_steps in [-2, 2]:
            try:
                y_aug = pitch_shift(y, sr, n_steps)
                f = extract_all_features(y_aug, sr)
                if f:
                    f['diagnosis'] = label
                    results.append(f)
            except Exception:
                pass

        # White noise
        try:
            y_aug = add_white_noise(y, snr_db=20)
            f = extract_all_features(y_aug, sr)
            if f:
                f['diagnosis'] = label
                results.append(f)
        except Exception:
            pass

        # Time shift
        try:
            y_aug = time_shift(y)
            f = extract_all_features(y_aug, sr)
            if f:
                f['diagnosis'] = label
                results.append(f)
        except Exception:
            pass

    return results


from joblib import Parallel, delayed

def process_one_cycle(idx, total, row, minority_classes):
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{total} cycles...")

    audio_path = os.path.join(CYCLES_DIR, row['filename'])
    if not os.path.exists(audio_path):
        return []

    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=10)
        if len(y) < sr * 0.2:
            return []

        should_augment = row['diagnosis'] in minority_classes
        feats_list = augment_and_extract(y, sr, row['diagnosis'], augment=should_augment)
        return feats_list
    except Exception as e:
        print(f"  Skipping {row['filename']}: {e}")
        return []


def process_all_features():
    """Process all cycle files and save features to CSV."""
    csv_path = os.path.join(BASE_DIR, 'data', 'cycles.csv')
    if not os.path.exists(csv_path):
        print("No cycles.csv found. Run load_dataset.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} respiratory cycles...")

    # Determine minority classes for augmentation
    class_counts = df['diagnosis'].value_counts()
    max_count = class_counts.max()
    minority_classes = set(class_counts[class_counts < max_count * 0.7].index)

    print("Using multiprocessing with joblib (n_jobs=-1)...")
    total = len(df)
    results = Parallel(n_jobs=-1)(
        delayed(process_one_cycle)(idx, total, row, minority_classes)
        for idx, row in df.iterrows()
    )
    
    all_features = []
    for r in results:
        all_features.extend(r)

    if not all_features:
        print("No features extracted!")
        return

    features_df = pd.DataFrame(all_features)

    # Replace NaN/Inf
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)

    output_path = os.path.join(FEATURES_DIR, 'respiratory_features.csv')
    features_df.to_csv(output_path, index=False)

    n_features = len(features_df.columns) - 1  # minus diagnosis column
    print(f"\nFeature extraction complete!")
    print(f"Total samples: {len(features_df)}")
    print(f"Total features per sample: {n_features}")
    print(f"Saved to {output_path}")

    return features_df


if __name__ == '__main__':
    process_all_features()
