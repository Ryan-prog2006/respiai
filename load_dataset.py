"""
load_dataset.py — ICBHI 2017 Respiratory Sound Database Loader

Downloads the ICBHI 2017 dataset, reads audio + annotation files,
splits audio into individual respiratory cycles, and creates a master CSV.
"""

import os
import glob
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
ANNOT_DIR = os.path.join(DATA_DIR, 'annotations')
CYCLES_DIR = os.path.join(DATA_DIR, 'cycles')

# ICBHI diagnosis mapping from patient diagnosis file
DIAGNOSIS_MAP = {
    'COPD': 'COPD',
    'Healthy': 'Normal',
    'Bronchiectasis': 'Bronchiectasis',
    'Bronchiolitis': 'Bronchiolitis',
    'URTI': 'URTI',
    'LRTI': 'LRTI',
    'Pneumonia': 'Pneumonia',
    'Asthma': 'Asthma',
}

# Recording devices in the ICBHI dataset
DEVICES = {
    'AKGC417L': 'AKG C417L Microphone',
    'LittC2SE': 'Littmann Classic II SE Stethoscope',
    'Meditron': 'Meditron Electronic Stethoscope',
    '3M': '3M Littmann 3200',
}


def ensure_directories():
    """Create all necessary directories."""
    for d in [DATA_DIR, AUDIO_DIR, ANNOT_DIR, CYCLES_DIR]:
        os.makedirs(d, exist_ok=True)


def parse_filename(filename):
    """
    ICBHI filenames follow pattern: PatientID_RecordingIndex_ChestLocation_AcquisitionMode_Device
    Example: 101_1b1_Al_sc_Meditron.wav
    """
    parts = os.path.splitext(filename)[0].split('_')
    if len(parts) >= 5:
        return {
            'patient_id': parts[0],
            'recording_index': parts[1],
            'chest_location': parts[2],
            'acquisition_mode': parts[3],
            'device': parts[4] if len(parts) > 4 else 'Unknown'
        }
    return {
        'patient_id': parts[0] if len(parts) > 0 else 'Unknown',
        'recording_index': parts[1] if len(parts) > 1 else '0',
        'chest_location': 'Unknown',
        'acquisition_mode': 'Unknown',
        'device': 'Unknown'
    }


def read_annotation(annot_path):
    """
    Read annotation file. Each line: start_time end_time crackle wheeze
    """
    cycles = []
    try:
        with open(annot_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    cycles.append({
                        'start': float(parts[0]),
                        'end': float(parts[1]),
                        'crackle': int(parts[2]),
                        'wheeze': int(parts[3]),
                    })
    except Exception as e:
        print(f"Error reading {annot_path}: {e}")
    return cycles


def split_audio_into_cycles(audio_path, annot_path, diagnosis, patient_id):
    """
    Split a single audio recording into individual respiratory cycles
    based on annotation timestamps. Save each cycle as separate .wav.
    """
    records = []
    try:
        y, sr = librosa.load(audio_path, sr=22050)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return records

    cycles = read_annotation(annot_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    for i, cycle in enumerate(cycles):
        start_sample = int(cycle['start'] * sr)
        end_sample = int(cycle['end'] * sr)

        if end_sample > len(y):
            end_sample = len(y)
        if start_sample >= end_sample:
            continue

        cycle_audio = y[start_sample:end_sample]
        duration = len(cycle_audio) / sr

        if duration < 0.2:  # Skip very short cycles
            continue

        cycle_filename = f"{base_name}_cycle{i}.wav"
        cycle_path = os.path.join(CYCLES_DIR, cycle_filename)
        sf.write(cycle_path, cycle_audio, sr)

        records.append({
            'filename': cycle_filename,
            'patient_id': patient_id,
            'start': cycle['start'],
            'end': cycle['end'],
            'duration': duration,
            'crackle': cycle['crackle'],
            'wheeze': cycle['wheeze'],
            'diagnosis': diagnosis,
        })

    return records


def load_diagnosis_file(diag_path):
    """Load the patient diagnosis text file."""
    diag_map = {}
    if os.path.exists(diag_path):
        with open(diag_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    diag_map[parts[0]] = DIAGNOSIS_MAP.get(parts[1], parts[1])
    return diag_map


def create_train_test_split(df, test_ratio=0.2, seed=42):
    """Patient-wise stratified train/test split."""
    np.random.seed(seed)
    patients = df['patient_id'].unique()
    np.random.shuffle(patients)
    n_test = max(1, int(len(patients) * test_ratio))
    test_patients = set(patients[:n_test])
    df['split'] = df['patient_id'].apply(lambda x: 'test' if x in test_patients else 'train')
    return df


def process_dataset():
    """Process all audio files and annotations into respiratory cycles."""
    ensure_directories()

    print("=" * 60)
    print("  ICBHI 2017 Respiratory Sound Database Loader")
    print("=" * 60)

    # Check if data exists
    wav_files = glob.glob(os.path.join(AUDIO_DIR, '*.wav'))
    txt_files = glob.glob(os.path.join(ANNOT_DIR, '*.txt'))

    # Also check if annotations are in the audio directory (common layout)
    if not txt_files:
        txt_files = glob.glob(os.path.join(AUDIO_DIR, '*.txt'))
        ANNOT_DIR_ACTUAL = AUDIO_DIR
    else:
        ANNOT_DIR_ACTUAL = ANNOT_DIR

    if not wav_files:
        print(f"\n[!] No .wav files found in {AUDIO_DIR}")
        print("    Please download the ICBHI 2017 dataset from:")
        print("    https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge")
        print("    and place audio files in: data/audio/")
        print("    and annotation files in: data/annotations/")
        print("\n    Generating synthetic demo dataset for testing...\n")
        return generate_demo_dataset()

    # Load diagnosis file
    diag_path = os.path.join(DATA_DIR, 'patient_diagnosis.csv')
    if not os.path.exists(diag_path):
        diag_path = os.path.join(DATA_DIR, 'patient_diagnosis.txt')
    diag_map = load_diagnosis_file(diag_path)

    print(f"\nFound {len(wav_files)} audio files")
    print(f"Found {len(txt_files)} annotation files")

    all_records = []
    for wav_path in wav_files:
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        annot_path = os.path.join(ANNOT_DIR_ACTUAL, f"{base_name}.txt")

        if not os.path.exists(annot_path):
            continue

        info = parse_filename(os.path.basename(wav_path))
        patient_id = info['patient_id']
        diagnosis = diag_map.get(patient_id, 'Normal')

        records = split_audio_into_cycles(wav_path, annot_path, diagnosis, patient_id)
        all_records.extend(records)

    if not all_records:
        print("No cycles extracted. Check your data files.")
        return generate_demo_dataset()

    df = pd.DataFrame(all_records)
    df = create_train_test_split(df)

    csv_path = os.path.join(DATA_DIR, 'cycles.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved master CSV to {csv_path}")

    print_statistics(df)
    plot_class_distribution(df)
    return df


def generate_demo_dataset():
    """Generate a synthetic demo dataset for testing without real data."""
    print("Generating synthetic respiratory cycle audio files...")

    diseases = ['Normal', 'COPD', 'Pneumonia', 'Bronchiectasis',
                'Bronchiolitis', 'URTI', 'LRTI', 'Asthma']
    samples_per_class = 30
    sr = 22050
    records = []

    for disease in diseases:
        for i in range(samples_per_class):
            duration = np.random.uniform(1.0, 4.0)
            t = np.linspace(0, duration, int(sr * duration))

            # Create synthetic lung sounds with disease-specific characteristics
            base_freq = np.random.uniform(100, 300)
            y = 0.3 * np.sin(2 * np.pi * base_freq * t)

            if disease in ['COPD', 'Asthma']:
                # Add wheeze-like harmonics
                wheeze_freq = np.random.uniform(400, 1000)
                y += 0.15 * np.sin(2 * np.pi * wheeze_freq * t)
            elif disease in ['Pneumonia', 'Bronchiectasis']:
                # Add crackle-like noise bursts
                n_crackles = np.random.randint(5, 20)
                for _ in range(n_crackles):
                    pos = np.random.randint(0, len(t))
                    width = np.random.randint(50, 200)
                    end_pos = min(pos + width, len(t))
                    y[pos:end_pos] += np.random.randn(end_pos - pos) * 0.2
            elif disease in ['URTI', 'LRTI']:
                # Add noise
                y += np.random.randn(len(t)) * 0.08

            # Add breathing envelope
            breath_cycle = np.sin(2 * np.pi * 0.3 * t)
            y *= (0.5 + 0.5 * breath_cycle)

            # Normalize
            y = y / (np.max(np.abs(y)) + 1e-9)

            filename = f"demo_{disease}_{i:03d}.wav"
            filepath = os.path.join(CYCLES_DIR, filename)
            sf.write(filepath, y, sr)

            patient_id = f"P{diseases.index(disease):02d}{i // 5:02d}"
            crackle = 1 if disease in ['Pneumonia', 'Bronchiectasis', 'LRTI'] else 0
            wheeze = 1 if disease in ['COPD', 'Asthma', 'Bronchiolitis'] else 0

            records.append({
                'filename': filename,
                'patient_id': patient_id,
                'start': 0.0,
                'end': duration,
                'duration': duration,
                'crackle': crackle,
                'wheeze': wheeze,
                'diagnosis': disease,
            })

    df = pd.DataFrame(records)
    df = create_train_test_split(df)
    csv_path = os.path.join(DATA_DIR, 'cycles.csv')
    df.to_csv(csv_path, index=False)

    print(f"Generated {len(records)} synthetic cycles across {len(diseases)} classes")
    print(f"Saved master CSV to {csv_path}")

    print_statistics(df)
    plot_class_distribution(df)
    return df


def print_statistics(df):
    """Print comprehensive dataset statistics."""
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)

    print(f"\nTotal respiratory cycles: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")

    print("\n--- Cycles per Disease ---")
    disease_counts = df['diagnosis'].value_counts()
    for disease, count in disease_counts.items():
        pct = count / len(df) * 100
        bar = '█' * int(pct / 2)
        print(f"  {disease:<20} {count:>5} ({pct:5.1f}%) {bar}")

    print("\n--- Train/Test Split ---")
    split_counts = df['split'].value_counts()
    for split, count in split_counts.items():
        print(f"  {split:<10} {count:>5} cycles ({count/len(df)*100:.1f}%)")

    print("\n--- Crackle/Wheeze Distribution ---")
    print(f"  Crackle present: {df['crackle'].sum()} ({df['crackle'].mean()*100:.1f}%)")
    print(f"  Wheeze present:  {df['wheeze'].sum()} ({df['wheeze'].mean()*100:.1f}%)")


def plot_class_distribution(df):
    """Plot and save class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#00d4ff', '#7b2fff', '#ff4757', '#2ed573',
              '#ffa502', '#ff6b81', '#70a1ff', '#eccc68']
    disease_counts = df['diagnosis'].value_counts()
    bars = ax.bar(disease_counts.index, disease_counts.values, color=colors[:len(disease_counts)])
    ax.set_title('Respiratory Disease Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Disease', fontsize=12)
    ax.set_ylabel('Number of Cycles', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, disease_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha='center', fontweight='bold')
    plt.tight_layout()

    chart_path = os.path.join(DATA_DIR, 'class_distribution.png')
    plt.savefig(chart_path, dpi=150)
    print(f"\nClass distribution chart saved to {chart_path}")
    plt.close()


if __name__ == '__main__':
    process_dataset()
