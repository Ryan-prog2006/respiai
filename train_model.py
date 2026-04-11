"""
train_model.py — Ensemble Model Training for Respiratory Disease Detection

Trains RF, XGBoost, LightGBM, SVM, MLP with GridSearchCV.
Creates soft-voting ensemble from best 3 models. Target: >92% accuracy.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(BASE_DIR, 'features', 'respiratory_features.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def load_features():
    """Load the features CSV."""
    if not os.path.exists(FEATURES_PATH):
        print(f"Features file not found: {FEATURES_PATH}")
        print("Run extract_features.py first.")
        return None, None
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    return X, y


def preprocess(X, y):
    """Scale, encode, balance, and select features."""
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection
    n_select = min(200, X_scaled.shape[1])
    selector = SelectKBest(f_classif, k=n_select)
    X_selected = selector.fit_transform(X_scaled, y_enc)

    # SMOTE + RandomUnderSampler for balance
    smote = SMOTE(random_state=42, k_neighbors=min(5, min(pd.Series(y_enc).value_counts()) - 1))
    rus = RandomUnderSampler(random_state=42)

    X_bal, y_bal = smote.fit_resample(X_selected, y_enc)
    X_bal, y_bal = rus.fit_resample(X_bal, y_bal)

    print(f"After balancing: {len(X_bal)} samples")
    print(f"Feature dimensions: {X_bal.shape[1]}")
    return X_bal, y_bal, scaler, selector, le


def get_models():
    """Define models with hyperparameter grids."""
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {'n_estimators': [200, 500], 'max_depth': [None, 20]}
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False,
                                   eval_metric='mlogloss', verbosity=0),
            'params': {'n_estimators': [200, 500], 'learning_rate': [0.01, 0.1]}
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42, verbose=-1),
            'params': {'n_estimators': [200, 500], 'num_leaves': [31, 63]}
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {'C': [1, 10, 100], 'kernel': ['rbf']}
        },
        'MLP': {
            'model': MLPClassifier(random_state=42, max_iter=500),
            'params': {'hidden_layer_sizes': [(512, 256, 128), (256, 128, 64)]}
        },
    }
    return models


def train_and_evaluate():
    """Full training pipeline with GridSearchCV and ensemble creation."""
    print("=" * 60)
    print("  RespiAI — Model Training Pipeline")
    print("=" * 60)

    X, y = load_features()
    if X is None:
        return

    X_bal, y_bal, scaler, selector, le = preprocess(X, y)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=min(10, min(pd.Series(y_bal).value_counts())),
                          shuffle=True, random_state=42)

    models_config = get_models()
    results = {}
    trained_models = {}

    print("\nTraining models with GridSearchCV...\n")

    for name, config in models_config.items():
        print(f"--- {name} ---")
        try:
            grid = GridSearchCV(config['model'], config['params'],
                                cv=skf, scoring='accuracy', n_jobs=-1, verbose=0)
            grid.fit(X_bal, y_bal)
            best = grid.best_estimator_

            y_pred = best.predict(X_bal)
            acc = accuracy_score(y_bal, y_pred)
            prec = precision_score(y_bal, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_bal, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_bal, y_pred, average='weighted', zero_division=0)

            results[name] = {
                'accuracy': acc, 'precision': prec,
                'recall': rec, 'f1': f1,
                'cv_score': grid.best_score_,
                'best_params': grid.best_params_,
            }
            trained_models[name] = best

            print(f"  CV Score: {grid.best_score_:.4f} | Train Acc: {acc:.4f}")
            print(f"  Best Params: {grid.best_params_}")

        except Exception as e:
            print(f"  Error training {name}: {e}")

    if not trained_models:
        print("No models trained successfully.")
        return

    # Sort by CV score and create ensemble from top 3
    sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_score'], reverse=True)
    top_3 = sorted_models[:3]

    print(f"\n--- Model Comparison ---")
    print(f"{'Model':<20} {'CV Score':>10} {'Accuracy':>10} {'F1':>10}")
    print("-" * 55)
    for name, r in sorted_models:
        print(f"{name:<20} {r['cv_score']:>10.4f} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")

    # Create Soft Voting Ensemble
    ensemble_estimators = [(name, trained_models[name]) for name, _ in top_3]
    ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    ensemble.fit(X_bal, y_bal)
    y_pred_ens = ensemble.predict(X_bal)
    ens_acc = accuracy_score(y_bal, y_pred_ens)

    print(f"\n=== Ensemble (Top 3) Accuracy: {ens_acc:.4f} ===")

    # Confusion matrix
    cm = confusion_matrix(y_bal, y_pred_ens)
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_bal, y_pred_ens, target_names=le.classes_, zero_division=0))

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title('Confusion Matrix — Ensemble Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Save the best model (ensemble), or fall back to single best
    best_model = ensemble if ens_acc >= sorted_models[0][1]['accuracy'] else trained_models[sorted_models[0][0]]

    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_respiratory_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'respiratory_scaler.pkl'))
    joblib.dump(selector, os.path.join(MODELS_DIR, 'respiratory_selector.pkl'))
    joblib.dump(le, os.path.join(MODELS_DIR, 'respiratory_encoder.pkl'))

    print(f"\nModels saved to {MODELS_DIR}/")
    print(f"  best_respiratory_model.pkl")
    print(f"  respiratory_scaler.pkl")
    print(f"  respiratory_selector.pkl")
    print(f"  respiratory_encoder.pkl")


if __name__ == '__main__':
    train_and_evaluate()
