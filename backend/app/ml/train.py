"""
VisionX — ML Retraining Pipeline
Trains XGBoost (prediction) + KMeans (clustering) on:
  1. The original 8507-sample synthetic dataset (always included)
  2. Any real user feedback in OutcomeFeedback (merged in when >= 50 samples)

Saves:
  trained_models/prediction.pkl   — XGBoost classifier
  trained_models/clustering.pkl   — KMeans (4 clusters)
  trained_models/scaler.pkl       — StandardScaler
  trained_models/feature_columns.pkl
  trained_models/training_results.json
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "value_score", "quality_score", "growth_score",
    "risk_score", "fit_score", "speed_score",
]

MODELS_DIR = Path(__file__).parent.parent.parent / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 4
CLUSTER_LABELS = {
    0: {"label": "Independent Thinker & Risk-Averse",      "dominant_features": ["fit_score", "risk_score"]},
    1: {"label": "Growth-Oriented & Value-Conscious",       "dominant_features": ["growth_score", "value_score"]},
    2: {"label": "Budget Pragmatist & Stability-Seeker",    "dominant_features": ["quality_score", "growth_score"]},
    3: {"label": "Socially-Validated & Speed-Driven",       "dominant_features": ["fit_score", "speed_score"]},
}


# ── Synthetic dataset ────────────────────────────────────────────────────────

def _generate_synthetic_data(n: int = 8507, seed: int = 42) -> tuple:
    """
    Generates the same reproducible dataset that the original model was trained on.
    Each row is one option in a comparison. Label=1 means "this option won".
    """
    rng = np.random.default_rng(seed)

    # Build 4 cluster centroids
    centroids = np.array([
        [0.35, 0.45, 0.30, 0.60, 0.70, 0.40],  # Independent / Risk-Averse
        [0.70, 0.65, 0.75, 0.35, 0.50, 0.55],  # Growth / Value
        [0.55, 0.80, 0.60, 0.25, 0.45, 0.50],  # Budget / Stability
        [0.50, 0.55, 0.45, 0.40, 0.80, 0.75],  # Social / Speed
    ])

    X_parts, y_parts = [], []
    per_cluster = n // N_CLUSTERS
    for c_idx, centroid in enumerate(centroids):
        n_c = per_cluster + (n % N_CLUSTERS if c_idx == N_CLUSTERS - 1 else 0)
        samples = rng.normal(loc=centroid, scale=0.18, size=(n_c, 6))
        samples = np.clip(samples, 0, 1)

        # Label: weighted score per cluster preference
        weights = _cluster_weights(c_idx)
        scores = samples @ weights
        # Soft threshold at median
        labels = (scores > np.median(scores)).astype(int)
        X_parts.append(samples)
        y_parts.append(labels)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def _cluster_weights(cluster_id: int) -> np.ndarray:
    """Per-cluster feature weights for synthetic label generation."""
    w = {
        0: [0.15, 0.25, 0.10, -0.30, 0.35, 0.15],  # values fit, penalises risk
        1: [0.35, 0.25, 0.30,  0.00, 0.05, 0.05],  # values growth + value
        2: [0.20, 0.40, 0.25, -0.05, 0.05, 0.05],  # quality-first
        3: [0.10, 0.15, 0.10,  0.05, 0.35, 0.25],  # social + speed
    }
    arr = np.array(w[cluster_id])
    return arr / np.abs(arr).sum()


# ── Real feedback loader ─────────────────────────────────────────────────────

def _load_feedback_from_db(min_samples: int = 50) -> Optional[tuple]:
    """
    Loads OutcomeFeedback rows from the DB.
    Returns (X, y) numpy arrays or None if < min_samples.
    """
    try:
        from database import SessionLocal
        import models as m

        db = SessionLocal()
        try:
            rows = (
                db.query(m.OutcomeFeedback)
                .filter(m.OutcomeFeedback.features_snapshot.isnot(None))
                .order_by(m.OutcomeFeedback.created_at.desc())
                .limit(10000)
                .all()
            )
            if len(rows) < min_samples:
                logger.info(f"Only {len(rows)} feedback rows — skipping real data merge")
                return None

            X_rows, y_rows = [], []
            for row in rows:
                snap = row.features_snapshot
                if not isinstance(snap, dict):
                    continue
                vec = [float(snap.get(f, 0.5)) for f in FEATURE_NAMES]
                X_rows.append(vec)
                y_rows.append(int(bool(row.accepted)))

            if len(X_rows) < min_samples:
                return None

            logger.info(f"Loaded {len(X_rows)} real feedback samples for retraining")
            return np.array(X_rows), np.array(y_rows)
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Could not load feedback from DB: {e}")
        return None


# ── Training ─────────────────────────────────────────────────────────────────

def train(force_synthetic_only: bool = False) -> dict:
    """
    Full training run. Returns metrics dict.
    """
    t0 = time.time()
    logger.info("Starting VisionX ML training pipeline")

    # 1. Build dataset
    X_syn, y_syn = _generate_synthetic_data()
    X, y = X_syn, y_syn

    if not force_synthetic_only:
        real = _load_feedback_from_db(min_samples=50)
        if real is not None:
            X_real, y_real = real
            # Weight real data 3x: repeat it
            X = np.vstack([X_syn, X_real, X_real, X_real])
            y = np.concatenate([y_syn, y_real, y_real, y_real])
            logger.info(f"Merged dataset: {len(X_syn)} synthetic + {len(X_real)} real (3x weighted)")

    # 2. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )

    # 4. XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )

    # 5. Platt scaling (real calibration)
    calibrated = CalibratedClassifierCV(xgb, cv="prefit", method="sigmoid")
    calibrated.fit(X_te, y_te)

    # 6. Metrics
    y_pred = calibrated.predict(X_te)
    y_proba = calibrated.predict_proba(X_te)[:, 1]
    accuracy = float(accuracy_score(y_te, y_pred))
    roc_auc = float(roc_auc_score(y_te, y_proba))

    # 7. KMeans clustering (on raw features — NOT scaled)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    kmeans.fit(X[:, :])  # cluster on raw feature space

    # 8. Feature importance
    fi = xgb.feature_importances_
    feature_importance = {FEATURE_NAMES[i]: round(float(fi[i]), 4) for i in range(len(FEATURE_NAMES))}
    top_feature = max(feature_importance, key=feature_importance.get)

    # 9. Save models
    with open(MODELS_DIR / "prediction.pkl", "wb") as f:
        pickle.dump(calibrated, f, protocol=4)
    with open(MODELS_DIR / "clustering.pkl", "wb") as f:
        pickle.dump(kmeans, f, protocol=4)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f, protocol=4)
    with open(MODELS_DIR / "feature_columns.pkl", "wb") as f:
        pickle.dump(FEATURE_NAMES, f, protocol=4)

    elapsed = round(time.time() - t0, 2)
    results = {
        "trained_at": datetime.utcnow().isoformat(),
        "duration_seconds": elapsed,
        "dataset_size": len(X),
        "synthetic_size": len(X_syn),
        "real_feedback_size": len(X) - len(X_syn),
        "accuracy": round(accuracy, 6),
        "roc_auc": round(roc_auc, 6),
        "feature_importance": feature_importance,
        "top_predictive_feature": top_feature,
        "n_clusters": N_CLUSTERS,
        "cluster_labels": CLUSTER_LABELS,
        "calibration": "platt_sigmoid",
        "model_type": "XGBoost_CalibratedCV",
    }

    with open(MODELS_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"Training complete in {elapsed}s — "
        f"accuracy={accuracy:.4f}, roc_auc={roc_auc:.4f}, "
        f"top_feature={top_feature}"
    )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = train()
    print(json.dumps(results, indent=2))