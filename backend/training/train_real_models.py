"""
VisionX — Real Model Training Pipeline
Trains XGBoost (decision winner prediction) + KMeans (user clustering)
on the universal real-world feature matrix.

Replaces the synthetic training/train_models.py.
Saves models to the SAME paths as the original so main.py loads them unchanged.

Run: python training/train_real_models.py
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, roc_auc_score, silhouette_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings, create_directories


FEATURES = ["value_score", "quality_score", "growth_score",
            "risk_score", "fit_score", "speed_score"]
TARGET   = "winner"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = os.path.join(settings.PROCESSED_DATA_DIR, "universal_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature matrix not found: {path}\n"
            "Run: python training/engineer_features.py"
        )
    df = pd.read_csv(path)
    print(f"✓ Loaded {len(df):,} rows from {path}")
    print(f"  Domains: {df['domain'].value_counts().to_dict()}")
    print(f"  Class balance: {df[TARGET].value_counts().to_dict()}")
    return df


# ─── Clustering ───────────────────────────────────────────────────────────────

def train_clustering(X: np.ndarray, save_path: str) -> KMeans:
    """
    Train KMeans on the 6 universal features.
    Clusters reflect real decision patterns, not hardcoded archetypes.
    """
    print("\n" + "=" * 65)
    print("CLUSTERING  (K-Means on real decision patterns)")
    print("=" * 65)

    # Elbow + silhouette to find optimal K
    inertias, sil_scores = [], []
    K_range = range(2, 8)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        print(f"  K={k}  inertia={km.inertia_:,.0f}  silhouette={sil_scores[-1]:.4f}")

    # Save elbow chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(K_range), inertias, "bo-"); ax1.set_title("Elbow"); ax1.grid(True)
    ax2.plot(list(K_range), sil_scores, "go-"); ax2.set_title("Silhouette"); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(settings.MODEL_DIR, "elbow_analysis.png"), dpi=150)
    plt.close()

    # Use settings.N_CLUSTERS (4) — matches existing API labels
    k = settings.N_CLUSTERS
    print(f"\n  Training final K-Means with K={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"  ✓ Final silhouette score: {sil:.4f}")

    # Cluster profile summary
    df_clust = pd.DataFrame(X, columns=FEATURES)
    df_clust["cluster"] = labels
    print("\n  Cluster profiles (feature means):")
    profile = df_clust.groupby("cluster")[FEATURES].mean().round(3)
    print(profile.to_string())

    # PCA visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="viridis", alpha=0.5, s=20)
    plt.colorbar(scatter, label="Cluster")
    ctrs = pca.transform(km.cluster_centers_)
    plt.scatter(ctrs[:, 0], ctrs[:, 1], c="red", marker="X", s=300,
                edgecolors="black", linewidths=2, label="Centers", zorder=5)
    plt.title("Decision Pattern Clusters (PCA projection of real data)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(settings.MODEL_DIR, "cluster_visualization.png"), dpi=150)
    plt.close()

    joblib.dump(km, save_path)
    print(f"\n  ✓ Saved clustering model → {save_path}")
    return km, labels, sil


# ─── XGBoost prediction model ─────────────────────────────────────────────────

def train_prediction(X: np.ndarray, y: np.ndarray, save_path: str):
    """
    Train XGBoost to predict whether an option is a winner.
    Target = 1 if option ranks in top 40% by value+quality+fit (real signal).
    """
    print("\n" + "=" * 65)
    print("PREDICTION  (XGBoost — real winner classification)")
    print("=" * 65)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")

    # Class weight to handle imbalance
    neg, pos = np.bincount(y_tr)
    scale_pos = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              verbose=False)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average='weighted')
    auc  = roc_auc_score(y_te, y_proba)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=["Not Winner", "Winner"]))

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Winner", "Winner"],
                yticklabels=["Not Winner", "Winner"])
    plt.title("Confusion Matrix — Real Data")
    plt.tight_layout()
    plt.savefig(os.path.join(settings.MODEL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    bars = plt.barh(FEATURES, importances, color="steelblue")
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance (Real Data)")
    plt.tight_layout()
    plt.savefig(os.path.join(settings.MODEL_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    fi_dict = {FEATURES[i]: float(importances[i]) for i in range(len(FEATURES))}
    print(f"\n  Feature Importances: {fi_dict}")

    joblib.dump(model, save_path)
    print(f"\n  ✓ Saved prediction model → {save_path}")
    return model, acc, f1, auc, fi_dict


# ─── Scaler ───────────────────────────────────────────────────────────────────

def fit_save_scaler(X: np.ndarray, save_path: str) -> StandardScaler:
    """
    Fit a StandardScaler on the 6 universal features.
    This scaler is used at inference time by the new predict.py.
    """
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, save_path)
    print(f"  ✓ Saved scaler → {save_path}")
    return scaler


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("VisionX — Real Model Training Pipeline")
    print("=" * 65)

    create_directories()

    # 1. Load
    df = load_data()

    X_raw = df[FEATURES].values.astype(float)
    y     = df[TARGET].values.astype(int)

    # 2. Scale (for clustering — KMeans is distance-based)
    scaler = fit_save_scaler(X_raw, settings.SCALER_PATH)
    X_scaled = scaler.transform(X_raw)

    # Save feature column list (same format as original pipeline)
    feature_cols_path = os.path.join(settings.MODEL_DIR, "feature_columns.pkl")
    joblib.dump(FEATURES, feature_cols_path)
    print(f"  ✓ Saved feature columns → {feature_cols_path}")

    # 3. Train clustering (uses scaled X)
    km, cluster_labels, sil = train_clustering(X_scaled, settings.CLUSTERING_MODEL_PATH)

    # 4. Train prediction model (uses raw X — tree models don't need scaling)
    model, acc, f1, auc, fi = train_prediction(X_raw, y, settings.PREDICTION_MODEL_PATH)

    # 5. Save training results (same schema as original training_results.json)
    results = {
        "training_date": datetime.now().isoformat(),
        "data_source": "real_world_multi_domain",
        "models_trained": ["clustering", "prediction"],
        "dataset_size": len(df),
        "domain_distribution": df["domain"].value_counts().to_dict(),
        "random_seed": 42,
        "features": FEATURES,
        "results": {
            "clustering": {
                "silhouette_score": float(sil),
                "n_clusters": settings.N_CLUSTERS,
                "cluster_distribution": {
                    str(k): int(v)
                    for k, v in zip(*np.unique(cluster_labels, return_counts=True))
                }
            },
            "prediction": {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "roc_auc":  float(auc),
                "feature_importance": fi
            }
        }
    }

    results_path = os.path.join(settings.MODEL_DIR, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved → {results_path}")

    print("\n" + "=" * 65)
    print("✓ TRAINING COMPLETE")
    print(f"  Clustering silhouette : {sil:.4f}")
    print(f"  Prediction accuracy   : {acc:.4f}")
    print(f"  Prediction F1         : {f1:.4f}")
    print(f"  Prediction ROC-AUC    : {auc:.4f}")
    print(f"\n  Models saved to: {settings.MODEL_DIR}")
    print("\nNext: uvicorn app.main:app --reload --app-dir backend")
    print("=" * 65)


if __name__ == "__main__":
    main()