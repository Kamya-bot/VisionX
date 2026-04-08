"""
VisionX — Dynamic Cluster Profiler
Auto-generates cluster labels and characteristics directly from
the trained KMeans model's cluster centers.
NO hardcoded labels anywhere.
"""

from __future__ import annotations
import numpy as np
import joblib
import json
import os
from typing import Dict, List, Tuple

FEATURE_NAMES = ["value_score", "quality_score", "growth_score",
                 "risk_score", "fit_score", "speed_score"]

# Thresholds for describing feature levels
HIGH = 0.15   # z-score above this = "high"
LOW  = -0.15  # z-score below this = "low"

# Human-readable descriptors per feature
FEATURE_HIGH = {
    "value_score":   ("Value-Conscious",    "prioritises cost-effectiveness"),
    "quality_score": ("Quality-Focused",    "prioritises objective quality"),
    "growth_score":  ("Growth-Oriented",    "prioritises future potential"),
    "risk_score":    ("Risk-Averse",        "avoids high-risk options"),
    "fit_score":     ("Socially-Validated", "values popularity and fit"),
    "speed_score":   ("Speed-Driven",       "prioritises fast outcomes"),
}
FEATURE_LOW = {
    "value_score":   ("Premium Buyer",      "willing to pay above market"),
    "quality_score": ("Budget Pragmatist",  "accepts lower quality for cost savings"),
    "growth_score":  ("Stability-Seeker",   "prefers stable over high-growth"),
    "risk_score":    ("Risk-Tolerant",      "comfortable with uncertain options"),
    "fit_score":     ("Independent Thinker","ignores social proof"),
    "speed_score":   ("Long-term Planner",  "willing to wait for best outcome"),
}


def _describe_center(center: np.ndarray) -> Tuple[str, List[str]]:
    """
    Given a cluster center vector (6 features, z-scored),
    return (label, [characteristics]) derived purely from the data.
    """
    dominant = []
    characteristics = []

    for i, fname in enumerate(FEATURE_NAMES):
        val = center[i]
        if val > HIGH and fname in FEATURE_HIGH:
            label_part, char = FEATURE_HIGH[fname]
            dominant.append((val, label_part))
            characteristics.append(char)
        elif val < LOW and fname in FEATURE_LOW:
            label_part, char = FEATURE_LOW[fname]
            dominant.append((abs(val), label_part))
            characteristics.append(char)

    if not dominant:
        # Flat profile — balanced decision maker
        return "Balanced Decision Maker", ["considers all factors equally",
                                            "no single dominant priority",
                                            "well-rounded approach"]

    # Sort by signal strength, take top 2 for label
    dominant.sort(reverse=True)
    label_parts = [d[1] for d in dominant[:2]]
    label = " & ".join(label_parts)

    if not characteristics:
        characteristics = ["balanced across all dimensions"]

    return label, characteristics[:4]


def generate_cluster_profiles(model_path: str, scaler_path: str | None = None) -> Dict:
    """
    Load KMeans model, read cluster centers, auto-generate labels.

    Returns dict:
    {
      cluster_id (int): {
        "label": str,
        "characteristics": [str],
        "center": {feature: value},
        "dominant_features": [str]
      }
    }
    """
    km = joblib.load(model_path)
    centers = km.cluster_centers_  # shape (n_clusters, n_features)

    # If scaler provided AND has 6 features, inverse-transform to 0-1 space
    # Otherwise use raw z-scored centers for description
    if scaler_path and os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            if hasattr(scaler, 'mean_') and len(scaler.mean_) == 6:
                centers_01 = scaler.inverse_transform(centers)
            else:
                centers_01 = None
        except Exception:
            centers_01 = None
    else:
        centers_01 = None

    profiles = {}
    for cluster_id, center in enumerate(centers):
        label, characteristics = _describe_center(center)

        # Center values in 0-1 space (for display)
        if centers_01 is not None:
            center_display = {
                fname: round(float(np.clip(centers_01[cluster_id][i], 0, 1)), 3)
                for i, fname in enumerate(FEATURE_NAMES)
            }
        else:
            # Normalize z-scores to approx 0-1 for display
            center_display = {
                fname: round(float(np.clip((center[i] + 2) / 4, 0, 1)), 3)
                for i, fname in enumerate(FEATURE_NAMES)
            }

        # Dominant features = top 2 by absolute z-score
        top_indices = np.argsort(np.abs(center))[::-1][:2]
        dominant = [FEATURE_NAMES[i] for i in top_indices]

        profiles[cluster_id] = {
            "label":            label,
            "characteristics":  characteristics,
            "center":           center_display,
            "dominant_features": dominant,
        }

    return profiles


def save_cluster_profiles(profiles: Dict, output_path: str):
    """Save profiles to JSON for use by the API."""
    serializable = {str(k): v for k, v in profiles.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"✓ Cluster profiles saved → {output_path}")


def load_cluster_profiles(profiles_path: str) -> Dict:
    """Load profiles from JSON. Returns {int: profile_dict}."""
    with open(profiles_path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def get_or_generate_profiles(model_path: str, scaler_path: str,
                              profiles_path: str) -> Dict:
    """
    Load profiles from cache if available, else generate and save.
    Called at API startup.
    """
    if os.path.exists(profiles_path):
        try:
            return load_cluster_profiles(profiles_path)
        except Exception:
            pass  # regenerate if corrupt

    profiles = generate_cluster_profiles(model_path, scaler_path)
    save_cluster_profiles(profiles, profiles_path)
    return profiles


# ── CLI: run standalone to inspect cluster labels ─────────────────────────────
if __name__ == "__main__":
    import sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_p  = os.path.join(base, "trained_models", "clustering.pkl")
    scaler_p = os.path.join(base, "trained_models", "scaler.pkl")
    out_p    = os.path.join(base, "trained_models", "cluster_profiles.json")

    if not os.path.exists(model_p):
        print(f"Model not found: {model_p}")
        sys.exit(1)

    profiles = generate_cluster_profiles(model_p, scaler_p)
    save_cluster_profiles(profiles, out_p)

    print("\n=== Auto-Generated Cluster Labels ===")
    for cid, prof in profiles.items():
        print(f"\nCluster {cid}: {prof['label']}")
        print(f"  Dominant features : {prof['dominant_features']}")
        print(f"  Characteristics   : {prof['characteristics']}")
        print(f"  Center (0-1 scale): {prof['center']}")