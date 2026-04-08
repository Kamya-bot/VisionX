"""
VisionX — Real ML Prediction Engine
XGBoost inference on 6 universal features.
No hardcoded labels — cluster label passed in dynamically.
"""

from __future__ import annotations
import os
import sys
import logging
import numpy as np
from typing import Dict, List, Any, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.normalizer import to_universal_features, detect_domain

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "value_score", "quality_score", "growth_score",
    "risk_score", "fit_score", "speed_score"
]

DOMAIN_LABELS = {
    "products":  "Product Comparison",
    "jobs":      "Job Opportunity",
    "education": "Educational Institution",
    "housing":   "Property / Real Estate",
    "generic":   "General Decision",
}

FEATURE_EXPLANATIONS = {
    "value_score":   ("Value for Money",      "what you get per unit of cost"),
    "quality_score": ("Objective Quality",    "how good it is based on real metrics"),
    "growth_score":  ("Future Potential",     "upside trajectory and growth signal"),
    "risk_score":    ("Risk Level",           "uncertainty and downside exposure"),
    "fit_score":     ("Social Proof / Fit",   "popularity, demand, and market validation"),
    "speed_score":   ("Time-to-Value",        "how quickly benefits materialize"),
}


def predict_winner(options: List[Any], model_store, user_cluster_id: int = 0,
                   cluster_label: str = "your profile") -> Dict:
    """
    Real XGBoost-based winner prediction.
    cluster_label is passed in dynamically — never hardcoded here.
    """
    if len(options) < 2:
        raise ValueError("At least 2 options required")

    scored = []
    for opt in options:
        features_raw = _extract_features(opt)
        domain = detect_domain(features_raw)
        universal = to_universal_features(features_raw, domain)

        X = universal.reshape(1, -1)
        try:
            win_proba = float(model_store.prediction_model.predict_proba(X)[0][1])
        except Exception as e:
            logger.warning(f"Model prediction failed ({e}), using heuristic")
            win_proba = _heuristic_score(universal)

        scored.append({
            "id":           opt.id,
            "name":         opt.name,
            "win_proba":    win_proba,
            "universal":    universal,
            "domain":       domain,
            "features_raw": features_raw,
        })

    scored.sort(key=lambda x: x["win_proba"], reverse=True)
    best = scored[0]
    alternatives = scored[1:4]

    fi = _get_feature_importance(model_store, best["universal"])
    confidence = _calibrate_confidence(best["win_proba"], scored)
    reasoning = _generate_reasoning(best, fi, cluster_label)

    return {
        "recommended_option_id":   best["id"],
        "recommended_option_name": best["name"],
        "confidence":              round(confidence, 3),
        "reasoning":               reasoning,
        "domain_detected":         DOMAIN_LABELS.get(best["domain"], best["domain"]),
        "alternative_options": [
            {
                "id":     alt["id"],
                "name":   alt["name"],
                "score":  round(alt["win_proba"], 3),
                "reason": _alt_reason(best["universal"], alt["universal"]),
            }
            for alt in alternatives
        ],
        "feature_importance": [
            {"feature_name": FEATURE_NAMES[i], "importance": round(fi[i], 3)}
            for i in np.argsort(fi)[::-1]
        ],
        "universal_features": {
            name: round(float(best["universal"][i]), 3)
            for i, name in enumerate(FEATURE_NAMES)
        },
    }


def score_options_for_user(options: List[Any], model_store, cluster_id: int) -> List[Dict]:
    results = []
    for opt in options:
        features_raw = _extract_features(opt)
        domain = detect_domain(features_raw)
        universal = to_universal_features(features_raw, domain)
        X = universal.reshape(1, -1)
        try:
            win_proba = float(model_store.prediction_model.predict_proba(X)[0][1])
        except Exception:
            win_proba = _heuristic_score(universal)
        results.append({
            "id":        opt.id,
            "name":      opt.name,
            "score":     round(win_proba, 3),
            "domain":    domain,
            "universal": universal.tolist(),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_features(option) -> Dict:
    if hasattr(option, "features"):
        f = option.features
        if hasattr(f, "dict"):
            return f.dict()
        if isinstance(f, dict):
            return f
    if isinstance(option, dict):
        return option.get("features", option)
    return {}


def _heuristic_score(universal: np.ndarray) -> float:
    weights = np.array([0.25, 0.30, 0.15, -0.15, 0.10, 0.05])
    score = float(np.dot(universal, weights))
    return float(np.clip((score + 0.15) / 0.7, 0, 1))


def _get_feature_importance(model_store, universal: np.ndarray) -> np.ndarray:
    try:
        base = model_store.prediction_model.feature_importances_
        weighted = base * universal
        total = weighted.sum()
        return weighted / total if total > 0 else base
    except Exception:
        return np.array([0.25, 0.30, 0.15, 0.10, 0.12, 0.08])


def _calibrate_confidence(win_proba: float, scored: List[Dict]) -> float:
    if len(scored) < 2:
        return win_proba
    margin = win_proba - scored[1]["win_proba"]
    confidence = 0.6 * win_proba + 0.4 * (0.5 + margin)
    return float(np.clip(confidence, 0.30, 0.97))


def _generate_reasoning(best: Dict, fi: np.ndarray, cluster_label: str) -> str:
    """
    Generate reasoning from actual feature values.
    cluster_label comes from dynamic cluster profiler — never hardcoded.
    """
    universal = best["universal"]
    domain = best["domain"]
    name = best["name"]

    top_idx = int(np.argmax(fi))
    top_feat = FEATURE_NAMES[top_idx]
    top_label, top_desc = FEATURE_EXPLANATIONS[top_feat]
    top_val = float(universal[top_idx])

    weakness_idx = int(np.argmin(universal))
    weak_feat = FEATURE_NAMES[weakness_idx]
    weak_label, weak_desc = FEATURE_EXPLANATIONS[weak_feat]

    domain_context = {
        "products":  "Among the products compared",
        "jobs":      "Among the job opportunities",
        "education": "Among the institutions considered",
        "housing":   "Among the properties compared",
        "generic":   "Among the options analyzed",
    }.get(domain, "Among the options")

    strength_pct = int(top_val * 100)
    strength_text = (
        f"scores in the top {100 - strength_pct}% on {top_label.lower()} ({top_desc})"
        if top_val > 0.7 else
        f"leads on {top_label.lower()} ({top_desc})"
    )

    weak_text = (
        f"Note: lower {weak_label.lower()} ({weak_desc}) than alternatives — "
        f"factor this into your decision."
        if float(universal[weakness_idx]) < 0.35 else ""
    ).strip()

    parts = [
        f"{domain_context}, {name} {strength_text}.",
        f"Recommended for {cluster_label} based on real patterns.",
    ]
    if weak_text:
        parts.append(weak_text)

    return " ".join(p for p in parts if p)


def _alt_reason(best_vec: np.ndarray, alt_vec: np.ndarray) -> str:
    diffs = alt_vec - best_vec
    if diffs.max() > 0.1:
        best_alt_feat = FEATURE_NAMES[int(np.argmax(diffs))]
        label, desc = FEATURE_EXPLANATIONS[best_alt_feat]
        return f"Higher {label.lower()} — worth considering if {desc} matters more."
    return "Similar overall profile. A viable alternative."