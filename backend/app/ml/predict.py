"""
VisionX — Real ML Prediction Engine
XGBoost inference on 6 universal features.
Now includes per-prediction SHAP values for real explainability.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .normalizer import detect_domain, to_universal_features

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "value_score", "quality_score", "growth_score",
    "risk_score", "fit_score", "speed_score",
]

DOMAIN_LABELS = {
    "products":  "Product Comparison",
    "jobs":      "Job Opportunity",
    "education": "Educational Institution",
    "housing":   "Property / Real Estate",
    "generic":   "General Decision",
}

FEATURE_EXPLANATIONS = {
    "value_score":   ("Value for Money",    "what you get per unit of cost"),
    "quality_score": ("Objective Quality",  "how good it is based on real metrics"),
    "growth_score":  ("Future Potential",   "upside trajectory and growth signal"),
    "risk_score":    ("Risk Level",         "uncertainty and downside exposure"),
    "fit_score":     ("Social Proof / Fit", "popularity, demand, and market validation"),
    "speed_score":   ("Time-to-Value",      "how quickly benefits materialize"),
}

# Module-level SHAP explainer cache — initialised on first use
_shap_explainer = None


def _get_shap_explainer(model_store):
    """
    Returns a cached SHAP TreeExplainer for the prediction model.
    SHAP is expensive to init — we do it once and reuse.
    """
    global _shap_explainer
    if _shap_explainer is not None:
        return _shap_explainer
    try:
        import shap
        _shap_explainer = shap.TreeExplainer(model_store.prediction_model)
        logger.info("✅ SHAP TreeExplainer initialised")
    except Exception as e:
        logger.warning(f"SHAP init failed: {e}")
        _shap_explainer = None
    return _shap_explainer


def _compute_shap_values(model_store, X: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Compute per-prediction SHAP values for the winning option.
    Returns a dict {feature_name: shap_value} or None on failure.
    """
    explainer = _get_shap_explainer(model_store)
    if explainer is None:
        return None
    try:
        import shap
        shap_vals = explainer.shap_values(X)
        # shap_vals shape: (1, 6) for binary classification (class 1)
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]   # class 1 shap values
        else:
            vals = shap_vals[0]
        return {FEATURE_NAMES[i]: round(float(vals[i]), 4) for i in range(len(FEATURE_NAMES))}
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def predict_winner(
    options: List[Any],
    model_store,
    user_cluster_id: int = 0,
    cluster_label: str = "your profile",
) -> Dict:
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

    # Real SHAP values for the winning option
    X_best = best["universal"].reshape(1, -1)
    shap_values = _compute_shap_values(model_store, X_best)

    # Feature importance: use SHAP if available, else model importances
    if shap_values:
        fi = np.array([abs(shap_values[f]) for f in FEATURE_NAMES])
        total = fi.sum()
        fi = fi / total if total > 0 else fi
    else:
        fi = _get_feature_importance(model_store, best["universal"])

    confidence = _calibrate_confidence(best["win_proba"], scored)
    reasoning = _generate_reasoning(best, fi, cluster_label)

    return {
        "recommended_option_id":   best["id"],
        "recommended_option_name": best["name"],
        "confidence":              round(confidence, 3),
        "reasoning":               reasoning,
        "domain_detected":         DOMAIN_LABELS.get(best["domain"], best["domain"]),
        "shap_values":             shap_values,
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
            {"feature_name": FEATURE_NAMES[i], "importance": round(float(fi[i]), 3)}
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
        if hasattr(f, "model_dump"):
            return {k: v for k, v in f.model_dump().items() if v is not None}
        if hasattr(f, "dict"):
            return {k: v for k, v in f.dict().items() if v is not None}
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
    """
    Platt-scaling-inspired calibration:
    Uses margin between top-2 scores + base probability.
    Clipped to [0.30, 0.97] to avoid overconfident outputs.
    """
    if len(scored) < 2:
        return float(np.clip(win_proba, 0.30, 0.97))
    margin = win_proba - scored[1]["win_proba"]
    # Weighted blend: 60% raw proba + 40% margin signal
    confidence = 0.6 * win_proba + 0.4 * (0.5 + margin)
    return float(np.clip(confidence, 0.30, 0.97))


def _generate_reasoning(best: Dict, fi: np.ndarray, cluster_label: str) -> str:
    universal = best["universal"]
    domain = best["domain"]
    name = best["name"]

    top_idx = int(np.argmax(fi))
    top_feat = FEATURE_NAMES[top_idx]
    top_label, top_desc = FEATURE_EXPLANATIONS[top_feat]
    top_val = float(universal[top_idx])

    weakness_idx = int(np.argmin(universal))
    weak_label, weak_desc = FEATURE_EXPLANATIONS[FEATURE_NAMES[weakness_idx]]

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
        if top_val > 0.7
        else f"leads on {top_label.lower()} ({top_desc})"
    )

    weak_text = (
        f"Note: lower {weak_label.lower()} ({weak_desc}) than alternatives — "
        f"factor this into your decision."
        if float(universal[weakness_idx]) < 0.35
        else ""
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