"""
VisionX — Universal Decision Feature Normalizer
Maps raw user inputs (any domain) into the 6-dimensional feature vector
that the real XGBoost model was trained on.

Supported domains (auto-detected from input keys):
  - products    (price, quality_score, feature_count, brand_score, delivery_time)
  - jobs        (salary, company_rating, seniority_level, company_size, remote)
  - education   (tuition, ranking, research_score, teaching_score, acceptance_rate)
  - housing     (price, overall_quality, area, year_built, neighborhood_score)
  - generic     (any 6 numeric features — normalized as-is)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


# ─── Domain detection ──────────────────────────────────────────────────────────

DOMAIN_SIGNALS = {
    "products": {"price", "quality_score", "feature_count", "brand_score",
                 "delivery_time", "rating", "reviews"},
    "jobs":     {"salary", "company_rating", "seniority_level", "company_size",
                 "remote", "work_type", "max_salary", "min_salary"},
    "education":{"tuition", "ranking", "world_rank", "research_score",
                 "teaching_score", "acceptance_rate", "citations"},
    "housing":  {"sqft", "area", "bedrooms", "bathrooms", "year_built",
                 "lot_size", "overall_quality", "garage"},
}

SENIORITY_MAP = {
    "intern": 0.1, "internship": 0.1,
    "entry": 0.3, "junior": 0.3,
    "associate": 0.4, "mid": 0.5,
    "senior": 0.7, "lead": 0.8,
    "principal": 0.85, "staff": 0.85,
    "director": 0.9, "vp": 0.95,
    "executive": 1.0, "cxo": 1.0, "c-level": 1.0,
}

COMPANY_SIZE_MAP = {
    "1-10": 0.05, "1-50": 0.1, "11-50": 0.1,
    "51-200": 0.3, "201-500": 0.5,
    "501-1000": 0.6, "1001-5000": 0.75,
    "5001-10000": 0.85, "10001+": 0.95, "5001+": 0.9,
}


def detect_domain(features: Dict[str, Any]) -> str:
    keys = {k.lower() for k in features.keys()}
    scores = {domain: len(keys & signals) for domain, signals in DOMAIN_SIGNALS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "generic"


# ─── Per-domain normalizers ────────────────────────────────────────────────────

def normalize_products(f: Dict) -> np.ndarray:
    """
    Products: price, quality_score (0-10), feature_count, brand_score (0-10),
              delivery_time (days), availability (0/1)
    """
    price        = float(f.get("price", 100))
    quality      = float(f.get("quality_score", 5)) / 10.0
    feature_cnt  = float(f.get("feature_count", 5))
    brand        = float(f.get("brand_score", 5)) / 10.0
    delivery     = float(f.get("delivery_time", 5))
    availability = float(f.get("availability", 1))
    rating       = float(f.get("rating", quality * 10)) / 10.0

    # value = quality per dollar (log scale to handle $1 vs $10,000)
    value_score   = quality / max(np.log10(price + 1), 0.01)
    quality_score = (quality + rating) / 2.0
    growth_score  = min(feature_cnt / 20.0, 1.0)               # feature richness
    risk_score    = 1.0 - availability * 0.5 - min(feature_cnt / 30.0, 0.5)
    fit_score     = brand
    speed_score   = 1.0 - min(delivery / 30.0, 1.0)            # faster delivery = higher score

    return _build(value_score, quality_score, growth_score, risk_score, fit_score, speed_score)


def normalize_jobs(f: Dict) -> np.ndarray:
    salary   = float(f.get("salary", f.get("max_salary", f.get("min_salary", 80000))))
    rating   = float(f.get("company_rating", 3.8)) / 5.0
    level    = f.get("seniority_level", "mid")
    size     = str(f.get("company_size", "201-500"))
    remote   = str(f.get("remote", f.get("work_type", "hybrid"))).lower()
    benefits = float(f.get("benefits_score", 5)) / 10.0

    level_score = SENIORITY_MAP.get(str(level).lower().strip(), 0.5)
    for k, v in SENIORITY_MAP.items():
        if k in str(level).lower():
            level_score = v
            break

    size_score = 0.5
    for k, v in COMPANY_SIZE_MAP.items():
        if k in size:
            size_score = v
            break

    remote_score = {"remote": 1.0, "hybrid": 0.65, "on-site": 0.35, "onsite": 0.35}.get(
        remote, 0.5
    )

    value_score   = min(salary / 200000.0, 1.0)
    quality_score = rating
    growth_score  = level_score
    risk_score    = 1.0 - size_score      # smaller company = more risk
    fit_score     = (rating + benefits) / 2.0
    speed_score   = remote_score

    return _build(value_score, quality_score, growth_score, risk_score, fit_score, speed_score)


def normalize_education(f: Dict) -> np.ndarray:
    tuition      = float(f.get("tuition", f.get("annual_cost", 30000)))
    rank         = float(f.get("ranking", f.get("world_rank", 500)))
    research     = float(f.get("research_score", 50)) / 100.0
    teaching     = float(f.get("teaching_score", 50)) / 100.0
    acceptance   = float(f.get("acceptance_rate", 50)) / 100.0
    citations    = float(f.get("citations", 50)) / 100.0
    total_score  = float(f.get("total_score", (research + teaching) * 50)) / 100.0

    # value = ranking / tuition (better rank + lower cost)
    value_score   = total_score / max(np.log10(tuition + 1) / 5.0, 0.01)
    quality_score = total_score
    growth_score  = research
    risk_score    = acceptance   # high acceptance = less exclusive = lower prestige signal
    fit_score     = citations
    speed_score   = teaching     # high teaching score = faster learning outcome

    return _build(value_score, quality_score, growth_score, risk_score, fit_score, speed_score)


def normalize_housing(f: Dict) -> np.ndarray:
    price   = float(f.get("price", f.get("saleprice", 250000)))
    quality = float(f.get("overall_quality", f.get("quality", 6))) / 10.0
    area    = float(f.get("area", f.get("sqft", f.get("gr_liv_area", 1500))))
    year    = float(f.get("year_built", 1990))
    garage  = float(f.get("garage", f.get("garage_cars", 1))) / 4.0
    beds    = float(f.get("bedrooms", 3)) / 6.0
    nbr     = float(f.get("neighborhood_score", 5)) / 10.0

    value_score   = (quality * area) / max(price / 1000.0, 1.0)
    quality_score = quality
    growth_score  = min((year - 1900) / 125.0, 1.0)   # newer = more growth potential
    risk_score    = 1.0 - quality
    fit_score     = (beds + garage + nbr) / 3.0
    speed_score   = garage

    return _build(value_score, quality_score, growth_score, risk_score, fit_score, speed_score)


def normalize_generic(f: Dict) -> np.ndarray:
    """
    Fallback: treat any numeric values as direct scores,
    normalize each to [0,1] relative to the other options.
    Since we only have one option here, we normalize each value
    by assuming it's in a 0-10 scale if between 0-10, else by magnitude.
    """
    vals = []
    for v in list(f.values())[:6]:
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(0.0)

    # Pad to 6 values
    while len(vals) < 6:
        vals.append(0.5)
    vals = vals[:6]

    # Normalize each to [0,1]
    normed = []
    for v in vals:
        if 0 <= v <= 1:
            normed.append(v)
        elif 0 <= v <= 10:
            normed.append(v / 10.0)
        elif 0 <= v <= 100:
            normed.append(v / 100.0)
        else:
            normed.append(min(v / 1000.0, 1.0))

    return _build(*normed)


# ─── Core interface ────────────────────────────────────────────────────────────

NORMALIZERS = {
    "products":  normalize_products,
    "jobs":      normalize_jobs,
    "education": normalize_education,
    "housing":   normalize_housing,
    "generic":   normalize_generic,
}


def to_universal_features(features: Dict[str, Any], domain: str | None = None) -> np.ndarray:
    """
    Convert any option's feature dict → 6-element numpy array
    [value_score, quality_score, growth_score, risk_score, fit_score, speed_score]

    Args:
        features: Raw feature dict from the API request
        domain:   Optional override. If None, auto-detected from feature keys.

    Returns:
        np.ndarray of shape (6,) with all values in [0, 1]
    """
    if domain is None:
        domain = detect_domain(features)
    fn = NORMALIZERS.get(domain, normalize_generic)
    vec = fn(features)
    return np.clip(vec, 0.0, 1.0)


def to_feature_dict(features: Dict[str, Any], domain: str | None = None) -> Dict[str, float]:
    """Same as to_universal_features but returns a labelled dict."""
    vec = to_universal_features(features, domain)
    return {
        "value_score":   float(vec[0]),
        "quality_score": float(vec[1]),
        "growth_score":  float(vec[2]),
        "risk_score":    float(vec[3]),
        "fit_score":     float(vec[4]),
        "speed_score":   float(vec[5]),
    }


# ─── Internal ─────────────────────────────────────────────────────────────────

def _build(v, q, g, r, f, s) -> np.ndarray:
    return np.array([
        float(np.clip(v, 0, 1)),
        float(np.clip(q, 0, 1)),
        float(np.clip(g, 0, 1)),
        float(np.clip(r, 0, 1)),
        float(np.clip(f, 0, 1)),
        float(np.clip(s, 0, 1)),
    ], dtype=float)