"""VisionX ML module - real inference engine."""
from .predict import predict_winner, score_options_for_user
from .normalizer import to_universal_features, to_feature_dict, detect_domain
__all__ = [
    "predict_winner",
    "score_options_for_user",
    "to_universal_features",
    "to_feature_dict",
    "detect_domain",
]
 
