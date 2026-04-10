"""ML model training, inference, and prediction wrappers."""

from src.models.prediction import (
    MatchPrediction,
    predict_and_store,
    predict_match,
    predict_upcoming_matches,
)

__all__ = [
    "MatchPrediction",
    "predict_and_store",
    "predict_match",
    "predict_upcoming_matches",
]
