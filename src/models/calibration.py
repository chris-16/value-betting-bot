"""Probability calibration using isotonic regression.

Transforms raw model probabilities into well-calibrated probabilities
by fitting on historical (predicted, actual_outcome) pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.config import settings

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH = "data/models/calibrator_isotonic.joblib"


class ProbabilityCalibrator:
    """Isotonic regression calibrator for match outcome probabilities.

    Fits on historical predictions vs actual outcomes, then transforms
    raw probabilities into calibrated ones.
    """

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._is_fitted

    def fit(
        self,
        predicted_probs: list[float],
        actual_outcomes: list[int],
    ) -> None:
        """Fit the calibrator on historical data.

        Args:
            predicted_probs: List of predicted probabilities (0 to 1).
            actual_outcomes: List of binary outcomes (1 = event occurred, 0 = not).
        """
        if len(predicted_probs) < 10:
            logger.warning(
                "Too few samples (%d) for calibration — need at least 10",
                len(predicted_probs),
            )
            return

        X = np.array(predicted_probs, dtype=np.float64)
        y = np.array(actual_outcomes, dtype=np.float64)

        self._model = IsotonicRegression(
            y_min=0.01,
            y_max=0.99,
            out_of_bounds="clip",
        )
        self._model.fit(X, y)
        self._is_fitted = True

        logger.info("Calibrator fitted on %d samples", len(predicted_probs))

    def calibrate(self, prob: float) -> float:
        """Calibrate a single probability.

        Args:
            prob: Raw predicted probability.

        Returns:
            Calibrated probability. Returns input unchanged if not fitted.
        """
        if not self._is_fitted or self._model is None:
            return prob

        result = self._model.predict(np.array([prob]))[0]
        return float(np.clip(result, 0.01, 0.99))

    def calibrate_triple(
        self, home: float, draw: float, away: float
    ) -> tuple[float, float, float]:
        """Calibrate home/draw/away probabilities and renormalise.

        Args:
            home: Raw P(home_win).
            draw: Raw P(draw).
            away: Raw P(away_win).

        Returns:
            Calibrated and renormalised (home, draw, away) tuple.
        """
        cal_home = self.calibrate(home)
        cal_draw = self.calibrate(draw)
        cal_away = self.calibrate(away)

        # Renormalise to sum to 1.0
        total = cal_home + cal_draw + cal_away
        if total <= 0:
            return (1 / 3, 1 / 3, 1 / 3)

        return (
            cal_home / total,
            cal_draw / total,
            cal_away / total,
        )

    def save(self, path: str | None = None) -> None:
        """Save the fitted calibrator to disk.

        Args:
            path: File path for the artifact. Defaults to data/models/calibrator_isotonic.joblib.
        """
        if not self._is_fitted or self._model is None:
            logger.warning("Cannot save unfitted calibrator")
            return

        if path is None:
            path = str(Path(settings.model_data_dir) / "calibrator_isotonic.joblib")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("Calibrator saved to %s", path)

    def load(self, path: str | None = None) -> bool:
        """Load a fitted calibrator from disk.

        Args:
            path: File path for the artifact.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if path is None:
            path = str(Path(settings.model_data_dir) / "calibrator_isotonic.joblib")

        if not Path(path).exists():
            logger.debug("No calibrator artifact found at %s", path)
            return False

        try:
            self._model = joblib.load(path)
            self._is_fitted = True
            logger.info("Calibrator loaded from %s", path)
            return True
        except Exception:
            logger.exception("Failed to load calibrator from %s", path)
            return False
