"""Tests for the prediction wrapper module (Poisson + ELO ensemble)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.db.models import League, MarketType, Match, MatchStatus, Prediction, Team, TeamXGStats
from src.db.session import Base
from src.models.prediction import (
    MODEL_VERSION,
    MatchPrediction,
    _normalise_probabilities,
    predict_and_store,
    predict_upcoming_matches,
)


def _make_session() -> Session:
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)


def _create_match(session: Session) -> Match:
    """Helper to create a match with all required related entities."""
    league = League(name="Premier League", country="England")
    home = Team(name="Arsenal")
    away = Team(name="Chelsea")
    session.add_all([league, home, away])
    session.flush()

    match = Match(
        league_id=league.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=datetime(2026, 4, 20, 15, 0, tzinfo=UTC),
        status=MatchStatus.SCHEDULED,
    )
    session.add(match)
    session.flush()
    return match


def _create_finished_matches(session: Session, count: int = 10) -> list[Match]:
    """Create finished matches for Poisson model fitting."""
    league = session.query(League).first()
    if league is None:
        league = League(name="Premier League", country="England")
        session.add(league)
        session.flush()

    teams = session.query(Team).all()
    if len(teams) < 4:
        for name in ["TeamA", "TeamB", "TeamC", "TeamD"]:
            t = Team(name=name)
            session.add(t)
        session.flush()
        teams = session.query(Team).all()

    matches = []
    for i in range(count):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        m = Match(
            league_id=league.id,
            home_team_id=home.id,
            away_team_id=away.id,
            kickoff=datetime(2026, 1, 1 + i, 15, 0, tzinfo=UTC),
            status=MatchStatus.FINISHED,
            home_goals=i % 3,
            away_goals=(i + 1) % 3,
        )
        session.add(m)
        matches.append(m)

    session.flush()
    return matches


# ---------------------------------------------------------------------------
# MatchPrediction dataclass tests
# ---------------------------------------------------------------------------


class TestMatchPrediction:
    def test_valid_prediction(self) -> None:
        pred = MatchPrediction(
            home_prob=Decimal("0.500000"),
            draw_prob=Decimal("0.250000"),
            away_prob=Decimal("0.250000"),
            confidence=Decimal("0.8"),
            model_version="test-v1",
            reasoning="Test reasoning",
        )
        assert pred.home_prob == Decimal("0.500000")
        assert pred.draw_prob == Decimal("0.250000")
        assert pred.away_prob == Decimal("0.250000")

    def test_probabilities_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to ~1.0"):
            MatchPrediction(
                home_prob=Decimal("0.5"),
                draw_prob=Decimal("0.5"),
                away_prob=Decimal("0.5"),
                confidence=Decimal("0.8"),
                model_version="test-v1",
                reasoning="Bad probabilities",
            )

    def test_small_rounding_tolerance(self) -> None:
        """Allow probabilities summing to e.g. 0.999 or 1.001."""
        pred = MatchPrediction(
            home_prob=Decimal("0.334"),
            draw_prob=Decimal("0.333"),
            away_prob=Decimal("0.334"),
            confidence=Decimal("0.7"),
            model_version="test-v1",
            reasoning="Close enough",
        )
        assert pred.home_prob == Decimal("0.334")

    def test_frozen(self) -> None:
        pred = MatchPrediction(
            home_prob=Decimal("0.5"),
            draw_prob=Decimal("0.25"),
            away_prob=Decimal("0.25"),
            confidence=Decimal("0.8"),
            model_version="test-v1",
            reasoning="Immutable",
        )
        with pytest.raises(AttributeError):
            pred.home_prob = Decimal("0.6")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Normalisation tests
# ---------------------------------------------------------------------------


class TestNormaliseProbabilities:
    def test_already_normalised(self) -> None:
        h, d, a = _normalise_probabilities(Decimal("0.5"), Decimal("0.3"), Decimal("0.2"))
        assert h + d + a == Decimal("1.000000")

    def test_unnormalised(self) -> None:
        h, d, a = _normalise_probabilities(Decimal("2"), Decimal("1"), Decimal("1"))
        assert h == Decimal("0.500000")
        assert d == Decimal("0.250000")
        assert a == Decimal("0.250000")

    def test_zero_total_returns_uniform(self) -> None:
        h, d, a = _normalise_probabilities(Decimal("0"), Decimal("0"), Decimal("0"))
        assert h == Decimal("0.333333")
        assert d == Decimal("0.333334")
        assert a == Decimal("0.333333")


# ---------------------------------------------------------------------------
# Model version tests
# ---------------------------------------------------------------------------


class TestModelVersion:
    def test_model_version_is_poisson(self) -> None:
        assert MODEL_VERSION == "poisson-v1"


# ---------------------------------------------------------------------------
# predict_and_store tests (Poisson model + in-memory DB)
# ---------------------------------------------------------------------------


class TestPredictAndStore:
    def test_stores_three_predictions(self) -> None:
        session = _make_session()
        match = _create_match(session)
        # Create some finished matches so Poisson can fit
        _create_finished_matches(session, count=10)

        result = predict_and_store(session, match.id)

        # Should create 3 prediction rows (home, draw, away)
        preds = session.query(Prediction).filter(Prediction.match_id == match.id).all()
        assert len(preds) == 3
        selections = {p.selection for p in preds}
        assert selections == {"home", "draw", "away"}

        # All should have correct model version
        for p in preds:
            assert p.model_version == MODEL_VERSION
            assert p.market == MarketType.MATCH_WINNER

        assert isinstance(result, MatchPrediction)
        session.close()

    def test_probabilities_sum_to_one(self) -> None:
        session = _make_session()
        match = _create_match(session)
        _create_finished_matches(session, count=10)

        result = predict_and_store(session, match.id)
        total = result.home_prob + result.draw_prob + result.away_prob
        assert abs(total - Decimal("1")) < Decimal("0.01")
        session.close()

    def test_updates_existing_predictions(self) -> None:
        session = _make_session()
        match = _create_match(session)
        _create_finished_matches(session, count=10)

        # First prediction
        predict_and_store(session, match.id)

        # Second prediction (should update, not duplicate)
        predict_and_store(session, match.id)

        # Should still have only 3 rows (updated, not duplicated)
        preds = session.query(Prediction).filter(Prediction.match_id == match.id).all()
        assert len(preds) == 3
        session.close()

    def test_raises_for_invalid_match_id(self) -> None:
        session = _make_session()
        with pytest.raises(ValueError, match="not found"):
            predict_and_store(session, 99999)
        session.close()


# ---------------------------------------------------------------------------
# predict_upcoming_matches tests
# ---------------------------------------------------------------------------


class TestPredictUpcomingMatches:
    def test_predicts_scheduled_matches(self) -> None:
        session = _make_session()
        match = _create_match(session)
        _create_finished_matches(session, count=10)

        results = predict_upcoming_matches(session, limit=5)

        assert len(results) == 1
        assert results[0][0].id == match.id
        assert isinstance(results[0][1], MatchPrediction)
        session.close()

    def test_skips_already_predicted_matches(self) -> None:
        session = _make_session()
        _create_match(session)
        _create_finished_matches(session, count=10)

        # First run — predicts the match
        predict_upcoming_matches(session, limit=5)

        # Second run — should skip the already-predicted match
        results = predict_upcoming_matches(session, limit=5)
        assert len(results) == 0
        session.close()


# ---------------------------------------------------------------------------
# Poisson model tests
# ---------------------------------------------------------------------------


class TestPoissonPredictor:
    def test_predict_returns_valid_probabilities(self) -> None:
        from src.models.poisson import PoissonPredictor

        session = _make_session()
        _create_match(session)
        _create_finished_matches(session, count=20)

        predictor = PoissonPredictor()
        predictor.fit(session)

        teams = session.query(Team).all()
        if len(teams) >= 2:
            result = predictor.predict(teams[0].id, teams[1].id)
            total = result.home_win + result.draw + result.away_win
            assert abs(total - 1.0) < 0.01
            assert result.home_win >= 0
            assert result.draw >= 0
            assert result.away_win >= 0
            assert result.home_lambda > 0
            assert result.away_lambda > 0

        session.close()

    def test_score_matrix_shape(self) -> None:
        from src.models.poisson import MAX_GOALS, PoissonPredictor

        session = _make_session()
        _create_match(session)
        _create_finished_matches(session, count=20)

        predictor = PoissonPredictor()
        predictor.fit(session)

        teams = session.query(Team).all()
        if len(teams) >= 2:
            result = predictor.predict(teams[0].id, teams[1].id)
            assert len(result.score_matrix) == MAX_GOALS + 1
            assert len(result.score_matrix[0]) == MAX_GOALS + 1

        session.close()

    def test_unknown_team_defaults_to_average(self) -> None:
        from src.models.poisson import PoissonPredictor

        session = _make_session()
        _create_finished_matches(session, count=10)

        predictor = PoissonPredictor()
        predictor.fit(session)

        # Predict for team IDs that don't exist
        result = predictor.predict(99999, 99998)
        total = result.home_win + result.draw + result.away_win
        assert abs(total - 1.0) < 0.01

        session.close()


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestProbabilityCalibrator:
    def test_calibrate_unfitted_returns_input(self) -> None:
        from src.models.calibration import ProbabilityCalibrator

        cal = ProbabilityCalibrator()
        assert cal.calibrate(0.5) == 0.5

    def test_calibrate_triple_normalises(self) -> None:
        from src.models.calibration import ProbabilityCalibrator

        cal = ProbabilityCalibrator()
        h, d, a = cal.calibrate_triple(0.5, 0.3, 0.2)
        assert abs(h + d + a - 1.0) < 0.01

    def test_fit_and_calibrate(self) -> None:
        from src.models.calibration import ProbabilityCalibrator

        cal = ProbabilityCalibrator()
        # Simple calibration data
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.5, 0.9]
        outcomes = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]

        cal.fit(probs, outcomes)
        assert cal.is_fitted

        # Calibrated output should be between 0 and 1
        result = cal.calibrate(0.5)
        assert 0.0 < result < 1.0
