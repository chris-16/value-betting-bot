"""Tests for the prediction wrapper module."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.db.models import League, MarketType, Match, MatchStatus, Prediction, Team
from src.db.session import Base
from src.models.claude_llm import ClaudeCLIError
from src.models.prediction import (
    MODEL_VERSION,
    MatchPrediction,
    _build_consensus_prompt,
    _normalise_probabilities,
    predict_and_store,
    predict_match,
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
# Prompt building tests
# ---------------------------------------------------------------------------


class TestBuildConsensusPrompt:
    def test_contains_team_names(self) -> None:
        prompt = _build_consensus_prompt("Liverpool", "Man City")
        assert "Liverpool" in prompt
        assert "Man City" in prompt

    def test_contains_league(self) -> None:
        prompt = _build_consensus_prompt("Bayern", "Dortmund", league="Bundesliga")
        assert "Bundesliga" in prompt

    def test_contains_all_personas(self) -> None:
        prompt = _build_consensus_prompt("TeamA", "TeamB")
        assert "Statistician" in prompt
        assert "Tactician" in prompt
        assert "Sentiment Analyst" in prompt
        assert "Context Analyst" in prompt
        assert "Risk Assessor" in prompt

    def test_requests_json(self) -> None:
        prompt = _build_consensus_prompt("TeamA", "TeamB")
        assert "JSON" in prompt
        assert "home_win_probability" in prompt

    def test_additional_context_included(self) -> None:
        prompt = _build_consensus_prompt(
            "TeamA", "TeamB", additional_context="Star striker injured"
        )
        assert "Star striker injured" in prompt


# ---------------------------------------------------------------------------
# predict_match tests (mock Claude CLI)
# ---------------------------------------------------------------------------

MOCK_CLAUDE_RESPONSE: dict = {
    "home_win_probability": 0.55,
    "draw_probability": 0.25,
    "away_win_probability": 0.20,
    "confidence": 0.75,
    "reasoning": "Home team has strong recent form and home advantage.",
    "agent_analyses": [
        {
            "agent": "Statistician",
            "home_prob": 0.60,
            "draw_prob": 0.20,
            "away_prob": 0.20,
            "key_factors": ["form", "xG"],
        }
    ],
}


class TestPredictMatch:
    @patch("src.models.prediction.query_claude_json")
    def test_returns_prediction(self, mock_claude: object) -> None:
        mock_claude.return_value = MOCK_CLAUDE_RESPONSE  # type: ignore[attr-defined]

        result = predict_match("Arsenal", "Chelsea", league="Premier League")

        assert isinstance(result, MatchPrediction)
        assert result.home_prob > Decimal("0")
        assert result.draw_prob > Decimal("0")
        assert result.away_prob > Decimal("0")
        assert result.model_version == MODEL_VERSION

    @patch("src.models.prediction.query_claude_json")
    def test_probabilities_normalised(self, mock_claude: object) -> None:
        mock_claude.return_value = MOCK_CLAUDE_RESPONSE  # type: ignore[attr-defined]

        result = predict_match("Arsenal", "Chelsea")
        total = result.home_prob + result.draw_prob + result.away_prob
        assert abs(total - Decimal("1")) < Decimal("0.01")

    @patch("src.models.prediction.query_claude_json")
    def test_handles_unnormalised_response(self, mock_claude: object) -> None:
        mock_claude.return_value = {  # type: ignore[attr-defined]
            "home_win_probability": 0.6,
            "draw_probability": 0.3,
            "away_win_probability": 0.3,
            "confidence": 0.5,
            "reasoning": "Bad maths",
        }

        result = predict_match("TeamA", "TeamB")
        total = result.home_prob + result.draw_prob + result.away_prob
        assert abs(total - Decimal("1")) < Decimal("0.01")

    @patch("src.models.prediction.query_claude_json")
    def test_claude_error_propagates(self, mock_claude: object) -> None:
        mock_claude.side_effect = ClaudeCLIError("CLI failed")  # type: ignore[attr-defined]

        with pytest.raises(ClaudeCLIError, match="CLI failed"):
            predict_match("TeamA", "TeamB")


# ---------------------------------------------------------------------------
# predict_and_store tests (mock Claude CLI + in-memory DB)
# ---------------------------------------------------------------------------


class TestPredictAndStore:
    @patch("src.models.prediction.query_claude_json")
    def test_stores_three_predictions(self, mock_claude: object) -> None:
        mock_claude.return_value = MOCK_CLAUDE_RESPONSE  # type: ignore[attr-defined]

        session = _make_session()
        match = _create_match(session)

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

    @patch("src.models.prediction.query_claude_json")
    def test_updates_existing_predictions(self, mock_claude: object) -> None:
        mock_claude.return_value = MOCK_CLAUDE_RESPONSE  # type: ignore[attr-defined]

        session = _make_session()
        match = _create_match(session)

        # First prediction
        predict_and_store(session, match.id)

        # Second prediction with different values
        mock_claude.return_value = {  # type: ignore[attr-defined]
            "home_win_probability": 0.40,
            "draw_probability": 0.30,
            "away_win_probability": 0.30,
            "confidence": 0.60,
            "reasoning": "Updated assessment",
        }
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
    @patch("src.models.prediction.query_claude_json")
    def test_predicts_scheduled_matches(self, mock_claude: object) -> None:
        mock_claude.return_value = MOCK_CLAUDE_RESPONSE  # type: ignore[attr-defined]

        session = _make_session()
        match = _create_match(session)

        results = predict_upcoming_matches(session, limit=5)

        assert len(results) == 1
        assert results[0][0].id == match.id
        assert isinstance(results[0][1], MatchPrediction)
        session.close()

    @patch("src.models.prediction.query_claude_json")
    def test_skips_already_predicted_matches(self, mock_claude: object) -> None:
        mock_claude.return_value = MOCK_CLAUDE_RESPONSE  # type: ignore[attr-defined]

        session = _make_session()
        _create_match(session)

        # First run — predicts the match
        predict_upcoming_matches(session, limit=5)

        # Second run — should skip the already-predicted match
        results = predict_upcoming_matches(session, limit=5)
        assert len(results) == 0
        session.close()

    @patch("src.models.prediction.query_claude_json")
    def test_handles_claude_errors_gracefully(self, mock_claude: object) -> None:
        mock_claude.side_effect = ClaudeCLIError("CLI failed")  # type: ignore[attr-defined]

        session = _make_session()
        _create_match(session)

        # Should not raise, just skip the failed match
        results = predict_upcoming_matches(session, limit=5)
        assert len(results) == 0
        session.close()


# ---------------------------------------------------------------------------
# Claude LLM adapter tests
# ---------------------------------------------------------------------------


class TestClaudeLLM:
    @patch("src.models.claude_llm.subprocess.run")
    def test_query_claude_success(self, mock_run: object) -> None:
        from src.models.claude_llm import query_claude

        mock_run.return_value = type(  # type: ignore[attr-defined]
            "Result", (), {"returncode": 0, "stdout": "Hello world", "stderr": ""}
        )()
        result = query_claude("test prompt")
        assert result == "Hello world"

    @patch("src.models.claude_llm.subprocess.run")
    def test_query_claude_error(self, mock_run: object) -> None:
        from src.models.claude_llm import query_claude

        mock_run.return_value = type(  # type: ignore[attr-defined]
            "Result", (), {"returncode": 1, "stdout": "", "stderr": "Error occurred"}
        )()
        with pytest.raises(ClaudeCLIError, match="exited with code 1"):
            query_claude("test prompt")

    @patch("src.models.claude_llm.subprocess.run")
    def test_query_claude_json_parses_response(self, mock_run: object) -> None:
        from src.models.claude_llm import query_claude_json

        response_json = json.dumps({"key": "value"})
        mock_run.return_value = type(  # type: ignore[attr-defined]
            "Result", (), {"returncode": 0, "stdout": response_json, "stderr": ""}
        )()
        result = query_claude_json("test prompt")
        assert result == {"key": "value"}

    @patch("src.models.claude_llm.subprocess.run")
    def test_query_claude_json_handles_markdown_fences(self, mock_run: object) -> None:
        from src.models.claude_llm import query_claude_json

        response_text = '```json\n{"key": "value"}\n```'
        mock_run.return_value = type(  # type: ignore[attr-defined]
            "Result", (), {"returncode": 0, "stdout": response_text, "stderr": ""}
        )()
        result = query_claude_json("test prompt")
        assert result == {"key": "value"}

    @patch("src.models.claude_llm.subprocess.run")
    def test_query_claude_timeout(self, mock_run: object) -> None:
        import subprocess as sp

        from src.models.claude_llm import query_claude

        mock_run.side_effect = sp.TimeoutExpired(cmd="claude", timeout=120)  # type: ignore[attr-defined]
        with pytest.raises(ClaudeCLIError, match="timed out"):
            query_claude("test prompt")

    @patch("src.models.claude_llm.subprocess.run")
    def test_query_claude_not_found(self, mock_run: object) -> None:
        from src.models.claude_llm import query_claude

        mock_run.side_effect = FileNotFoundError()  # type: ignore[attr-defined]
        with pytest.raises(ClaudeCLIError, match="not found"):
            query_claude("test prompt")
