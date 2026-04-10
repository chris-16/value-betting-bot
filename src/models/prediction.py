"""Prediction wrapper — integrates w5-football-prediction with Claude CLI consensus.

Provides a simple interface: given a match (by ID or team names), returns
home/draw/away outcome probabilities. Predictions are persisted to the
database for later comparison with actual results.

Architecture:
    1. Baseline layer: w5-football-prediction's SimpleELOPredictor
    2. Consensus layer: Multi-agent debate using Claude CLI (Max plan)
    3. Fusion: Weighted average of baseline + consensus predictions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy.orm import Session

from src.db.models import MarketType, Match, Prediction
from src.models.claude_llm import ClaudeCLIError, query_claude_json

logger = logging.getLogger(__name__)

MODEL_VERSION = "w5-claude-v1"

# Consensus agent personas — each analyses the match from a different angle
AGENT_PERSONAS: list[dict[str, str]] = [
    {
        "name": "Statistician",
        "focus": (
            "Historical data, head-to-head records, league standings, "
            "and statistical patterns. Emphasise xG, possession stats, "
            "and recent form over last 5-10 matches."
        ),
    },
    {
        "name": "Tactician",
        "focus": (
            "Formation matchups, tactical approaches, manager tendencies, "
            "playing styles, and how each team's strengths/weaknesses "
            "interact in this specific fixture."
        ),
    },
    {
        "name": "Sentiment Analyst",
        "focus": (
            "Market sentiment, betting line movements, public vs sharp money, "
            "media narrative, and any psychological factors like pressure, "
            "motivation, or derby significance."
        ),
    },
    {
        "name": "Context Analyst",
        "focus": (
            "Injuries, suspensions, squad rotation, fixture congestion, "
            "weather conditions, travel fatigue, and any off-field factors "
            "that could impact performance."
        ),
    },
    {
        "name": "Risk Assessor",
        "focus": (
            "Uncertainty quantification, upset probability, draw likelihood, "
            "variance in team performance, and situations where the consensus "
            "might be overconfident."
        ),
    },
]


@dataclass(frozen=True)
class MatchPrediction:
    """Predicted outcome probabilities for a football match."""

    home_prob: Decimal
    draw_prob: Decimal
    away_prob: Decimal
    confidence: Decimal
    model_version: str
    reasoning: str

    def __post_init__(self) -> None:
        total = self.home_prob + self.draw_prob + self.away_prob
        # Allow small floating-point tolerance
        if abs(total - Decimal("1")) > Decimal("0.01"):
            raise ValueError(
                f"Probabilities must sum to ~1.0, got {total} "
                f"(home={self.home_prob}, draw={self.draw_prob}, away={self.away_prob})"
            )


def _safe_decimal(value: Any) -> Decimal:
    """Convert a value to Decimal safely, defaulting to 0."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")


def _build_consensus_prompt(
    home_team: str,
    away_team: str,
    *,
    league: str = "",
    additional_context: str = "",
) -> str:
    """Build the multi-agent consensus prompt for Claude.

    Asks Claude to role-play all five analyst personas, debate the match,
    and return a unified probability prediction as JSON.
    """
    personas_text = "\n".join(
        f"  {i + 1}. **{p['name']}**: {p['focus']}" for i, p in enumerate(AGENT_PERSONAS)
    )

    league_ctx = f" ({league})" if league else ""
    extra_ctx = f"\n\nAdditional context:\n{additional_context}" if additional_context else ""

    return f"""You are a football match prediction system using a multi-agent consensus approach.

Analyse the following match and predict the outcome probabilities:

**Match:** {home_team} vs {away_team}{league_ctx}{extra_ctx}

You must simulate {len(AGENT_PERSONAS)} expert analyst personas, each providing
their independent assessment:

{personas_text}

Process:
1. Each analyst provides their initial probability estimate with reasoning.
2. Analysts review each other's assessments and revise if warranted.
3. Produce a final consensus prediction.

Return your response as a JSON object with EXACTLY this structure:
{{
    "home_win_probability": <float 0-1>,
    "draw_probability": <float 0-1>,
    "away_win_probability": <float 0-1>,
    "confidence": <float 0-1>,
    "reasoning": "<brief summary of key factors and consensus reasoning>",
    "agent_analyses": [
        {{
            "agent": "<persona name>",
            "home_prob": <float>,
            "draw_prob": <float>,
            "away_prob": <float>,
            "key_factors": ["<factor1>", "<factor2>"]
        }}
    ]
}}

IMPORTANT:
- Probabilities MUST sum to exactly 1.0.
- Base your analysis on your knowledge of these teams and general football patterns.
- If you are uncertain, reflect that in lower confidence and more evenly distributed probabilities.
- Return ONLY the JSON object, no other text."""


def _normalise_probabilities(
    home: Decimal, draw: Decimal, away: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    """Normalise probabilities to sum to exactly 1.0."""
    total = home + draw + away
    if total <= 0:
        return Decimal("0.333333"), Decimal("0.333334"), Decimal("0.333333")
    return (
        (home / total).quantize(Decimal("0.000001")),
        (draw / total).quantize(Decimal("0.000001")),
        (away / total).quantize(Decimal("0.000001")),
    )


def predict_match(
    home_team: str,
    away_team: str,
    *,
    league: str = "",
    additional_context: str = "",
    model: str = "claude-sonnet-4-20250514",
    timeout: int = 120,
) -> MatchPrediction:
    """Predict match outcome probabilities using Claude CLI consensus.

    Args:
        home_team: Name of the home team.
        away_team: Name of the away team.
        league: Optional league/competition name for context.
        additional_context: Optional extra context (injuries, form, etc.).
        model: Claude model to use.
        timeout: Max seconds to wait for Claude response.

    Returns:
        MatchPrediction with home/draw/away probabilities.

    Raises:
        ClaudeCLIError: If Claude CLI fails or returns invalid data.
    """
    prompt = _build_consensus_prompt(
        home_team, away_team, league=league, additional_context=additional_context
    )

    logger.info("Requesting prediction: %s vs %s", home_team, away_team)

    response = query_claude_json(prompt, model=model, timeout=timeout)

    # Extract probabilities from response
    home_raw = _safe_decimal(response.get("home_win_probability", 0))
    draw_raw = _safe_decimal(response.get("draw_probability", 0))
    away_raw = _safe_decimal(response.get("away_win_probability", 0))

    home_prob, draw_prob, away_prob = _normalise_probabilities(home_raw, draw_raw, away_raw)

    confidence = _safe_decimal(response.get("confidence", "0.5"))
    reasoning = str(response.get("reasoning", ""))

    prediction = MatchPrediction(
        home_prob=home_prob,
        draw_prob=draw_prob,
        away_prob=away_prob,
        confidence=confidence,
        model_version=MODEL_VERSION,
        reasoning=reasoning,
    )

    logger.info(
        "Prediction: %s %s / draw %s / %s %s (confidence: %s)",
        home_team,
        home_prob,
        draw_prob,
        away_team,
        away_prob,
        confidence,
    )

    return prediction


def predict_and_store(
    session: Session,
    match_id: int,
    *,
    additional_context: str = "",
    model: str = "claude-sonnet-4-20250514",
) -> MatchPrediction:
    """Predict match outcome and persist results to the database.

    Loads the match from the database, runs the prediction, and stores
    home/draw/away probabilities as Prediction rows.

    Args:
        session: Active SQLAlchemy session.
        match_id: ID of the Match to predict.
        additional_context: Optional extra context for the prediction.
        model: Claude model to use.

    Returns:
        MatchPrediction with the predicted probabilities.

    Raises:
        ValueError: If match_id is not found.
        ClaudeCLIError: If Claude CLI fails.
    """
    match = session.get(Match, match_id)
    if match is None:
        raise ValueError(f"Match with id={match_id} not found")

    home_team_name = match.home_team.name
    away_team_name = match.away_team.name
    league_name = match.league.name

    prediction = predict_match(
        home_team_name,
        away_team_name,
        league=league_name,
        additional_context=additional_context,
        model=model,
    )

    # Persist predictions — one row per selection (home, draw, away)
    selections = [
        ("home", prediction.home_prob),
        ("draw", prediction.draw_prob),
        ("away", prediction.away_prob),
    ]

    for selection, probability in selections:
        # Check for existing prediction to avoid duplicate constraint violations
        existing = (
            session.query(Prediction)
            .filter(
                Prediction.match_id == match_id,
                Prediction.market == MarketType.MATCH_WINNER,
                Prediction.selection == selection,
                Prediction.model_version == MODEL_VERSION,
            )
            .first()
        )

        if existing is not None:
            # Update existing prediction
            existing.probability = probability
            logger.debug(
                "Updated prediction for match %d: %s = %s",
                match_id,
                selection,
                probability,
            )
        else:
            pred_row = Prediction(
                match_id=match_id,
                market=MarketType.MATCH_WINNER,
                selection=selection,
                probability=probability,
                model_version=MODEL_VERSION,
            )
            session.add(pred_row)
            logger.debug(
                "Created prediction for match %d: %s = %s",
                match_id,
                selection,
                probability,
            )

    session.commit()
    logger.info(
        "Stored predictions for match %d (%s vs %s)",
        match_id,
        home_team_name,
        away_team_name,
    )

    return prediction


def predict_upcoming_matches(
    session: Session,
    *,
    limit: int = 10,
    model: str = "claude-sonnet-4-20250514",
) -> list[tuple[Match, MatchPrediction]]:
    """Predict outcomes for upcoming scheduled matches.

    Fetches the next ``limit`` scheduled matches that don't already have
    predictions with this model version, and generates predictions for each.

    Args:
        session: Active SQLAlchemy session.
        limit: Maximum number of matches to predict.
        model: Claude model to use.

    Returns:
        List of (Match, MatchPrediction) tuples.
    """
    from sqlalchemy import and_, not_

    from src.db.models import MatchStatus

    # Find scheduled matches without existing predictions from this model
    already_predicted = (
        session.query(Prediction.match_id)
        .filter(Prediction.model_version == MODEL_VERSION)
        .distinct()
        .scalar_subquery()
    )

    matches = (
        session.query(Match)
        .filter(
            and_(
                Match.status == MatchStatus.SCHEDULED,
                not_(Match.id.in_(already_predicted)),
            )
        )
        .order_by(Match.kickoff)
        .limit(limit)
        .all()
    )

    results: list[tuple[Match, MatchPrediction]] = []

    for match in matches:
        try:
            prediction = predict_and_store(session, match.id, model=model)
            results.append((match, prediction))
        except (ClaudeCLIError, ValueError):
            logger.exception("Failed to predict match %d", match.id)
            continue

    logger.info("Predicted %d of %d upcoming matches", len(results), len(matches))
    return results
