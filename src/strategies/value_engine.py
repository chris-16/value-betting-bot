"""Value Engine — detects value bets by comparing model predictions against bookmaker odds.

Calculates edge (predicted probability minus implied probability), filters by a
configurable minimum edge threshold, and sizes stakes using the Kelly Criterion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import settings
from src.db.models import (
    Bet,
    BetOutcome,
    MarketType,
    Match,
    MatchStatus,
    Odds,
    Prediction,
)

logger = logging.getLogger(__name__)

# Precision constants
_PROB_QUANT = Decimal("0.000001")  # 6 decimal places
_STAKE_QUANT = Decimal("0.01")  # 2 decimal places (currency)


class ValueEngineError(Exception):
    """Raised when value engine encounters a fatal error."""


@dataclass(frozen=True)
class ValueBet:
    """A detected value betting opportunity."""

    match_id: int
    market: MarketType
    selection: str
    predicted_probability: Decimal
    implied_probability: Decimal
    odds_price: Decimal
    edge: Decimal
    kelly_fraction: Decimal
    recommended_stake: Decimal
    bookmaker_id: int

    def __repr__(self) -> str:
        return (
            f"<ValueBet {self.selection} edge={self.edge} "
            f"stake={self.recommended_stake} odds={self.odds_price}>"
        )


def calculate_edge(predicted_probability: Decimal, implied_probability: Decimal) -> Decimal:
    """Calculate value edge as predicted probability minus implied probability.

    Args:
        predicted_probability: Model's estimated probability of the outcome.
        implied_probability: Probability implied by bookmaker odds (1 / decimal_odds).

    Returns:
        Edge in probability points (e.g. 0.05 means 5 percentage points of edge).
    """
    return (predicted_probability - implied_probability).quantize(_PROB_QUANT, ROUND_HALF_UP)


def kelly_criterion(
    predicted_probability: Decimal,
    decimal_odds: Decimal,
    fraction: Decimal = Decimal("0.50"),
) -> Decimal:
    """Compute fractional Kelly Criterion stake as a fraction of bankroll.

    Full Kelly: f* = (b * p - q) / b
    where b = decimal_odds - 1, p = predicted probability, q = 1 - p.

    A fractional Kelly (default 25%) is used to reduce variance.

    Args:
        predicted_probability: Model's estimated probability of the outcome.
        decimal_odds: Bookmaker decimal odds (e.g. 2.50).
        fraction: Kelly fraction to apply (default 0.25 = quarter Kelly).

    Returns:
        Recommended stake as a fraction of bankroll (0 to 1). Returns 0 if
        the Kelly formula yields a negative value (no edge).
    """
    b = decimal_odds - Decimal("1")
    if b <= 0:
        return Decimal("0")

    p = predicted_probability
    q = Decimal("1") - p

    kelly_full = (b * p - q) / b
    if kelly_full <= 0:
        return Decimal("0")

    return (kelly_full * fraction).quantize(_PROB_QUANT, ROUND_HALF_UP)


def find_value_bets(
    session: Session,
    match_id: int,
    min_edge: Decimal | None = None,
    bankroll: Decimal | None = None,
    kelly_fraction: Decimal = Decimal("0.25"),
) -> list[ValueBet]:
    """Find value bets for a given match by comparing predictions to odds.

    Fetches the latest odds and model predictions for the match, calculates
    edge for each selection, and returns only those exceeding the minimum
    edge threshold with Kelly-sized stakes.

    Args:
        session: SQLAlchemy session.
        match_id: ID of the match to analyse.
        min_edge: Minimum edge threshold (default: settings.min_value_edge).
        bankroll: Current bankroll for stake sizing (default: settings.paper_trading_bankroll).
        kelly_fraction: Kelly fraction multiplier (default 0.25 = quarter Kelly).

    Returns:
        List of ValueBet opportunities sorted by edge descending.
    """
    if min_edge is None:
        min_edge = settings.min_value_edge
    if bankroll is None:
        bankroll = settings.paper_trading_bankroll

    # Fetch predictions for this match
    predictions = session.execute(
        select(Prediction).where(
            Prediction.match_id == match_id,
            Prediction.market == MarketType.MATCH_WINNER,
        )
    ).scalars().all()

    if not predictions:
        logger.debug("No predictions found for match %d", match_id)
        return []

    # Build lookup: selection -> predicted probability
    pred_map: dict[str, Decimal] = {p.selection: p.probability for p in predictions}

    # Fetch latest odds for this match (most recent per bookmaker + selection)
    odds_rows = session.execute(
        select(Odds).where(
            Odds.match_id == match_id,
            Odds.market == MarketType.MATCH_WINNER,
        ).order_by(Odds.retrieved_at.desc())
    ).scalars().all()

    if not odds_rows:
        logger.debug("No odds found for match %d", match_id)
        return []

    # Keep only the latest odds per (bookmaker, selection)
    latest_odds: dict[tuple[int, str], Odds] = {}
    for o in odds_rows:
        key = (o.bookmaker_id, o.selection)
        if key not in latest_odds:
            latest_odds[key] = o

    value_bets: list[ValueBet] = []

    for (bookmaker_id, selection), odds in latest_odds.items():
        predicted_prob = pred_map.get(selection)
        if predicted_prob is None:
            continue

        edge = calculate_edge(predicted_prob, odds.implied_probability)

        if edge < min_edge:
            logger.debug(
                "Match %d %s: edge %.4f < min %.4f — skipping",
                match_id,
                selection,
                edge,
                min_edge,
            )
            continue

        kelly_frac = kelly_criterion(predicted_prob, odds.price, kelly_fraction)
        stake = (bankroll * kelly_frac).quantize(_STAKE_QUANT, ROUND_HALF_UP)

        # Clamp stake: at least 0, at most bankroll
        stake = max(Decimal("0"), min(stake, bankroll))

        vb = ValueBet(
            match_id=match_id,
            market=MarketType.MATCH_WINNER,
            selection=selection,
            predicted_probability=predicted_prob,
            implied_probability=odds.implied_probability,
            odds_price=odds.price,
            edge=edge,
            kelly_fraction=kelly_frac,
            recommended_stake=stake,
            bookmaker_id=bookmaker_id,
        )
        value_bets.append(vb)
        logger.info(
            "Value bet found: match=%d sel=%s edge=%.4f stake=%s odds=%s",
            match_id,
            selection,
            edge,
            stake,
            odds.price,
        )

    value_bets.sort(key=lambda vb: vb.edge, reverse=True)
    return value_bets


def scan_for_value(
    session: Session,
    min_edge: Decimal | None = None,
    bankroll: Decimal | None = None,
    kelly_fraction: Decimal = Decimal("0.25"),
) -> list[ValueBet]:
    """Scan all scheduled matches with predictions for value bets.

    Args:
        session: SQLAlchemy session.
        min_edge: Minimum edge threshold override.
        bankroll: Current bankroll override.
        kelly_fraction: Kelly fraction multiplier.

    Returns:
        All value bets found across scheduled matches, sorted by edge descending.
    """
    # Find scheduled matches that have at least one prediction
    match_ids = session.execute(
        select(Prediction.match_id)
        .join(Match)
        .where(Match.status == MatchStatus.SCHEDULED)
        .distinct()
    ).scalars().all()

    logger.info("Scanning %d scheduled matches for value", len(match_ids))

    all_value_bets: list[ValueBet] = []
    for mid in match_ids:
        vbs = find_value_bets(session, mid, min_edge, bankroll, kelly_fraction)
        all_value_bets.extend(vbs)

    all_value_bets.sort(key=lambda vb: vb.edge, reverse=True)
    logger.info("Found %d value bets across %d matches", len(all_value_bets), len(match_ids))
    return all_value_bets


def place_paper_bet(session: Session, value_bet: ValueBet) -> Bet:
    """Place a paper trading bet from a detected value opportunity.

    Args:
        session: SQLAlchemy session.
        value_bet: The value bet opportunity to act on.

    Returns:
        The persisted Bet record.
    """
    bet = Bet(
        match_id=value_bet.match_id,
        market=value_bet.market,
        selection=value_bet.selection,
        odds_price=value_bet.odds_price,
        stake=value_bet.recommended_stake,
        model_probability=value_bet.predicted_probability,
        implied_probability=value_bet.implied_probability,
        value_edge=value_bet.edge,
        outcome=BetOutcome.PENDING,
    )
    session.add(bet)
    session.commit()
    logger.info(
        "Paper bet placed: match=%d sel=%s stake=%s edge=%.4f",
        bet.match_id,
        bet.selection,
        bet.stake,
        bet.value_edge,
    )
    return bet
