"""Paper Trading Simulator — records hypothetical bets and tracks P&L, ROI, drawdown.

Settles bets against actual match results, maintains a virtual bankroll derived
from the configured initial bankroll plus cumulative P&L, and computes key
performance metrics: ROI, maximum drawdown, and win rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from src.db.models import Bet, BetOutcome, League, Match, MatchStatus, Prediction, Team

logger = logging.getLogger(__name__)

_STAKE_QUANT = Decimal("0.01")
_PCT_QUANT = Decimal("0.01")


@dataclass(frozen=True)
class PortfolioStats:
    """Snapshot of paper trading portfolio performance."""

    initial_bankroll: Decimal
    current_bankroll: Decimal
    total_bets: int
    settled_bets: int
    pending_bets: int
    wins: int
    losses: int
    voids: int
    win_rate: Decimal  # percentage (0–100)
    total_staked: Decimal
    total_pnl: Decimal
    roi: Decimal  # percentage
    max_drawdown: Decimal  # absolute value in currency
    max_drawdown_pct: Decimal  # percentage of peak bankroll
    best_day_pnl: Decimal
    worst_day_pnl: Decimal
    avg_odds: Decimal
    avg_edge: Decimal


# ---------------------------------------------------------------------------
# Bet settlement
# ---------------------------------------------------------------------------


def _determine_outcome(bet: Bet, match: Match) -> BetOutcome:
    """Determine bet outcome from the finished match result.

    Supports MATCH_WINNER market (home / draw / away selections).

    Returns:
        BetOutcome.WIN, LOSS, or VOID.
    """
    if match.home_goals is None or match.away_goals is None:
        return BetOutcome.VOID

    if match.home_goals > match.away_goals:
        actual = "home"
    elif match.home_goals < match.away_goals:
        actual = "away"
    else:
        actual = "draw"

    if bet.selection.lower() == actual:
        return BetOutcome.WIN
    return BetOutcome.LOSS


def _calculate_bet_pnl(bet: Bet, outcome: BetOutcome) -> Decimal:
    """Calculate P&L for a single bet given its outcome.

    WIN  → profit = stake * (odds - 1)
    LOSS → loss   = -stake
    VOID → 0 (stake returned)
    """
    if outcome == BetOutcome.WIN:
        return (bet.stake * (bet.odds_price - Decimal("1"))).quantize(
            _STAKE_QUANT, ROUND_HALF_UP
        )
    if outcome == BetOutcome.LOSS:
        return -bet.stake
    # VOID or PENDING
    return Decimal("0.00")


def settle_bet(session: Session, bet: Bet) -> Bet:
    """Settle a single pending bet against its match result.

    The bet's outcome, pnl, and settled_at are updated in place.  The caller
    is responsible for committing the session.

    Args:
        session: SQLAlchemy session.
        bet: The Bet to settle (must be PENDING with a FINISHED match).

    Returns:
        The updated Bet.

    Raises:
        ValueError: If the bet is not pending or match is not finished.
    """
    if bet.outcome != BetOutcome.PENDING:
        raise ValueError(f"Bet {bet.id} is already settled ({bet.outcome})")

    match = session.get(Match, bet.match_id)
    if match is None:
        raise ValueError(f"Match {bet.match_id} not found")
    if match.status != MatchStatus.FINISHED:
        raise ValueError(
            f"Match {bet.match_id} is not finished (status={match.status})"
        )

    outcome = _determine_outcome(bet, match)
    pnl = _calculate_bet_pnl(bet, outcome)

    bet.outcome = outcome
    bet.pnl = pnl
    bet.settled_at = datetime.now(tz=None)

    logger.info(
        "Settled bet %d: %s → %s pnl=%s",
        bet.id,
        bet.selection,
        outcome.value,
        pnl,
    )
    return bet


def settle_pending_bets(session: Session) -> list[Bet]:
    """Auto-settle all pending bets whose matches have finished.

    Args:
        session: SQLAlchemy session (committed on success).

    Returns:
        List of newly settled Bet objects.
    """
    pending = (
        session.execute(
            select(Bet)
            .join(Match)
            .where(
                Bet.outcome == BetOutcome.PENDING,
                Match.status == MatchStatus.FINISHED,
            )
        )
        .scalars()
        .all()
    )

    settled: list[Bet] = []
    for bet in pending:
        try:
            settle_bet(session, bet)
            settled.append(bet)
        except ValueError:
            logger.warning("Skipping bet %d: cannot settle", bet.id, exc_info=True)

    if settled:
        session.commit()
        logger.info("Settled %d pending bets", len(settled))

    return settled


# ---------------------------------------------------------------------------
# Bankroll tracking
# ---------------------------------------------------------------------------


def get_current_bankroll(
    session: Session,
    initial_bankroll: Decimal,
) -> Decimal:
    """Compute the current bankroll: initial + sum of settled P&L - pending stakes.

    Pending bets have their stakes subtracted from the bankroll (money is
    "at risk" until settlement).

    Args:
        session: SQLAlchemy session.
        initial_bankroll: Starting virtual bankroll.

    Returns:
        Current bankroll as Decimal.
    """
    settled_pnl = (
        session.execute(
            select(func.coalesce(func.sum(Bet.pnl), Decimal("0.00"))).where(
                Bet.outcome.in_([BetOutcome.WIN, BetOutcome.LOSS, BetOutcome.VOID])
            )
        )
        .scalar_one()
    )

    pending_stakes = (
        session.execute(
            select(func.coalesce(func.sum(Bet.stake), Decimal("0.00"))).where(
                Bet.outcome == BetOutcome.PENDING
            )
        )
        .scalar_one()
    )

    return (initial_bankroll + Decimal(str(settled_pnl)) - Decimal(str(pending_stakes))).quantize(
        _STAKE_QUANT, ROUND_HALF_UP
    )


# ---------------------------------------------------------------------------
# P&L analytics
# ---------------------------------------------------------------------------


def get_daily_pnl(session: Session) -> list[tuple[date, Decimal]]:
    """Return P&L grouped by settlement date, ordered chronologically.

    Returns:
        List of (date, daily_pnl) tuples.
    """
    rows = (
        session.execute(
            select(
                func.date(Bet.settled_at).label("day"),
                func.sum(Bet.pnl).label("daily_pnl"),
            )
            .where(
                Bet.settled_at.is_not(None),
                Bet.outcome.in_([BetOutcome.WIN, BetOutcome.LOSS, BetOutcome.VOID]),
            )
            .group_by(func.date(Bet.settled_at))
            .order_by(func.date(Bet.settled_at))
        )
        .all()
    )

    return [
        (row.day, Decimal(str(row.daily_pnl)).quantize(_STAKE_QUANT, ROUND_HALF_UP))
        for row in rows
    ]


def get_cumulative_pnl(session: Session) -> list[tuple[date, Decimal]]:
    """Return cumulative P&L over time, ordered chronologically.

    Returns:
        List of (date, cumulative_pnl) tuples.
    """
    daily = get_daily_pnl(session)
    cumulative: list[tuple[date, Decimal]] = []
    running = Decimal("0.00")
    for day, pnl in daily:
        running = (running + pnl).quantize(_STAKE_QUANT, ROUND_HALF_UP)
        cumulative.append((day, running))
    return cumulative


def calculate_roi(session: Session, initial_bankroll: Decimal) -> Decimal:
    """Compute ROI as (total P&L / initial bankroll) * 100.

    Returns:
        ROI percentage (e.g. 5.25 means +5.25%).
    """
    total_pnl = (
        session.execute(
            select(func.coalesce(func.sum(Bet.pnl), Decimal("0.00"))).where(
                Bet.outcome.in_([BetOutcome.WIN, BetOutcome.LOSS, BetOutcome.VOID])
            )
        )
        .scalar_one()
    )

    if initial_bankroll == 0:
        return Decimal("0.00")

    return (Decimal(str(total_pnl)) / initial_bankroll * Decimal("100")).quantize(
        _PCT_QUANT, ROUND_HALF_UP
    )


def calculate_max_drawdown(
    session: Session,
    initial_bankroll: Decimal,
) -> tuple[Decimal, Decimal]:
    """Compute maximum drawdown from the bankroll equity curve.

    Maximum drawdown is the largest peak-to-trough decline in bankroll value.

    Args:
        session: SQLAlchemy session.
        initial_bankroll: Starting virtual bankroll.

    Returns:
        Tuple of (max_drawdown_absolute, max_drawdown_percentage).
        Both are positive values representing the size of the drawdown.
    """
    # Build equity curve from settled bets ordered by settlement time
    settled = (
        session.execute(
            select(Bet.pnl, Bet.settled_at)
            .where(
                Bet.settled_at.is_not(None),
                Bet.outcome.in_([BetOutcome.WIN, BetOutcome.LOSS, BetOutcome.VOID]),
            )
            .order_by(Bet.settled_at)
        )
        .all()
    )

    if not settled:
        return Decimal("0.00"), Decimal("0.00")

    bankroll = initial_bankroll
    peak = initial_bankroll
    max_dd = Decimal("0.00")
    max_dd_pct = Decimal("0.00")

    for row in settled:
        bankroll = bankroll + Decimal(str(row.pnl))
        if bankroll > peak:
            peak = bankroll
        drawdown = peak - bankroll
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_pct = (
                (drawdown / peak * Decimal("100")).quantize(_PCT_QUANT, ROUND_HALF_UP)
                if peak > 0
                else Decimal("0.00")
            )

    return (
        max_dd.quantize(_STAKE_QUANT, ROUND_HALF_UP),
        max_dd_pct,
    )


def calculate_win_rate(session: Session) -> Decimal:
    """Compute win rate as (wins / settled bets) * 100.

    Only considers WIN and LOSS outcomes (VOID bets are excluded).

    Returns:
        Win rate percentage (e.g. 55.00 means 55%).
    """
    counts = (
        session.execute(
            select(
                func.count(case((Bet.outcome == BetOutcome.WIN, 1))).label("wins"),
                func.count(
                    case((Bet.outcome.in_([BetOutcome.WIN, BetOutcome.LOSS]), 1))
                ).label("decided"),
            )
        )
        .one()
    )

    if counts.decided == 0:
        return Decimal("0.00")

    return (Decimal(str(counts.wins)) / Decimal(str(counts.decided)) * Decimal("100")).quantize(
        _PCT_QUANT, ROUND_HALF_UP
    )


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------


def get_portfolio_stats(
    session: Session,
    initial_bankroll: Decimal,
) -> PortfolioStats:
    """Compute all paper trading metrics in a single call.

    Args:
        session: SQLAlchemy session.
        initial_bankroll: Starting virtual bankroll.

    Returns:
        PortfolioStats dataclass with all metrics.
    """
    # Aggregate counts & sums in one query
    agg = (
        session.execute(
            select(
                func.count(Bet.id).label("total"),
                func.count(case((Bet.outcome == BetOutcome.PENDING, 1))).label("pending"),
                func.count(case((Bet.outcome == BetOutcome.WIN, 1))).label("wins"),
                func.count(case((Bet.outcome == BetOutcome.LOSS, 1))).label("losses"),
                func.count(case((Bet.outcome == BetOutcome.VOID, 1))).label("voids"),
                func.coalesce(func.sum(Bet.stake), Decimal("0.00")).label("total_staked"),
                func.coalesce(
                    func.sum(
                        case(
                            (
                                Bet.outcome.in_(
                                    [BetOutcome.WIN, BetOutcome.LOSS, BetOutcome.VOID]
                                ),
                                Bet.pnl,
                            ),
                            else_=Decimal("0.00"),
                        )
                    ),
                    Decimal("0.00"),
                ).label("total_pnl"),
                func.coalesce(func.avg(Bet.odds_price), Decimal("0.00")).label("avg_odds"),
                func.coalesce(func.avg(Bet.value_edge), Decimal("0.00")).label("avg_edge"),
            )
        )
        .one()
    )

    total = int(agg.total)
    pending = int(agg.pending)
    settled = total - pending
    wins = int(agg.wins)
    losses = int(agg.losses)
    voids = int(agg.voids)
    total_staked = Decimal(str(agg.total_staked)).quantize(_STAKE_QUANT, ROUND_HALF_UP)
    total_pnl = Decimal(str(agg.total_pnl)).quantize(_STAKE_QUANT, ROUND_HALF_UP)
    avg_odds = Decimal(str(agg.avg_odds)).quantize(Decimal("0.0001"), ROUND_HALF_UP)
    avg_edge = Decimal(str(agg.avg_edge)).quantize(Decimal("0.000001"), ROUND_HALF_UP)

    win_rate = calculate_win_rate(session)
    roi = calculate_roi(session, initial_bankroll)
    max_dd, max_dd_pct = calculate_max_drawdown(session, initial_bankroll)
    current_bankroll = get_current_bankroll(session, initial_bankroll)

    # Best / worst day
    daily = get_daily_pnl(session)
    best_day = max((pnl for _, pnl in daily), default=Decimal("0.00"))
    worst_day = min((pnl for _, pnl in daily), default=Decimal("0.00"))

    return PortfolioStats(
        initial_bankroll=initial_bankroll,
        current_bankroll=current_bankroll,
        total_bets=total,
        settled_bets=settled,
        pending_bets=pending,
        wins=wins,
        losses=losses,
        voids=voids,
        win_rate=win_rate,
        total_staked=total_staked,
        total_pnl=total_pnl,
        roi=roi,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        best_day_pnl=best_day,
        worst_day_pnl=worst_day,
        avg_odds=avg_odds,
        avg_edge=avg_edge,
    )


# ---------------------------------------------------------------------------
# ROI breakdown analytics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentROI:
    """ROI statistics for a single segment (league or team)."""

    name: str
    total_bets: int
    wins: int
    losses: int
    total_staked: Decimal
    total_pnl: Decimal
    roi: Decimal  # percentage


def get_roi_by_league(session: Session) -> list[SegmentROI]:
    """Compute ROI broken down by league for settled bets.

    Returns:
        List of SegmentROI, one per league, sorted by ROI descending.
    """
    rows = (
        session.execute(
            select(
                League.name.label("league_name"),
                func.count(Bet.id).label("total_bets"),
                func.count(case((Bet.outcome == BetOutcome.WIN, 1))).label("wins"),
                func.count(case((Bet.outcome == BetOutcome.LOSS, 1))).label("losses"),
                func.coalesce(func.sum(Bet.stake), Decimal("0.00")).label("total_staked"),
                func.coalesce(func.sum(Bet.pnl), Decimal("0.00")).label("total_pnl"),
            )
            .join(Match, Bet.match_id == Match.id)
            .join(League, Match.league_id == League.id)
            .where(Bet.outcome.in_([BetOutcome.WIN, BetOutcome.LOSS, BetOutcome.VOID]))
            .group_by(League.name)
            .order_by(func.sum(Bet.pnl).desc())
        )
        .all()
    )

    result: list[SegmentROI] = []
    for row in rows:
        staked = Decimal(str(row.total_staked))
        pnl = Decimal(str(row.total_pnl))
        roi = (
            (pnl / staked * Decimal("100")).quantize(_PCT_QUANT, ROUND_HALF_UP)
            if staked > 0
            else Decimal("0.00")
        )
        result.append(
            SegmentROI(
                name=row.league_name,
                total_bets=int(row.total_bets),
                wins=int(row.wins),
                losses=int(row.losses),
                total_staked=staked.quantize(_STAKE_QUANT, ROUND_HALF_UP),
                total_pnl=pnl.quantize(_STAKE_QUANT, ROUND_HALF_UP),
                roi=roi,
            )
        )
    return result


def get_roi_by_team(session: Session) -> list[SegmentROI]:
    """Compute ROI broken down by team involved in bet matches.

    A bet is attributed to *both* the home and away teams in the match.
    Returns:
        List of SegmentROI, one per team, sorted by ROI descending.
    """
    # Use a union approach: attribute each bet to both home and away team
    home_q = (
        select(
            Team.name.label("team_name"),
            Bet.id.label("bet_id"),
            Bet.outcome,
            Bet.stake,
            Bet.pnl,
        )
        .join(Match, Bet.match_id == Match.id)
        .join(Team, Match.home_team_id == Team.id)
        .where(Bet.outcome.in_([BetOutcome.WIN, BetOutcome.LOSS, BetOutcome.VOID]))
    )

    away_team = Team.__table__.alias("away_team")
    away_q = (
        select(
            away_team.c.name.label("team_name"),
            Bet.id.label("bet_id"),
            Bet.outcome,
            Bet.stake,
            Bet.pnl,
        )
        .join(Match, Bet.match_id == Match.id)
        .join(away_team, Match.away_team_id == away_team.c.id)
        .where(Bet.outcome.in_([BetOutcome.WIN, BetOutcome.LOSS, BetOutcome.VOID]))
    )

    union = home_q.union_all(away_q).subquery()

    rows = (
        session.execute(
            select(
                union.c.team_name,
                func.count(union.c.bet_id).label("total_bets"),
                func.count(
                    case((union.c.outcome == BetOutcome.WIN.value, 1))
                ).label("wins"),
                func.count(
                    case((union.c.outcome == BetOutcome.LOSS.value, 1))
                ).label("losses"),
                func.coalesce(func.sum(union.c.stake), Decimal("0.00")).label("total_staked"),
                func.coalesce(func.sum(union.c.pnl), Decimal("0.00")).label("total_pnl"),
            )
            .group_by(union.c.team_name)
            .order_by(func.sum(union.c.pnl).desc())
        )
        .all()
    )

    result: list[SegmentROI] = []
    for row in rows:
        staked = Decimal(str(row.total_staked))
        pnl = Decimal(str(row.total_pnl))
        roi = (
            (pnl / staked * Decimal("100")).quantize(_PCT_QUANT, ROUND_HALF_UP)
            if staked > 0
            else Decimal("0.00")
        )
        result.append(
            SegmentROI(
                name=row.team_name,
                total_bets=int(row.total_bets),
                wins=int(row.wins),
                losses=int(row.losses),
                total_staked=staked.quantize(_STAKE_QUANT, ROUND_HALF_UP),
                total_pnl=pnl.quantize(_STAKE_QUANT, ROUND_HALF_UP),
                roi=roi,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Model accuracy analytics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelAccuracyStats:
    """Accuracy metrics for the prediction model on finished matches."""

    total_predictions: int
    correct_predictions: int
    accuracy_pct: Decimal  # percentage (0–100)
    avg_predicted_probability: Decimal
    avg_actual_hit_rate: Decimal  # how often the predicted selection actually won


def get_model_accuracy(session: Session) -> ModelAccuracyStats:
    """Evaluate model accuracy by comparing predictions against actual match results.

    For each prediction on a FINISHED MATCH_WINNER match, determines if the
    predicted selection (home/draw/away) matches the actual result.

    Returns:
        ModelAccuracyStats with overall accuracy metrics.
    """
    # Get all predictions for finished matches
    rows = (
        session.execute(
            select(Prediction, Match)
            .join(Match, Prediction.match_id == Match.id)
            .where(
                Match.status == MatchStatus.FINISHED,
                Match.home_goals.is_not(None),
                Match.away_goals.is_not(None),
            )
        )
        .all()
    )

    if not rows:
        return ModelAccuracyStats(
            total_predictions=0,
            correct_predictions=0,
            accuracy_pct=Decimal("0.00"),
            avg_predicted_probability=Decimal("0.000000"),
            avg_actual_hit_rate=Decimal("0.00"),
        )

    total = 0
    correct = 0
    prob_sum = Decimal("0")

    for prediction, match in rows:
        # Determine actual result
        if match.home_goals > match.away_goals:
            actual = "home"
        elif match.home_goals < match.away_goals:
            actual = "away"
        else:
            actual = "draw"

        total += 1
        prob_sum += Decimal(str(prediction.probability))

        if prediction.selection.lower() == actual:
            correct += 1

    accuracy_pct = (
        (Decimal(str(correct)) / Decimal(str(total)) * Decimal("100")).quantize(
            _PCT_QUANT, ROUND_HALF_UP
        )
        if total > 0
        else Decimal("0.00")
    )

    avg_prob = (prob_sum / Decimal(str(total))).quantize(
        Decimal("0.000001"), ROUND_HALF_UP
    )

    return ModelAccuracyStats(
        total_predictions=total,
        correct_predictions=correct,
        accuracy_pct=accuracy_pct,
        avg_predicted_probability=avg_prob,
        avg_actual_hit_rate=accuracy_pct,
    )


def get_prediction_details(
    session: Session,
) -> list[tuple[str, str, Decimal, bool]]:
    """Return per-prediction detail: (selection, match_label, probability, is_correct).

    Used for building prediction accuracy visualisations.

    Returns:
        List of (selection, match_description, predicted_probability, was_correct) tuples.
    """
    rows = (
        session.execute(
            select(Prediction, Match, Team)
            .join(Match, Prediction.match_id == Match.id)
            .join(Team, Match.home_team_id == Team.id)
            .where(
                Match.status == MatchStatus.FINISHED,
                Match.home_goals.is_not(None),
                Match.away_goals.is_not(None),
            )
            .order_by(Match.kickoff.desc())
        )
        .all()
    )

    results: list[tuple[str, str, Decimal, bool]] = []
    for prediction, match, home_team in rows:
        away_team = session.get(Team, match.away_team_id)
        away_name = away_team.name if away_team else "?"

        if match.home_goals > match.away_goals:
            actual = "home"
        elif match.home_goals < match.away_goals:
            actual = "away"
        else:
            actual = "draw"

        is_correct = prediction.selection.lower() == actual
        match_label = f"{home_team.name} vs {away_name}"

        results.append((
            prediction.selection,
            match_label,
            Decimal(str(prediction.probability)),
            is_correct,
        ))

    return results
