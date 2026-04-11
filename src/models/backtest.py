"""Walk-forward backtester for the Poisson prediction model.

Simulates betting on historical matches with periodic model refitting
to avoid lookahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy.orm import Session

from src.db.models import Match, MatchStatus
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson import PoissonPredictor

logger = logging.getLogger(__name__)

# Refit the model every N matches (walk-forward window)
REFIT_INTERVAL = 50


@dataclass
class BacktestResult:
    """Results from a backtesting run."""

    total_matches: int = 0
    bets_placed: int = 0
    wins: int = 0
    losses: int = 0
    total_staked: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    bankroll_curve: list[float] = field(default_factory=list)
    brier_scores: list[float] = field(default_factory=list)

    @property
    def roi(self) -> float:
        """Return on investment as a percentage."""
        if self.total_staked <= 0:
            return 0.0
        return (self.total_pnl / self.total_staked) * 100

    @property
    def win_rate(self) -> float:
        """Win rate as a percentage."""
        if self.bets_placed <= 0:
            return 0.0
        return (self.wins / self.bets_placed) * 100

    @property
    def avg_brier_score(self) -> float:
        """Average Brier score across all predictions."""
        if not self.brier_scores:
            return 1.0
        return sum(self.brier_scores) / len(self.brier_scores)


def _brier_score(predicted: float, actual: int) -> float:
    """Compute Brier score for a single prediction."""
    return (predicted - actual) ** 2


def backtest(
    session: Session,
    league_id: int | None = None,
    initial_bankroll: float = 1000.0,
    min_edge: float = 0.10,
    kelly_frac: float = 0.50,
    refit_interval: int = REFIT_INTERVAL,
) -> BacktestResult:
    """Run walk-forward backtest on historical matches.

    Periodically refits the Poisson model using only data available at the
    time of each prediction (no lookahead bias).

    Args:
        session: SQLAlchemy session.
        league_id: Optional league filter.
        initial_bankroll: Starting bankroll.
        min_edge: Minimum edge to place a bet.
        kelly_frac: Kelly fraction for stake sizing.
        refit_interval: Refit model every N matches.

    Returns:
        BacktestResult with performance metrics.
    """
    # Fetch all finished matches in chronological order
    query = session.query(Match).filter(
        Match.status == MatchStatus.FINISHED,
        Match.home_goals.is_not(None),
        Match.away_goals.is_not(None),
    )
    if league_id is not None:
        query = query.filter(Match.league_id == league_id)

    matches = query.order_by(Match.kickoff).all()

    if len(matches) < 30:
        logger.warning("Not enough matches for backtest (%d)", len(matches))
        return BacktestResult()

    result = BacktestResult()
    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    result.bankroll_curve.append(bankroll)

    predictor = PoissonPredictor()
    calibrator = ProbabilityCalibrator()

    # Walk-forward: use first 50% for initial fit, backtest on the rest
    split_idx = len(matches) // 2
    train_matches = matches[:split_idx]
    test_matches = matches[split_idx:]

    # Initial fit on training data
    predictor._fit_from_goals_list(train_matches)

    for i, match in enumerate(test_matches):
        result.total_matches += 1

        # Periodically refit using all matches up to this point
        if i > 0 and i % refit_interval == 0:
            all_available = train_matches + test_matches[:i]
            predictor._fit_from_goals_list(all_available)
            logger.debug("Refitted model at match %d", i)

        # Predict
        pred = predictor.predict(match.home_team_id, match.away_team_id)

        # Determine actual outcome
        hg = match.home_goals or 0
        ag = match.away_goals or 0
        if hg > ag:
            actual_outcome = "home"
            actual_binary = {"home": 1, "draw": 0, "away": 0}
        elif hg == ag:
            actual_outcome = "draw"
            actual_binary = {"home": 0, "draw": 1, "away": 0}
        else:
            actual_outcome = "away"
            actual_binary = {"home": 0, "draw": 0, "away": 1}

        # Brier score
        for sel, p in [("home", pred.home_win), ("draw", pred.draw), ("away", pred.away_win)]:
            result.brier_scores.append(_brier_score(p, actual_binary[sel]))

        # Check for value bets (simulate against fair odds)
        for sel, prob in [("home", pred.home_win), ("draw", pred.draw), ("away", pred.away_win)]:
            # Simulate fair implied probability (no margin)
            implied = 1 / 3  # simplified for backtest
            edge = prob - implied

            if edge < min_edge:
                continue

            # Kelly stake
            fair_odds = 1 / implied if implied > 0 else 3.0
            b = fair_odds - 1
            q = 1 - prob
            kelly_full = (b * prob - q) / b if b > 0 else 0
            if kelly_full <= 0:
                continue

            stake = bankroll * kelly_full * kelly_frac
            stake = min(stake, bankroll * 0.10)  # cap at 10% of bankroll

            result.bets_placed += 1
            result.total_staked += stake

            if sel == actual_outcome:
                pnl = stake * (fair_odds - 1)
                result.wins += 1
            else:
                pnl = -stake
                result.losses += 1

            result.total_pnl += pnl
            bankroll += pnl
            result.bankroll_curve.append(bankroll)

            # Track drawdown
            peak_bankroll = max(peak_bankroll, bankroll)
            dd = peak_bankroll - bankroll
            if dd > result.max_drawdown:
                result.max_drawdown = dd
                result.max_drawdown_pct = (
                    (dd / peak_bankroll) * 100 if peak_bankroll > 0 else 0
                )

    logger.info(
        "Backtest complete: %d matches, %d bets, ROI=%.1f%%, Brier=%.4f",
        result.total_matches,
        result.bets_placed,
        result.roi,
        result.avg_brier_score,
    )

    return result
