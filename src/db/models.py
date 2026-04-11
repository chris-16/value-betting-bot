"""SQLAlchemy ORM models for all core entities."""

from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.db.session import Base


class MatchStatus(enum.Enum):
    """Status of a football match."""

    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class BetOutcome(enum.Enum):
    """Outcome of a resolved bet."""

    WIN = "win"
    LOSS = "loss"
    VOID = "void"
    PENDING = "pending"


class MarketType(enum.Enum):
    """Supported betting market types."""

    MATCH_WINNER = "match_winner"  # 1X2
    OVER_UNDER_25 = "over_under_2.5"
    BOTH_TEAMS_TO_SCORE = "btts"
    DOUBLE_CHANCE = "double_chance"


class TeamElo(Base):
    """ELO rating for a team."""

    __tablename__ = "team_elo"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False, unique=True)
    rating: Mapped[Decimal] = mapped_column(Numeric(10, 1), nullable=False, default=1500.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    team: Mapped[Team] = relationship()

    def __repr__(self) -> str:
        return f"<TeamElo team={self.team_id} rating={self.rating}>"


class League(Base):
    """Football league / competition."""

    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    matches: Mapped[list[Match]] = relationship(back_populates="league")

    def __repr__(self) -> str:
        return f"<League {self.name} ({self.country})>"


class Team(Base):
    """Football team."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    home_matches: Mapped[list[Match]] = relationship(
        back_populates="home_team", foreign_keys="Match.home_team_id"
    )
    away_matches: Mapped[list[Match]] = relationship(
        back_populates="away_team", foreign_keys="Match.away_team_id"
    )

    def __repr__(self) -> str:
        return f"<Team {self.name}>"


class Match(Base):
    """Football match."""

    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("home_team_id", "away_team_id", "kickoff", name="uq_match"),
        Index("ix_matches_kickoff", "kickoff"),
        Index("ix_matches_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), nullable=False)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    kickoff: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[MatchStatus] = mapped_column(
        Enum(MatchStatus), default=MatchStatus.SCHEDULED, nullable=False
    )
    home_goals: Mapped[int | None] = mapped_column()
    away_goals: Mapped[int | None] = mapped_column()
    home_xg: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    away_xg: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    external_id: Mapped[str | None] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    league: Mapped[League] = relationship(back_populates="matches")
    home_team: Mapped[Team] = relationship(
        back_populates="home_matches", foreign_keys=[home_team_id]
    )
    away_team: Mapped[Team] = relationship(
        back_populates="away_matches", foreign_keys=[away_team_id]
    )
    odds: Mapped[list[Odds]] = relationship(back_populates="match")
    predictions: Mapped[list[Prediction]] = relationship(back_populates="match")
    bets: Mapped[list[Bet]] = relationship(back_populates="match")

    def __repr__(self) -> str:
        return f"<Match {self.home_team_id} vs {self.away_team_id} @ {self.kickoff}>"


class Bookmaker(Base):
    """Bookmaker / sportsbook."""

    __tablename__ = "bookmakers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    odds: Mapped[list[Odds]] = relationship(back_populates="bookmaker")

    def __repr__(self) -> str:
        return f"<Bookmaker {self.name}>"


class Odds(Base):
    """Bookmaker odds for a specific match and market."""

    __tablename__ = "odds"
    __table_args__ = (
        UniqueConstraint(
            "match_id",
            "bookmaker_id",
            "market",
            "selection",
            "retrieved_at",
            name="uq_odds_snapshot",
        ),
        Index("ix_odds_match_id", "match_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    bookmaker_id: Mapped[int] = mapped_column(ForeignKey("bookmakers.id"), nullable=False)
    market: Mapped[MarketType] = mapped_column(Enum(MarketType), nullable=False)
    selection: Mapped[str] = mapped_column(String(100), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    implied_probability: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    match: Mapped[Match] = relationship(back_populates="odds")
    bookmaker: Mapped[Bookmaker] = relationship(back_populates="odds")

    def __repr__(self) -> str:
        return f"<Odds {self.selection}={self.price} ({self.bookmaker_id})>"


class Prediction(Base):
    """ML model prediction for a match outcome."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "match_id",
            "market",
            "selection",
            "model_version",
            name="uq_prediction",
        ),
        Index("ix_predictions_match_id", "match_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    market: Mapped[MarketType] = mapped_column(Enum(MarketType), nullable=False)
    selection: Mapped[str] = mapped_column(String(100), nullable=False)
    probability: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    match: Mapped[Match] = relationship(back_populates="predictions")

    def __repr__(self) -> str:
        return f"<Prediction {self.selection} p={self.probability} v{self.model_version}>"


class Bet(Base):
    """Paper trading bet placed by the bot."""

    __tablename__ = "bets"
    __table_args__ = (Index("ix_bets_placed_at", "placed_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    market: Mapped[MarketType] = mapped_column(Enum(MarketType), nullable=False)
    selection: Mapped[str] = mapped_column(String(100), nullable=False)
    odds_price: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    stake: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    model_probability: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    implied_probability: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    value_edge: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    outcome: Mapped[BetOutcome] = mapped_column(
        Enum(BetOutcome), default=BetOutcome.PENDING, nullable=False
    )
    pnl: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    notes: Mapped[str | None] = mapped_column(Text)
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    match: Mapped[Match] = relationship(back_populates="bets")

    def __repr__(self) -> str:
        return f"<Bet {self.selection} stake={self.stake} outcome={self.outcome}>"


class TeamXGStats(Base):
    """Expected goals statistics for a team in a season."""

    __tablename__ = "team_xg_stats"
    __table_args__ = (
        UniqueConstraint("team_id", "league_id", "season", name="uq_team_xg_season"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), nullable=False)
    season: Mapped[str] = mapped_column(String(20), nullable=False)
    xg: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    xga: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    xg_per_match: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    xga_per_match: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    matches_played: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    team: Mapped[Team] = relationship()
    league: Mapped[League] = relationship()

    def __repr__(self) -> str:
        return f"<TeamXGStats team={self.team_id} xG={self.xg} xGA={self.xga}>"


class ModelRun(Base):
    """Tracking table for model training runs."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    train_matches: Mapped[int] = mapped_column(default=0)
    brier_score: Mapped[float | None] = mapped_column(Float)
    log_loss: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<ModelRun {self.model_version} brier={self.brier_score}>"
