"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-04-10
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    # --- Enum types ---
    matchstatus = sa.Enum(
        "SCHEDULED", "LIVE", "FINISHED", "POSTPONED", "CANCELLED",
        name="matchstatus",
        create_type=False,
    )
    markettype = sa.Enum(
        "MATCH_WINNER", "OVER_UNDER_25", "BOTH_TEAMS_TO_SCORE", "DOUBLE_CHANCE",
        name="markettype",
        create_type=False,
    )
    betoutcome = sa.Enum("WIN", "LOSS", "VOID", "PENDING", name="betoutcome", create_type=False)

    matchstatus.create(op.get_bind(), checkfirst=True)
    markettype.create(op.get_bind(), checkfirst=True)
    betoutcome.create(op.get_bind(), checkfirst=True)

    # --- leagues ---
    op.create_table(
        "leagues",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("external_id", sa.String(100), unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # --- teams ---
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("external_id", sa.String(100), unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # --- bookmakers ---
    op.create_table(
        "bookmakers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False, unique=True),
        sa.Column("key", sa.String(100), nullable=False, unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # --- matches ---
    op.create_table(
        "matches",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("league_id", sa.Integer(), sa.ForeignKey("leagues.id"), nullable=False),
        sa.Column("home_team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("away_team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("kickoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", matchstatus, nullable=False),
        sa.Column("home_goals", sa.Integer()),
        sa.Column("away_goals", sa.Integer()),
        sa.Column("external_id", sa.String(100), unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("home_team_id", "away_team_id", "kickoff", name="uq_match"),
    )
    op.create_index("ix_matches_kickoff", "matches", ["kickoff"])
    op.create_index("ix_matches_status", "matches", ["status"])

    # --- odds ---
    op.create_table(
        "odds",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column(
            "bookmaker_id", sa.Integer(), sa.ForeignKey("bookmakers.id"), nullable=False
        ),
        sa.Column("market", markettype, nullable=False),
        sa.Column("selection", sa.String(100), nullable=False),
        sa.Column("price", sa.Numeric(10, 4), nullable=False),
        sa.Column("implied_probability", sa.Numeric(10, 6), nullable=False),
        sa.Column(
            "retrieved_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "match_id",
            "bookmaker_id",
            "market",
            "selection",
            "retrieved_at",
            name="uq_odds_snapshot",
        ),
    )
    op.create_index("ix_odds_match_id", "odds", ["match_id"])

    # --- predictions ---
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("market", markettype, nullable=False),
        sa.Column("selection", sa.String(100), nullable=False),
        sa.Column("probability", sa.Numeric(10, 6), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "match_id", "market", "selection", "model_version", name="uq_prediction"
        ),
    )
    op.create_index("ix_predictions_match_id", "predictions", ["match_id"])

    # --- bets ---
    op.create_table(
        "bets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("market", markettype, nullable=False),
        sa.Column("selection", sa.String(100), nullable=False),
        sa.Column("odds_price", sa.Numeric(10, 4), nullable=False),
        sa.Column("stake", sa.Numeric(10, 2), nullable=False),
        sa.Column("model_probability", sa.Numeric(10, 6), nullable=False),
        sa.Column("implied_probability", sa.Numeric(10, 6), nullable=False),
        sa.Column("value_edge", sa.Numeric(10, 6), nullable=False),
        sa.Column("outcome", betoutcome, nullable=False),
        sa.Column("pnl", sa.Numeric(10, 2)),
        sa.Column("notes", sa.Text()),
        sa.Column(
            "placed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("settled_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_bets_placed_at", "bets", ["placed_at"])


def downgrade() -> None:
    op.drop_table("bets")
    op.drop_table("predictions")
    op.drop_table("odds")
    op.drop_table("matches")
    op.drop_table("bookmakers")
    op.drop_table("teams")
    op.drop_table("leagues")

    op.execute("DROP TYPE IF EXISTS betoutcome")
    op.execute("DROP TYPE IF EXISTS markettype")
    op.execute("DROP TYPE IF EXISTS matchstatus")
