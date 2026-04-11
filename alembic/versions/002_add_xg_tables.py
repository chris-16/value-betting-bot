"""add xG tables and columns

Revision ID: 002
Revises: 001
Create Date: 2026-04-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: str = "001"
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    # New table: team_xg_stats
    op.create_table(
        "team_xg_stats",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("league_id", sa.Integer(), sa.ForeignKey("leagues.id"), nullable=False),
        sa.Column("season", sa.String(20), nullable=False),
        sa.Column("xg", sa.Numeric(10, 2), nullable=False),
        sa.Column("xga", sa.Numeric(10, 2), nullable=False),
        sa.Column("xg_per_match", sa.Numeric(10, 4), nullable=False),
        sa.Column("xga_per_match", sa.Numeric(10, 4), nullable=False),
        sa.Column("matches_played", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("team_id", "league_id", "season", name="uq_team_xg_season"),
    )

    # Add xG columns to matches
    op.add_column("matches", sa.Column("home_xg", sa.Numeric(10, 2), nullable=True))
    op.add_column("matches", sa.Column("away_xg", sa.Numeric(10, 2), nullable=True))


def downgrade() -> None:
    op.drop_column("matches", "away_xg")
    op.drop_column("matches", "home_xg")
    op.drop_table("team_xg_stats")
