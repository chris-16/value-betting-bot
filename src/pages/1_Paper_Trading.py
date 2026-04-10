"""Paper Trading Simulator — Streamlit dashboard page.

Displays virtual bankroll, P&L charts, and key performance metrics.
Allows settling pending bets and configuring the initial bankroll.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from src.config import settings
from src.db.models import Bet, Match, Team
from src.db.session import SessionLocal
from src.strategies.paper_trading import (
    get_cumulative_pnl,
    get_daily_pnl,
    get_portfolio_stats,
    settle_pending_bets,
)

st.set_page_config(page_title="Paper Trading", page_icon="", layout="wide")
st.title("Paper Trading Simulator")
st.write("Track hypothetical bets, P&L, and portfolio performance.")


# ---------------------------------------------------------------------------
# Sidebar: bankroll configuration
# ---------------------------------------------------------------------------

st.sidebar.header("Configuration")
initial_bankroll = Decimal(
    str(
        st.sidebar.number_input(
            "Initial Bankroll ($)",
            min_value=1.0,
            value=float(settings.paper_trading_bankroll),
            step=100.0,
            format="%.2f",
        )
    )
)

# ---------------------------------------------------------------------------
# Session & settlement
# ---------------------------------------------------------------------------

session = SessionLocal()

try:
    # Settlement action
    if st.sidebar.button("Settle Pending Bets"):
        settled = settle_pending_bets(session)
        if settled:
            st.sidebar.success(f"Settled {len(settled)} bet(s).")
        else:
            st.sidebar.info("No pending bets to settle.")

    # ---------------------------------------------------------------------------
    # Portfolio stats
    # ---------------------------------------------------------------------------

    stats = get_portfolio_stats(session, initial_bankroll)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Bankroll",
            f"${stats.current_bankroll:,.2f}",
            delta=f"${stats.total_pnl:,.2f}",
        )

    with col2:
        st.metric(
            "ROI",
            f"{stats.roi:+.2f}%",
        )

    with col3:
        st.metric(
            "Win Rate",
            f"{stats.win_rate:.1f}%",
            delta=f"{stats.wins}W / {stats.losses}L",
            delta_color="off",
        )

    with col4:
        st.metric(
            "Max Drawdown",
            f"${stats.max_drawdown:,.2f}",
            delta=f"-{stats.max_drawdown_pct:.1f}%",
            delta_color="inverse",
        )

    st.divider()

    # Secondary metrics
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("Total Bets", stats.total_bets)

    with col6:
        st.metric("Pending", stats.pending_bets)

    with col7:
        st.metric("Avg Odds", f"{stats.avg_odds:.2f}")

    with col8:
        st.metric("Avg Edge", f"{stats.avg_edge:.4f}")

    st.divider()

    # ---------------------------------------------------------------------------
    # P&L charts
    # ---------------------------------------------------------------------------

    tab_cumulative, tab_daily = st.tabs(["Cumulative P&L", "Daily P&L"])

    cumulative = get_cumulative_pnl(session)
    daily = get_daily_pnl(session)

    with tab_cumulative:
        if cumulative:
            df_cum = pd.DataFrame(cumulative, columns=["Date", "Cumulative P&L"])
            df_cum["Date"] = pd.to_datetime(df_cum["Date"])
            df_cum["Cumulative P&L"] = df_cum["Cumulative P&L"].astype(float)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_cum["Date"],
                    y=df_cum["Cumulative P&L"],
                    mode="lines+markers",
                    name="Cumulative P&L",
                    line={"color": "#2ecc71", "width": 2},
                    fill="tozeroy",
                    fillcolor="rgba(46, 204, 113, 0.1)",
                )
            )
            fig.update_layout(
                title="Cumulative P&L Over Time",
                xaxis_title="Date",
                yaxis_title="P&L ($)",
                hovermode="x unified",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No settled bets yet. P&L chart will appear after bets are settled.")

    with tab_daily:
        if daily:
            df_daily = pd.DataFrame(daily, columns=["Date", "Daily P&L"])
            df_daily["Date"] = pd.to_datetime(df_daily["Date"])
            df_daily["Daily P&L"] = df_daily["Daily P&L"].astype(float)

            colors = [
                "#2ecc71" if v >= 0 else "#e74c3c" for v in df_daily["Daily P&L"]
            ]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df_daily["Date"],
                    y=df_daily["Daily P&L"],
                    marker_color=colors,
                    name="Daily P&L",
                )
            )
            fig.update_layout(
                title="Daily P&L",
                xaxis_title="Date",
                yaxis_title="P&L ($)",
                hovermode="x unified",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No settled bets yet.")

    st.divider()

    # ---------------------------------------------------------------------------
    # Best / worst day
    # ---------------------------------------------------------------------------

    col_best, col_worst = st.columns(2)
    with col_best:
        st.metric("Best Day", f"${stats.best_day_pnl:+,.2f}")
    with col_worst:
        st.metric("Worst Day", f"${stats.worst_day_pnl:+,.2f}")

    # ---------------------------------------------------------------------------
    # Recent bets table
    # ---------------------------------------------------------------------------

    st.subheader("Recent Bets")

    recent_bets = (
        session.execute(
            select(Bet, Match, Team)
            .join(Match, Bet.match_id == Match.id)
            .join(Team, Match.home_team_id == Team.id)
            .order_by(Bet.placed_at.desc())
            .limit(50)
        )
        .all()
    )

    if recent_bets:
        rows = []
        for bet, match, home_team in recent_bets:
            away_team = session.get(Team, match.away_team_id)
            away_name = away_team.name if away_team else "?"
            rows.append(
                {
                    "Date": bet.placed_at.strftime("%Y-%m-%d %H:%M") if bet.placed_at else "",
                    "Match": f"{home_team.name} vs {away_name}",
                    "Selection": bet.selection,
                    "Odds": float(bet.odds_price),
                    "Stake ($)": float(bet.stake),
                    "Edge": float(bet.value_edge),
                    "Outcome": bet.outcome.value.upper(),
                    "P&L ($)": float(bet.pnl) if bet.pnl is not None else "",
                }
            )

        df_bets = pd.DataFrame(rows)
        st.dataframe(df_bets, use_container_width=True, hide_index=True)
    else:
        st.info("No bets recorded yet. The bot will place paper bets when value is detected.")

finally:
    session.close()
