"""Paper Trading Simulator — Streamlit dashboard page.

Displays virtual bankroll, P&L charts, ROI by league/team, model accuracy,
and a sortable/filterable bet history table.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from src.config import settings
from src.db.models import Bet, League, Match, Team
from src.db.session import SessionLocal
from src.strategies.paper_trading import (
    get_cumulative_pnl,
    get_daily_pnl,
    get_model_accuracy,
    get_portfolio_stats,
    get_prediction_details,
    get_roi_by_league,
    get_roi_by_team,
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

    st.divider()

    # ---------------------------------------------------------------------------
    # ROI by League & Team
    # ---------------------------------------------------------------------------

    st.subheader("ROI Breakdown")

    tab_league, tab_team = st.tabs(["By League", "By Team"])

    with tab_league:
        league_roi = get_roi_by_league(session)
        if league_roi:
            df_league = pd.DataFrame(
                [
                    {
                        "League": seg.name,
                        "Bets": seg.total_bets,
                        "Wins": seg.wins,
                        "Losses": seg.losses,
                        "Staked ($)": float(seg.total_staked),
                        "P&L ($)": float(seg.total_pnl),
                        "ROI (%)": float(seg.roi),
                    }
                    for seg in league_roi
                ]
            )

            # Bar chart
            colors = [
                "#2ecc71" if v >= 0 else "#e74c3c" for v in df_league["ROI (%)"]
            ]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df_league["League"],
                    y=df_league["ROI (%)"],
                    marker_color=colors,
                    text=[f"{v:+.1f}%" for v in df_league["ROI (%)"]],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="ROI by League",
                xaxis_title="League",
                yaxis_title="ROI (%)",
                showlegend=False,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(df_league, use_container_width=True, hide_index=True)
        else:
            st.info("No settled bets by league yet.")

    with tab_team:
        team_roi = get_roi_by_team(session)
        if team_roi:
            df_team = pd.DataFrame(
                [
                    {
                        "Team": seg.name,
                        "Bets": seg.total_bets,
                        "Wins": seg.wins,
                        "Losses": seg.losses,
                        "Staked ($)": float(seg.total_staked),
                        "P&L ($)": float(seg.total_pnl),
                        "ROI (%)": float(seg.roi),
                    }
                    for seg in team_roi
                ]
            )

            # Horizontal bar chart for teams (may be many)
            fig = px.bar(
                df_team.sort_values("ROI (%)", ascending=True),
                x="ROI (%)",
                y="Team",
                orientation="h",
                color="ROI (%)",
                color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                color_continuous_midpoint=0,
                title="ROI by Team",
                text=[
                    f"{v:+.1f}%"
                    for v in df_team.sort_values("ROI (%)", ascending=True)["ROI (%)"]
                ],
            )
            fig.update_layout(
                yaxis_title="",
                height=max(400, len(df_team) * 30),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(df_team, use_container_width=True, hide_index=True)
        else:
            st.info("No settled bets by team yet.")

    st.divider()

    # ---------------------------------------------------------------------------
    # Model Accuracy
    # ---------------------------------------------------------------------------

    st.subheader("Model Accuracy")

    accuracy_stats = get_model_accuracy(session)

    if accuracy_stats.total_predictions > 0:
        col_acc1, col_acc2, col_acc3 = st.columns(3)

        with col_acc1:
            st.metric(
                "Overall Accuracy",
                f"{accuracy_stats.accuracy_pct:.1f}%",
                delta=f"{accuracy_stats.correct_predictions}/{accuracy_stats.total_predictions}",
                delta_color="off",
            )

        with col_acc2:
            st.metric(
                "Total Predictions Evaluated",
                accuracy_stats.total_predictions,
            )

        with col_acc3:
            st.metric(
                "Avg Predicted Probability",
                f"{accuracy_stats.avg_predicted_probability:.4f}",
            )

        # Prediction details for visualisation
        pred_details = get_prediction_details(session)

        if pred_details:
            df_pred = pd.DataFrame(
                pred_details,
                columns=["Selection", "Match", "Predicted Probability", "Correct"],
            )
            df_pred["Predicted Probability"] = df_pred["Predicted Probability"].astype(float)
            df_pred["Result"] = df_pred["Correct"].map({True: "Correct", False: "Incorrect"})

            tab_overview, tab_calibration, tab_detail = st.tabs(
                ["Accuracy by Selection", "Calibration", "Prediction Detail"]
            )

            with tab_overview:
                # Accuracy by selection type (home/draw/away)
                acc_by_sel = (
                    df_pred.groupby("Selection")
                    .agg(
                        Total=("Correct", "count"),
                        Correct=("Correct", "sum"),
                    )
                    .reset_index()
                )
                acc_by_sel["Accuracy (%)"] = (
                    acc_by_sel["Correct"] / acc_by_sel["Total"] * 100
                ).round(1)

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=acc_by_sel["Selection"],
                        y=acc_by_sel["Accuracy (%)"],
                        marker_color=["#3498db", "#e67e22", "#9b59b6"],
                        text=[f"{v:.1f}%" for v in acc_by_sel["Accuracy (%)"]],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    title="Model Accuracy by Selection",
                    xaxis_title="Selection",
                    yaxis_title="Accuracy (%)",
                    yaxis_range=[0, 100],
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(acc_by_sel, use_container_width=True, hide_index=True)

            with tab_calibration:
                # Calibration: bucket predictions by probability, compare to actual hit rate
                df_cal = df_pred.copy()
                df_cal["Prob Bucket"] = pd.cut(
                    df_cal["Predicted Probability"],
                    bins=[0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                    labels=[
                        "0-20%", "20-30%", "30-40%", "40-50%",
                        "50-60%", "60-70%", "70-80%", "80-100%",
                    ],
                )
                cal_grouped = (
                    df_cal.groupby("Prob Bucket", observed=True)
                    .agg(
                        Predictions=("Correct", "count"),
                        Actual_Hit_Rate=("Correct", "mean"),
                        Avg_Predicted=("Predicted Probability", "mean"),
                    )
                    .reset_index()
                )
                cal_grouped["Actual Hit Rate (%)"] = (
                    cal_grouped["Actual_Hit_Rate"] * 100
                ).round(1)
                cal_grouped["Avg Predicted (%)"] = (
                    cal_grouped["Avg_Predicted"] * 100
                ).round(1)

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=cal_grouped["Prob Bucket"],
                        y=cal_grouped["Actual Hit Rate (%)"],
                        name="Actual Hit Rate",
                        marker_color="#2ecc71",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=cal_grouped["Prob Bucket"],
                        y=cal_grouped["Avg Predicted (%)"],
                        name="Avg Predicted Prob",
                        mode="lines+markers",
                        line={"color": "#e74c3c", "width": 2, "dash": "dash"},
                    )
                )
                fig.update_layout(
                    title="Model Calibration: Predicted vs Actual",
                    xaxis_title="Predicted Probability Bucket",
                    yaxis_title="%",
                    yaxis_range=[0, 100],
                    barmode="group",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "A well-calibrated model has actual hit rates close to predicted "
                    "probabilities. Bars above the line indicate the model is under-confident; "
                    "below means over-confident."
                )

            with tab_detail:
                # Sortable detail table
                df_show = df_pred[
                    ["Match", "Selection", "Predicted Probability", "Result"]
                ].copy()
                df_show["Predicted Probability"] = df_show[
                    "Predicted Probability"
                ].map("{:.4f}".format)

                st.dataframe(
                    df_show,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Result": st.column_config.TextColumn(
                            "Result",
                            help="Whether the model's prediction matched the actual outcome",
                        ),
                    },
                )
    else:
        st.info(
            "No predictions evaluated yet. Model accuracy will appear once matches "
            "with predictions are finished."
        )

    st.divider()

    # ---------------------------------------------------------------------------
    # Bet History — sortable & filterable
    # ---------------------------------------------------------------------------

    st.subheader("Bet History")

    # Load all bets with match/team/league context
    all_bets = (
        session.execute(
            select(Bet, Match, League)
            .join(Match, Bet.match_id == Match.id)
            .join(League, Match.league_id == League.id)
            .order_by(Bet.placed_at.desc())
        )
        .all()
    )

    if all_bets:
        rows = []
        for bet, match, league in all_bets:
            home_team = session.get(Team, match.home_team_id)
            away_team = session.get(Team, match.away_team_id)
            home_name = home_team.name if home_team else "?"
            away_name = away_team.name if away_team else "?"
            rows.append(
                {
                    "Date": bet.placed_at.strftime("%Y-%m-%d %H:%M") if bet.placed_at else "",
                    "League": league.name,
                    "Match": f"{home_name} vs {away_name}",
                    "Home Team": home_name,
                    "Away Team": away_name,
                    "Selection": bet.selection,
                    "Odds": float(bet.odds_price),
                    "Stake ($)": float(bet.stake),
                    "Edge": float(bet.value_edge),
                    "Model Prob": float(bet.model_probability),
                    "Outcome": bet.outcome.value.upper(),
                    "P&L ($)": float(bet.pnl) if bet.pnl is not None else None,
                }
            )

        df_bets = pd.DataFrame(rows)

        # --- Sidebar filters ---
        st.sidebar.header("Bet Filters")

        # League filter
        leagues_available = sorted(df_bets["League"].unique().tolist())
        selected_leagues = st.sidebar.multiselect(
            "League",
            options=leagues_available,
            default=leagues_available,
        )

        # Outcome filter
        outcomes_available = sorted(df_bets["Outcome"].unique().tolist())
        selected_outcomes = st.sidebar.multiselect(
            "Outcome",
            options=outcomes_available,
            default=outcomes_available,
        )

        # Selection filter
        selections_available = sorted(df_bets["Selection"].unique().tolist())
        selected_selections = st.sidebar.multiselect(
            "Selection",
            options=selections_available,
            default=selections_available,
        )

        # Apply filters
        df_filtered = df_bets[
            (df_bets["League"].isin(selected_leagues))
            & (df_bets["Outcome"].isin(selected_outcomes))
            & (df_bets["Selection"].isin(selected_selections))
        ]

        st.write(f"Showing **{len(df_filtered)}** of **{len(df_bets)}** bets")

        # Display columns (hide helper columns used for filtering)
        display_cols = [
            "Date", "League", "Match", "Selection", "Odds",
            "Stake ($)", "Edge", "Model Prob", "Outcome", "P&L ($)",
        ]

        st.dataframe(
            df_filtered[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Odds": st.column_config.NumberColumn(format="%.2f"),
                "Stake ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Edge": st.column_config.NumberColumn(format="%.4f"),
                "Model Prob": st.column_config.NumberColumn(format="%.4f"),
                "P&L ($)": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
    else:
        st.info("No bets recorded yet. The bot will place paper bets when value is detected.")

finally:
    session.close()
