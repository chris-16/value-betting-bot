"""Value Betting Bot — Dashboard principal."""

import streamlit as st

from decimal import Decimal
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import select

from src.config import settings
from src.db.models import Bet, BetOutcome, League, Match, ModelRun, Prediction, Team, TeamElo, TeamXGStats
from src.db.session import SessionLocal
from src.strategies.paper_trading import (
    get_cumulative_pnl,
    get_daily_pnl,
    get_model_accuracy,
    get_portfolio_stats,
    get_roi_by_league,
    settle_pending_bets,
)

st.set_page_config(page_title="Value Betting Bot", layout="wide")

session = SessionLocal()

try:
    stats = get_portfolio_stats(session, settings.paper_trading_bankroll)

    # ---- Header ----
    st.title("Value Betting Bot")
    st.caption("Paper trading — apuestas simuladas con IA")

    # ---- Métricas principales ----
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bankroll", f"${stats.current_bankroll:,.2f}", f"${stats.total_pnl:+,.2f}")
    c2.metric("ROI", f"{stats.roi:+.1f}%")
    c3.metric("Aciertos", f"{stats.win_rate:.0f}%", f"{stats.wins}G / {stats.losses}P", delta_color="off")
    c4.metric("Apuestas", stats.total_bets, f"{stats.pending_bets} pendientes", delta_color="off")
    c5.metric("Max Drawdown", f"${stats.max_drawdown:,.2f}", f"-{stats.max_drawdown_pct:.1f}%", delta_color="inverse")

    st.divider()

    # ---- Gráfico P&L ----
    cumulative = get_cumulative_pnl(session)
    daily = get_daily_pnl(session)

    tab_cum, tab_day = st.tabs(["Balance acumulado", "Balance diario"])

    with tab_cum:
        if cumulative:
            df = pd.DataFrame(cumulative, columns=["Fecha", "P&L"])
            df["Fecha"] = pd.to_datetime(df["Fecha"])
            df["P&L"] = df["P&L"].astype(float)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Fecha"], y=df["P&L"], mode="lines+markers",
                fill="tozeroy", fillcolor="rgba(46,204,113,0.1)",
                line={"color": "#2ecc71", "width": 2},
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(xaxis_title="", yaxis_title="Balance ($)", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Todavía no hay apuestas resueltas.")

    with tab_day:
        if daily:
            df = pd.DataFrame(daily, columns=["Fecha", "P&L"])
            df["Fecha"] = pd.to_datetime(df["Fecha"])
            df["P&L"] = df["P&L"].astype(float)
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["P&L"]]
            fig = go.Figure(go.Bar(x=df["Fecha"], y=df["P&L"], marker_color=colors))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(xaxis_title="", yaxis_title="Balance ($)", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Todavía no hay apuestas resueltas.")

    st.divider()

    # ---- ROI por liga ----
    league_roi = get_roi_by_league(session)
    if league_roi:
        st.subheader("Rendimiento por liga")
        df_lr = pd.DataFrame([{
            "Liga": s.name, "Apuestas": s.total_bets,
            "Ganadas": s.wins, "Perdidas": s.losses,
            "Apostado": float(s.total_staked),
            "Balance": float(s.total_pnl),
            "ROI %": float(s.roi),
        } for s in league_roi])
        st.dataframe(df_lr, use_container_width=True, hide_index=True)
        st.divider()

    # ---- Precisión del modelo ----
    accuracy = get_model_accuracy(session)
    if accuracy.total_predictions > 0:
        st.subheader("Precisión del modelo")
        ca1, ca2, ca3 = st.columns(3)
        ca1.metric("Aciertos", f"{accuracy.accuracy_pct:.1f}%",
                    f"{accuracy.correct_predictions}/{accuracy.total_predictions}", delta_color="off")
        ca2.metric("Predicciones evaluadas", accuracy.total_predictions)
        ca3.metric("Prob. promedio predicha", f"{accuracy.avg_predicted_probability:.2%}")
        st.divider()

    # ---- Historial de apuestas ----
    st.subheader("Historial de apuestas")

    all_bets = (
        session.execute(
            select(Bet, Match, League)
            .join(Match, Bet.match_id == Match.id)
            .join(League, Match.league_id == League.id)
            .order_by(Bet.placed_at.desc())
        ).all()
    )

    if all_bets:
        rows = []
        for bet, match, league in all_bets:
            home = session.get(Team, match.home_team_id)
            away = session.get(Team, match.away_team_id)
            home_name = home.name if home else "?"
            away_name = away.name if away else "?"

            # Selection label
            if bet.selection == "home":
                sel_label = home_name
            elif bet.selection == "away":
                sel_label = away_name
            else:
                sel_label = "Empate"

            outcome_map = {
                BetOutcome.WIN: "Ganada",
                BetOutcome.LOSS: "Perdida",
                BetOutcome.VOID: "Anulada",
                BetOutcome.PENDING: "Pendiente",
            }

            rows.append({
                "Fecha": bet.placed_at.strftime("%d/%m %H:%M") if bet.placed_at else "",
                "Liga": league.name,
                "Partido": f"{home_name} vs {away_name}",
                "Apuesta": sel_label,
                "Certeza": f"{float(bet.model_probability) * 100:.0f}%",
                "Ventaja": f"{float(bet.value_edge) * 100:.1f}%",
                "Odds": f"{float(bet.odds_price):.2f}x",
                "Monto": f"${float(bet.stake):.2f}",
                "Resultado": outcome_map.get(bet.outcome, "?"),
                "Balance": f"${float(bet.pnl):+.2f}" if bet.pnl is not None else "—",
            })

        df_bets = pd.DataFrame(rows)

        # Filtros
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            leagues_avail = sorted(df_bets["Liga"].unique().tolist())
            sel_leagues = st.multiselect("Filtrar por liga", leagues_avail, default=leagues_avail)
        with col_f2:
            outcomes_avail = sorted(df_bets["Resultado"].unique().tolist())
            sel_outcomes = st.multiselect("Filtrar por resultado", outcomes_avail, default=outcomes_avail)

        df_filtered = df_bets[
            df_bets["Liga"].isin(sel_leagues) & df_bets["Resultado"].isin(sel_outcomes)
        ]

        st.write(f"**{len(df_filtered)}** apuestas")
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)
    else:
        st.info("Todavía no hay apuestas. El bot apostará cuando detecte oportunidades.")

    # ---- Predicciones pendientes ----
    pending_predictions = (
        session.execute(
            select(Prediction, Match, League)
            .join(Match, Prediction.match_id == Match.id)
            .join(League, Match.league_id == League.id)
            .where(Match.status == "SCHEDULED")
            .order_by(Match.kickoff)
        ).all()
    )

    if pending_predictions:
        st.divider()
        st.subheader("Próximas predicciones")
        pred_rows = []
        seen = set()
        for pred, match, league in pending_predictions:
            if match.id in seen:
                continue
            seen.add(match.id)

            home = session.get(Team, match.home_team_id)
            away = session.get(Team, match.away_team_id)

            # Get all 3 predictions for this match
            match_preds = [p for p, m, l in pending_predictions if m.id == match.id]
            probs = {p.selection: float(p.probability) for p in match_preds}

            pred_rows.append({
                "Fecha": match.kickoff.strftime("%d/%m %H:%M") if match.kickoff else "",
                "Liga": league.name,
                "Partido": f"{home.name if home else '?'} vs {away.name if away else '?'}",
                "Local": f"{probs.get('home', 0) * 100:.0f}%",
                "Empate": f"{probs.get('draw', 0) * 100:.0f}%",
                "Visitante": f"{probs.get('away', 0) * 100:.0f}%",
            })

        if pred_rows:
            st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

    # ---- xG Data Status ----
    st.divider()
    st.subheader("Estado datos xG")

    xg_stats = session.query(TeamXGStats).all()
    if xg_stats:
        xg_rows = []
        for stat in xg_stats:
            team = session.get(Team, stat.team_id)
            league = session.get(League, stat.league_id)
            xg_rows.append({
                "Equipo": team.name if team else f"#{stat.team_id}",
                "Liga": league.name if league else "?",
                "Temporada": stat.season,
                "xG": float(stat.xg),
                "xGA": float(stat.xga),
                "xG/partido": float(stat.xg_per_match),
                "xGA/partido": float(stat.xga_per_match),
                "Partidos": stat.matches_played,
            })
        st.write(f"**{len(xg_stats)}** equipos con datos xG")
        st.dataframe(pd.DataFrame(xg_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No hay datos xG cargados. Ejecutá `python -m src.data.loader --xg`.")

    # ---- Model Runs / Calibration ----
    model_runs = (
        session.query(ModelRun)
        .order_by(ModelRun.trained_at.desc())
        .limit(10)
        .all()
    )

    if model_runs:
        st.divider()
        st.subheader("Historial de entrenamiento del modelo")
        run_rows = []
        for run in model_runs:
            run_rows.append({
                "Versión": run.model_version,
                "Fecha": run.trained_at.strftime("%d/%m/%Y %H:%M") if run.trained_at else "",
                "Partidos": run.train_matches,
                "Brier Score": f"{run.brier_score:.4f}" if run.brier_score else "—",
                "Log Loss": f"{run.log_loss:.4f}" if run.log_loss else "—",
                "Notas": run.notes or "",
            })
        st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)

        # Calibration reliability diagram (if we have Brier scores)
        brier_vals = [r.brier_score for r in model_runs if r.brier_score is not None]
        if brier_vals:
            st.subheader("Brier Score por entrenamiento")
            df_brier = pd.DataFrame({
                "Fecha": [r.trained_at for r in model_runs if r.brier_score is not None],
                "Brier": brier_vals,
            })
            df_brier["Fecha"] = pd.to_datetime(df_brier["Fecha"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_brier["Fecha"], y=df_brier["Brier"],
                mode="lines+markers",
                line={"color": "#3498db", "width": 2},
            ))
            fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                          annotation_text="Coinflip baseline (0.25)")
            fig.update_layout(
                xaxis_title="", yaxis_title="Brier Score (menor = mejor)",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

finally:
    session.close()
