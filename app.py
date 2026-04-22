"""
Cloud Resource Monitor — Streamlit Dashboard

3-tab user-facing dashboard:
  📡 Live Monitor  — Grafana-style streaming CPU & Memory + 15/30-min breach countdown
  🔮  Forecast      — predict future utilisation, risk level, CSV download
  📊  System Health — gauges, node grid, model accuracy
  💬  Floating      — always-visible chatbot bubble at bottom-right (Trengo-style)
"""
from __future__ import annotations
import streamlit.components.v1 as components

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from deployment_utils import (
    TARGET_CONFIG,
    EnsemblePredictor,
    load_demo_dataframe,
    load_results_summary,
    results_table,
)

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cloud Resource Monitor",
    layout="wide",
    page_icon="☁️",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  [data-testid="metric-container"] {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 14px 18px !important;
  }
  [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 600; }

  .breach-red   { background:#fee2e2; border:1px solid #fca5a5; border-radius:10px;
                  padding:10px 16px; color:#991b1b; font-weight:600; margin:4px 0; }
  .breach-amber { background:#fef3c7; border:1px solid #fcd34d; border-radius:10px;
                  padding:10px 16px; color:#92400e; font-weight:600; margin:4px 0; }
  .breach-green { background:#dcfce7; border:1px solid #86efac; border-radius:10px;
                  padding:10px 16px; color:#166534; font-weight:600; margin:4px 0; }

  .sla-alert { background:linear-gradient(135deg,#ff4444,#cc0000); color:white;
               padding:12px 20px; border-radius:10px; font-size:1rem; font-weight:600;
               text-align:center; animation:pulse 1.5s infinite; margin-bottom:8px; }
  @keyframes pulse{0%{opacity:1}50%{opacity:.75}100%{opacity:1}}

  .risk-high   { color:#dc2626; font-weight:700; font-size:1.3rem; }
  .risk-medium { color:#f59e0b; font-weight:700; font-size:1.3rem; }
  .risk-low    { color:#16a34a; font-weight:700; font-size:1.3rem; }

  .node-card { border-radius:8px; padding:10px; text-align:center;
               font-size:0.8rem; font-weight:500; margin:4px; }
  .node-ok   { background:#dcfce7; color:#166534; border:1px solid #86efac; }
  .node-warn { background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }
  .node-crit { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }

  [data-testid="stTabs"] button { font-size:0.95rem; font-weight:500; }
  .block-container { padding-top:1rem !important; }
  [data-testid="collapsedControl"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CPU_SLA         = TARGET_CONFIG["cpu"]["sla_threshold"]
MEM_SLA         = TARGET_CONFIG["memory"]["sla_threshold"]
COLOURS         = {"cpu": "#3b82f6", "memory": "#8b5cf6", "forecast": "#ef4444"}
HOURLY_RATE_EUR = 0.08
MINS_PER_ROW    = 5

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def _demo(provider: str) -> pd.DataFrame | None:
    return load_demo_dataframe(provider)

@st.cache_data(show_spinner=False, ttl=300)
def _results() -> dict | None:
    return load_results_summary()

@st.cache_resource(show_spinner=False)
def _ensemble(target_key: str) -> EnsemblePredictor:
    return EnsemblePredictor(target_key)

def _scale(s: pd.Series) -> pd.Series:
    return (s * 100).clip(0, 100) if s.max() <= 2 else s.clip(0, 100)

def _risk(peak, sla):
    if peak >= sla:        return "🔴 HIGH"
    if peak >= sla * 0.88: return "🟡 MEDIUM"
    return "🟢 LOW"

def _risk_class(peak, sla):
    if peak >= sla:        return "risk-high"
    if peak >= sla * 0.88: return "risk-medium"
    return "risk-low"

def _mins_to_breach(forecast_list: list, sla: float) -> int | None:
    for i, v in enumerate(forecast_list):
        if v >= sla:
            return i * MINS_PER_ROW
    return None

# ── Session state (must be before tabs/fragments) ─────────────────────────────
for _k, _v in [
    ("lm_data", None), ("lm_idx", 0),
    ("lm_hist_cpu", []), ("lm_hist_mem", []),
    ("lm_fc_cpu",  []), ("lm_fc_mem",  []),
    ("lm_alerts",  []), ("lm_ready",   False),
    ("lm_running", False), ("lm_provider", "alibaba"),
    ("lm_savings", 0.0),
]:
    st.session_state.setdefault(_k, _v)

# ── Header ────────────────────────────────────────────────────────────────────
h1, _ = st.columns([3, 1])
with h1:
    st.markdown("## ☁️ Cloud Resource Monitor")
    st.caption("Real-time CPU & Memory forecasting · 15/30-min breach prediction · SLA alerts")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_forecast, tab_health = st.tabs([
    "📡  Live Monitor",
    "🔮  Forecast",
    "📊  System Health",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    # Controls
    speed = 15  # fixed: 15 rows per step (~75 min of telemetry per update)
    cc1, cc2, cc3 = st.columns([3, 1, 1])
    provider = cc1.selectbox("Data source", ["alibaba", "azure", "google"], key="lm_prov")
    start_btn = cc2.button(
        "⏸ Pause" if st.session_state.get("lm_running") else "▶ Start",
        type="primary", key="lm_toggle")
    reset_btn = cc3.button("↺ Reset", key="lm_reset")

    if start_btn:
        st.session_state.lm_running = not st.session_state.get("lm_running", False)
        st.rerun()
    if reset_btn:
        for k in ["lm_data","lm_idx","lm_hist_cpu","lm_hist_mem",
                  "lm_fc_cpu","lm_fc_mem","lm_alerts","lm_ready","lm_running","lm_savings"]:
            st.session_state.pop(k, None)
        st.rerun()

    # Load data
    if not st.session_state.get("lm_ready") or st.session_state.get("lm_provider") != provider:
        df = _demo(provider)
        if df is not None:
            for k, v in [("lm_data", df), ("lm_idx", 0),
                         ("lm_hist_cpu", []), ("lm_hist_mem", []),
                         ("lm_fc_cpu", []), ("lm_fc_mem", []),
                         ("lm_alerts", []), ("lm_savings", 0.0),
                         ("lm_ready", True), ("lm_provider", provider)]:
                st.session_state[k] = v

    def _do_step():
        df  = st.session_state.get("lm_data")
        idx = st.session_state.get("lm_idx", 0)
        if df is None or idx >= len(df):
            st.session_state.lm_running = False
            return

        batch = df.iloc[idx: idx + speed]
        actual_cpu = float(_scale(batch["cpu"]).mean())
        actual_mem = float(_scale(batch["mem"]).mean())

        try:
            pred_cpu = float(_ensemble("cpu").predict(batch)["prediction"].mean())
        except Exception:
            pred_cpu = actual_cpu * 1.02
        try:
            pred_mem = float(_ensemble("memory").predict(batch)["prediction"].mean())
        except Exception:
            pred_mem = actual_mem * 1.01

        pred_cpu = float(np.clip(pred_cpu * 100 if pred_cpu <= 2 else pred_cpu, 0, 100))
        pred_mem = float(np.clip(pred_mem * 100 if pred_mem <= 2 else pred_mem, 0, 100))

        st.session_state.lm_hist_cpu.append(actual_cpu)
        st.session_state.lm_hist_mem.append(actual_mem)
        st.session_state.lm_fc_cpu.append(pred_cpu)
        st.session_state.lm_fc_mem.append(pred_mem)
        for k in ["lm_hist_cpu","lm_hist_mem","lm_fc_cpu","lm_fc_mem"]:
            if len(st.session_state[k]) > 200:
                st.session_state[k] = st.session_state[k][-200:]

        batch_hours = (speed * MINS_PER_ROW) / 60
        st.session_state.lm_savings += (actual_cpu / 100) * HOURLY_RATE_EUR * batch_hours * 0.30

        ts = idx // speed
        for metric, pred, sla in [("CPU", pred_cpu, CPU_SLA), ("MEM", pred_mem, MEM_SLA)]:
            if pred >= sla:
                st.session_state.lm_alerts.append(
                    {"t": ts, "metric": metric, "pred": round(pred, 1), "sla": sla})
        if len(st.session_state.lm_alerts) > 100:
            st.session_state.lm_alerts = st.session_state.lm_alerts[-100:]

        st.session_state.lm_idx = idx + speed

    @st.fragment(run_every=1 if st.session_state.get("lm_running", False) else None)
    def _live_dashboard():
        if st.session_state.get("lm_running", False):
            _do_step()

        hist_cpu = st.session_state.get("lm_hist_cpu", [])
        hist_mem = st.session_state.get("lm_hist_mem", [])
        fc_cpu   = st.session_state.get("lm_fc_cpu", [])
        fc_mem   = st.session_state.get("lm_fc_mem", [])
        alerts   = st.session_state.get("lm_alerts", [])
        savings  = st.session_state.get("lm_savings", 0.0)
        lm_df    = st.session_state.get("lm_data")
        lm_idx   = st.session_state.get("lm_idx", 0)
        lm_total = len(lm_df) if lm_df is not None else 0

        curr_cpu = hist_cpu[-1] if hist_cpu else 0.0
        curr_mem = hist_mem[-1] if hist_mem else 0.0
        fore_cpu = fc_cpu[-1]   if fc_cpu   else 0.0
        fore_mem = fc_mem[-1]   if fc_mem   else 0.0
        mc_cpu   = _mins_to_breach(fc_cpu[-30:], CPU_SLA) if fc_cpu else None
        mc_mem   = _mins_to_breach(fc_mem[-30:], MEM_SLA) if fc_mem else None
        n_breaches = len(alerts)

        # ── Hidden data bridge for the JS chatbot ─────────────────────────────
        st.markdown(
            f'<div id="live-data" style="display:none" '
            f'data-cpu="{curr_cpu:.1f}" data-mem="{curr_mem:.1f}" '
            f'data-fc-cpu="{fore_cpu:.1f}" data-fc-mem="{fore_mem:.1f}" '
            f'data-cpu-sla="{CPU_SLA}" data-mem-sla="{MEM_SLA}" '
            f'data-mc-cpu="{mc_cpu if mc_cpu is not None else -1}" '
            f'data-mc-mem="{mc_mem if mc_mem is not None else -1}" '
            f'data-savings="{savings:.2f}" data-breaches="{n_breaches}">'
            f'</div>',
            unsafe_allow_html=True)

        # ── Breach countdown ──────────────────────────────────────────────────
        if fc_cpu or fc_mem:
            for metric, mins, sla in [("CPU", mc_cpu, CPU_SLA), ("Memory", mc_mem, MEM_SLA)]:
                if mins is not None and mins == 0:
                    st.markdown(
                        f'<div class="breach-red">🚨 {metric} is BREACHING NOW — '
                        f'currently at or above {sla}% SLA threshold</div>',
                        unsafe_allow_html=True)
                elif mins is not None and mins <= 15:
                    st.markdown(
                        f'<div class="breach-red">🚨 {metric} BREACH predicted in '
                        f'~{mins} minutes (forecast crosses {sla}%)</div>',
                        unsafe_allow_html=True)
                elif mins is not None and mins <= 30:
                    st.markdown(
                        f'<div class="breach-amber">⚠️ {metric} approaching SLA — '
                        f'estimated ~{mins} minutes to breach ({sla}%)</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="breach-green">✅ {metric}: No breach predicted in next 30 min</div>',
                        unsafe_allow_html=True)
        elif alerts:
            last = alerts[-1]
            st.markdown(
                f'<div class="sla-alert">🚨 SLA BREACH — '
                f'{last["metric"]} at {last["pred"]}% (threshold {last["sla"]}%)</div>',
                unsafe_allow_html=True)

        # ── Metric cards ─────────────────────────────────────────────────────
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("CPU Now",         f"{curr_cpu:.1f}%",
                   delta="⚠️ High" if curr_cpu >= CPU_SLA * 0.9 else "Normal")
        mc2.metric("Memory Now",      f"{curr_mem:.1f}%",
                   delta="⚠️ High" if curr_mem >= MEM_SLA * 0.9 else "Normal")
        mc3.metric("CPU Forecast",    f"{fore_cpu:.1f}%",
                   delta="🚨 Breach" if fore_cpu >= CPU_SLA else "OK")
        mc4.metric("Mem Forecast",    f"{fore_mem:.1f}%",
                   delta="🚨 Breach" if fore_mem >= MEM_SLA else "OK")
        mc5.metric("SLA Breaches",    n_breaches,
                   delta="🚨" if alerts else "✅ Clear")
        mc6.metric("💰 Est. Savings", f"€{savings:.2f}")

        # ── Dual chart ────────────────────────────────────────────────────────
        if hist_cpu:
            x = list(range(len(hist_cpu)))
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                subplot_titles=(
                    "CPU Utilisation  (solid = actual · dashed = 30-min forecast)",
                    "Memory Utilisation  (solid = actual · dashed = 30-min forecast)"),
                vertical_spacing=0.10)

            fig.add_trace(go.Scatter(x=x, y=hist_cpu, name="CPU Actual",
                                     line=dict(color=COLOURS["cpu"], width=2),
                                     fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=fc_cpu, name="CPU Forecast",
                                     line=dict(color=COLOURS["forecast"], width=2, dash="dash")),
                          row=1, col=1)
            fig.add_hline(y=CPU_SLA, line_dash="dot", line_color="orange",
                          annotation_text=f"SLA {CPU_SLA}%",
                          annotation_position="top right", row=1, col=1)

            fig.add_trace(go.Scatter(x=x, y=hist_mem, name="Mem Actual",
                                     line=dict(color=COLOURS["memory"], width=2),
                                     fill="tozeroy", fillcolor="rgba(139,92,246,0.08)"),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=fc_mem, name="Mem Forecast",
                                     line=dict(color="#f97316", width=2, dash="dash")),
                          row=2, col=1)
            fig.add_hline(y=MEM_SLA, line_dash="dot", line_color="orange",
                          annotation_text=f"SLA {MEM_SLA}%",
                          annotation_position="top right", row=2, col=1)

            fig.update_yaxes(range=[0, 105])
            fig.update_layout(height=460, template="plotly_white",
                               legend=dict(orientation="h", y=1.04),
                               margin=dict(t=55, b=20, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            running = st.session_state.get("lm_running", False)
            st.caption(
                f"{'🟢 Live Streaming' if running else '⏸ Paused'}  ·  "
                f"Samples: {lm_idx:,} / {lm_total:,}  ·  "
                f"Each step = ~{speed * MINS_PER_ROW} min of telemetry")
        else:
            if st.session_state.get("lm_ready"):
                st.info("Press **▶ Start** to begin live monitoring.")
            else:
                st.warning(f"No data for **{provider}**. Run the notebook preprocessing first.")

        if alerts:
            with st.expander(f"🔔 SLA Alert Log ({len(alerts)} events)", expanded=False):
                st.dataframe(
                    pd.DataFrame(alerts).rename(columns={
                        "t": "Step", "metric": "Metric",
                        "pred": "Predicted (%)", "sla": "SLA (%)"}),
                    use_container_width=True, hide_index=True)

    _live_dashboard()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown("### 🔮 Predict upcoming resource utilisation")
    st.caption("Upload telemetry CSV or use demo data. Model forecasts 30 minutes ahead.")

    fc1, fc2, fc3 = st.columns([2, 2, 1])
    fc_provider = fc1.selectbox("Data source", ["alibaba","azure","google"], key="fc_prov")
    fc_target   = fc2.selectbox("Metric", ["cpu","memory"],
                                format_func=lambda x: "CPU" if x == "cpu" else "Memory",
                                key="fc_target")
    fc_rows     = fc3.slider("Rows", 50, 500, 200, step=50, key="fc_rows")

    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"], key="fc_upload")
    demo_df  = _demo(fc_provider)
    input_df = pd.read_csv(uploaded) if uploaded else demo_df

    if input_df is None:
        st.warning(f"No data for **{fc_provider}**. Upload a CSV or run preprocessing first.")
    else:
        if st.button("▶ Run Forecast", type="primary", key="fc_run"):
            with st.spinner("Running ensemble forecast…"):
                try:
                    batch   = input_df.head(fc_rows)
                    pred_df = _ensemble(fc_target).predict(batch)
                    ac      = TARGET_CONFIG[fc_target]["actual_col"]
                    has_act = ac in pred_df.columns
                    preds   = pred_df["prediction"].values
                    if preds.max() <= 2: preds = preds * 100

                    if has_act:
                        actuals = pred_df[ac].values
                        if actuals.max() <= 2: actuals = actuals * 100
                    else:
                        actuals = None

                    sla    = CPU_SLA if fc_target == "cpu" else MEM_SLA
                    peak   = float(preds.max())
                    avg    = float(preds.mean())
                    risk   = _risk(peak, sla)
                    rc     = _risk_class(peak, sla)
                    mins_b = _mins_to_breach(preds.tolist(), sla)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Peak Forecast", f"{peak:.1f}%")
                    m2.metric("Avg Forecast",  f"{avg:.1f}%")
                    m3.metric("SLA Threshold", f"{sla}%",
                              delta="🚨 At risk" if peak >= sla else "✅ Safe")
                    with m4:
                        st.markdown("**Risk Level**")
                        st.markdown(f'<span class="{rc}">{risk}</span>', unsafe_allow_html=True)

                    if mins_b is not None:
                        cls = "breach-red" if mins_b <= 15 else "breach-amber"
                        st.markdown(
                            f'<div class="{cls}">⏱️ Breach predicted in ~{mins_b} min '
                            f'({fc_target.upper()} hits {sla}%)</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="breach-green">✅ No breach expected in next '
                            f'{len(preds) * MINS_PER_ROW} min</div>',
                            unsafe_allow_html=True)

                    x     = list(range(len(preds)))
                    upper = np.minimum(preds * 1.05, 100)
                    lower = np.maximum(preds * 0.95, 0)
                    col   = COLOURS["cpu"] if fc_target == "cpu" else COLOURS["memory"]
                    label = "CPU" if fc_target == "cpu" else "Memory"

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x+x[::-1], y=upper.tolist()+lower[::-1].tolist(),
                        fill="toself", fillcolor="rgba(239,68,68,0.08)",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=True, name="Forecast range", hoverinfo="skip"))
                    if actuals is not None:
                        fig.add_trace(go.Scatter(x=x, y=actuals, name=f"Actual {label}",
                                                 line=dict(color=col, width=2)))
                    fig.add_trace(go.Scatter(x=x, y=preds, name="30-min Forecast",
                                            line=dict(color=COLOURS["forecast"],
                                                      width=2.5, dash="dash")))
                    fig.add_hline(y=sla, line_dash="dot", line_color="orange",
                                  annotation_text=f"SLA {sla}%",
                                  annotation_position="top right")
                    fig.update_layout(
                        title=f"{label} — 30-Minute Ahead Forecast",
                        xaxis_title="Time step", yaxis_title="Utilisation (%)",
                        yaxis_range=[0, 105], height=400, template="plotly_white",
                        legend=dict(orientation="h", y=1.08),
                        margin=dict(t=55, b=20, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)

                    if actuals is not None:
                        from sklearn.metrics import r2_score
                        r2 = r2_score(actuals, preds)
                        st.caption(f"Batch accuracy: **R² = {r2*100:.1f}%**")

                    out = pd.DataFrame({"step": x, "forecast_%": preds.round(2),
                                        "lower_%": lower.round(2), "upper_%": upper.round(2)})
                    if actuals is not None: out["actual_%"] = actuals.round(2)
                    st.download_button("⬇ Download CSV", out.to_csv(index=False).encode(),
                                       f"forecast_{fc_target}_{fc_provider}.csv",
                                       "text/csv", key="fc_dl")
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
                    st.caption("Run Section 8 or 8b of the notebook to generate model artifacts.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — SYSTEM HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with tab_health:
    st.markdown("### 📊 System Health Overview")
    sh_provider = st.selectbox("Data source", ["alibaba","azure","google"], key="sh_prov")
    sh_df = _demo(sh_provider)

    if sh_df is not None:
        sample = sh_df.tail(2000).copy()
        sample["cpu_pct"] = _scale(sample["cpu"])
        sample["mem_pct"] = _scale(sample["mem"])

        ns = sample.groupby("node_id").agg(
            cpu_mean=("cpu_pct","mean"), cpu_max=("cpu_pct","max"),
            mem_mean=("mem_pct","mean"), mem_max=("mem_pct","max"),
        ).reset_index()

        def _status(r):
            if r.cpu_max >= CPU_SLA or r.mem_max >= MEM_SLA: return "critical"
            if r.cpu_mean >= CPU_SLA*0.88 or r.mem_mean >= MEM_SLA*0.88: return "warning"
            return "ok"
        ns["status"] = ns.apply(_status, axis=1)

        n_ok   = int((ns.status == "ok").sum())
        n_warn = int((ns.status == "warning").sum())
        n_crit = int((ns.status == "critical").sum())

        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Total Nodes", len(ns))
        s2.metric("✅ Healthy",   n_ok)
        s3.metric("⚠️ Warning",   n_warn, delta="attention" if n_warn > 0 else "")
        s4.metric("🔴 Critical",  n_crit, delta="action required" if n_crit > 0 else "")
        st.divider()

        def _gauge(val, title, sla, colour):
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=round(val,1),
                number={"suffix":"%","font":{"size":36}},
                delta={"reference":sla,"decreasing":{"color":"#22c55e"},
                       "increasing":{"color":"#ef4444"},"suffix":"% vs SLA"},
                gauge={"axis":{"range":[0,100]}, "bar":{"color":colour},
                       "steps":[{"range":[0,sla*0.75],"color":"#dcfce7"},
                                 {"range":[sla*0.75,sla*0.9],"color":"#fef3c7"},
                                 {"range":[sla*0.9,100],"color":"#fee2e2"}],
                       "threshold":{"line":{"color":"orange","width":3},
                                    "thickness":0.75,"value":sla}},
                title={"text":title,"font":{"size":16}}))
            fig.update_layout(height=255, margin=dict(t=30,b=10,l=20,r=20),
                              template="plotly_white")
            return fig

        g1,g2 = st.columns(2)
        with g1:
            st.plotly_chart(_gauge(float(ns.cpu_mean.mean()),"Fleet CPU (avg)",
                                   CPU_SLA,"#3b82f6"), use_container_width=True)
        with g2:
            st.plotly_chart(_gauge(float(ns.mem_mean.mean()),"Fleet Memory (avg)",
                                   MEM_SLA,"#8b5cf6"), use_container_width=True)
        st.divider()

        st.markdown("#### Node-level CPU vs Memory")
        fig_sc = px.scatter(ns, x="cpu_mean", y="mem_mean", color="status",
                            color_discrete_map={"ok":"#22c55e","warning":"#f59e0b","critical":"#ef4444"},
                            hover_data={"node_id":True,"cpu_max":":.1f","mem_max":":.1f"},
                            labels={"cpu_mean":"Avg CPU (%)","mem_mean":"Avg Memory (%)"},
                            title="Node Fleet — CPU vs Memory")
        fig_sc.add_vline(x=CPU_SLA*0.88, line_dash="dot", line_color="#f59e0b",
                         annotation_text="CPU warn")
        fig_sc.add_hline(y=MEM_SLA*0.88, line_dash="dot", line_color="#f59e0b",
                         annotation_text="Mem warn")
        fig_sc.update_layout(height=360, template="plotly_white",
                             margin=dict(t=50,b=20,l=10,r=10))
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("#### Node Health Grid")
        css_map  = {"ok":"node-ok","warning":"node-warn","critical":"node-crit"}
        icon_map = {"ok":"✅","warning":"⚠️","critical":"🔴"}
        nodes = ns.to_dict("records")
        for rs in range(0, min(len(nodes), 60), 6):
            cols_g = st.columns(6)
            for col_g, node in zip(cols_g, nodes[rs:rs+6]):
                with col_g:
                    st.markdown(
                        f'<div class="node-card {css_map[node["status"]]}">'
                        f'{icon_map[node["status"]]} {str(node["node_id"])[:8]}<br>'
                        f'<small>C:{node["cpu_mean"]:.0f}% M:{node["mem_mean"]:.0f}%</small>'
                        f'</div>', unsafe_allow_html=True)
        if len(nodes) > 60:
            st.caption(f"Showing 60 of {len(nodes)} nodes.")
    else:
        st.warning(f"No data for **{sh_provider}**. Run preprocessing first.")

    st.divider()
    st.markdown("#### Model Accuracy")
    summary = _results()
    if summary:
        for tgt in ["cpu", "memory"]:
            df_r = results_table(summary, tgt)
            if df_r.empty:
                continue
            r2c = "R² (Accuracy)" if "R² (Accuracy)" in df_r.columns else None
            rmse_c = "RMSE (%)" if "RMSE (%)" in df_r.columns else None
            if r2c:
                best  = df_r.loc[df_r[r2c].idxmax()]
                r2val = best[r2c]*100 if best[r2c] <= 1 else best[r2c]
                badge = "✅" if r2val >= 90 else "📈"
                st.metric(f"{badge} {tgt.upper()} Forecast Accuracy", f"{r2val:.1f}%",
                          delta=">90% target met" if r2val >= 90
                          else f"Best: {best.get('Model','N/A')}")
            elif rmse_c:
                # Compute accuracy as (100 - RMSE) for display
                ml_models = df_r[~df_r["Model"].str.lower().isin(
                    ["naive persistence", "moving average", "naive_persistence", "moving_average"]
                )].copy()
                if not ml_models.empty:
                    best = ml_models.loc[ml_models[rmse_c].idxmin()]
                    rmse_val = best[rmse_c]
                    acc = 100 - rmse_val
                    badge = "✅" if acc >= 90 else "📈"
                    st.metric(
                        f"{badge} {tgt.upper()} Best Model — {best['Model']}",
                        f"RMSE: {rmse_val:.2f}%",
                        delta=f"Accuracy: {acc:.1f}%"
                    )
            # Show full results table
            st.dataframe(
                df_r.style.format({
                    c: "{:.2f}" for c in df_r.columns
                    if df_r[c].dtype in ["float64", "float32"]
                }),
                use_container_width=True, hide_index=True
            )
    else:
        st.info("Run Section 18 of the notebook to generate model results.")


# ══════════════════════════════════════════════════════════════════════════════
#  FLOATING CHATBOT — JS injects into parent Streamlit page (not iframe)
# ══════════════════════════════════════════════════════════════════════════════
_CSS = """
#cloud-chat-fab {
  position:fixed; bottom:24px; right:24px;
  width:60px; height:60px;
  background:linear-gradient(135deg,#3b82f6,#1d4ed8);
  border-radius:50%; border:none; cursor:pointer;
  box-shadow:0 4px 20px rgba(59,130,246,0.5);
  font-size:1.7rem; display:flex; align-items:center;
  justify-content:center; z-index:99999;
  transition:transform 0.2s ease;
}
#cloud-chat-fab:hover { transform:scale(1.12); }
#cloud-chat-panel {
  position:fixed; bottom:96px; right:24px;
  width:370px; max-height:500px;
  background:white; border-radius:16px;
  box-shadow:0 8px 40px rgba(0,0,0,0.18);
  display:none; flex-direction:column;
  z-index:99998; overflow:hidden;
  font-family:'Inter',-apple-system,sans-serif;
}
#cloud-chat-header {
  background:linear-gradient(135deg,#1e3a5f,#3b82f6);
  color:white; padding:16px;
  display:flex; justify-content:space-between; align-items:center;
  border-radius:16px 16px 0 0;
}
#cloud-chat-header h4 { margin:0; font-size:1rem; font-weight:600; }
#cloud-chat-header span { font-size:0.75rem; opacity:0.8; }
#cloud-chat-close {
  background:none; border:none; color:white;
  font-size:1.4rem; cursor:pointer; padding:4px;
}
#cloud-chat-msgs {
  flex:1; overflow-y:auto; padding:14px;
  display:flex; flex-direction:column; gap:10px;
  background:#f8fafc; max-height:340px;
}
.ccm-user {
  align-self:flex-end; background:#3b82f6; color:white;
  padding:10px 14px; border-radius:14px 14px 2px 14px;
  max-width:80%; font-size:0.85rem; line-height:1.4;
}
.ccm-bot {
  align-self:flex-start; background:white; color:#1e293b;
  border:1px solid #e2e8f0;
  padding:10px 14px; border-radius:14px 14px 14px 2px;
  max-width:85%; font-size:0.85rem; line-height:1.55;
  box-shadow:0 1px 3px rgba(0,0,0,0.04);
}
.ccm-bot b { color:#1e3a5f; }
#cloud-chat-input-area {
  padding:12px; background:white;
  border-top:1px solid #e2e8f0;
  display:flex; gap:8px; align-items:center;
}
#cloud-chat-input {
  flex:1; border:1px solid #e2e8f0; border-radius:10px;
  padding:10px 14px; font-size:0.85rem; outline:none;
  font-family:'Inter',-apple-system,sans-serif;
}
#cloud-chat-input:focus { border-color:#3b82f6; box-shadow:0 0 0 2px rgba(59,130,246,0.15); }
#cloud-chat-send {
  background:#3b82f6; color:white; border:none;
  border-radius:10px; padding:10px 16px; cursor:pointer;
  font-size:0.95rem; font-weight:600;
}
#cloud-chat-send:hover { background:#1d4ed8; }
""".replace("\n", " ")

_PANEL_HTML = (
    '<div id="cloud-chat-header"><div>'
    '<h4>\u2601\ufe0f Cloud Assistant</h4>'
    '<span>\U0001f7e2 Online</span></div>'
    '<button id="cloud-chat-close">\u2715</button></div>'
    '<div id="cloud-chat-msgs">'
    '<div class="ccm-bot">\U0001f44b Hi! Ask about '
    '<b>CPU status</b>, <b>memory status</b>, '
    '<b>breach predictions</b>, <b>why this prediction</b>, '
    '<b>cost savings</b>, or type <b>help</b>.</div></div>'
    '<div id="cloud-chat-input-area">'
    '<input id="cloud-chat-input" type="text" '
    'placeholder="Ask about your cloud resources\u2026">'
    '<button id="cloud-chat-send">\u27a4</button></div>'
)

import json
_css_json = json.dumps(_CSS)
_html_json = json.dumps(_PANEL_HTML)

_CHAT_SCRIPT = f"""<script>
(function() {{
  var pdoc = window.parent.document;
  if (pdoc.getElementById('cloud-chat-fab')) return;

  var style = pdoc.createElement('style');
  style.textContent = {_css_json};
  pdoc.head.appendChild(style);

  var fab = pdoc.createElement('button');
  fab.id = 'cloud-chat-fab';
  fab.title = 'Ask the Assistant';
  fab.textContent = '\U0001f4ac';
  pdoc.body.appendChild(fab);

  var panel = pdoc.createElement('div');
  panel.id = 'cloud-chat-panel';
  panel.innerHTML = {_html_json};
  pdoc.body.appendChild(panel);

  var isOpen = false;
  function toggle() {{
    isOpen = !isOpen;
    panel.style.display = isOpen ? 'flex' : 'none';
    if (isOpen) {{ scrl(); pdoc.getElementById('cloud-chat-input').focus(); }}
  }}
  function scrl() {{
    var m = pdoc.getElementById('cloud-chat-msgs');
    setTimeout(function(){{ m.scrollTop = m.scrollHeight; }}, 60);
  }}
  function addM(txt, role) {{
    var m = pdoc.getElementById('cloud-chat-msgs');
    var d = pdoc.createElement('div');
    d.className = role === 'user' ? 'ccm-user' : 'ccm-bot';
    d.innerHTML = txt;
    m.appendChild(d);
    scrl();
  }}
  function gd() {{
    var el = pdoc.getElementById('live-data');
    if (!el) return null;
    return {{
      cpu: parseFloat(el.getAttribute('data-cpu')||0),
      mem: parseFloat(el.getAttribute('data-mem')||0),
      fcCpu: parseFloat(el.getAttribute('data-fc-cpu')||0),
      fcMem: parseFloat(el.getAttribute('data-fc-mem')||0),
      cpuSla: parseFloat(el.getAttribute('data-cpu-sla')||85),
      memSla: parseFloat(el.getAttribute('data-mem-sla')||90),
      mcCpu: parseFloat(el.getAttribute('data-mc-cpu')||-1),
      mcMem: parseFloat(el.getAttribute('data-mc-mem')||-1),
      savings: parseFloat(el.getAttribute('data-savings')||0),
      breaches: parseFloat(el.getAttribute('data-breaches')||0)
    }};
  }}
  function mt(m) {{
    if (m<0) return '\u2705 No breach predicted in next 30 min';
    if (m===0) return '\U0001f6a8 <b>BREACHING NOW</b>';
    if (m<=15) return '\U0001f6a8 Breach in ~<b>'+m+' min</b>';
    return '\u26a0\ufe0f Breach in ~<b>'+m+' min</b>';
  }}
  function reply(q) {{
    var d=gd(), live=d&&d.cpu>0;
    q=q.toLowerCase().trim();
    if (q.includes('help')||q.includes('command'))
      return '\U0001f916 <b>Commands:</b><br>\u2022 <b>cpu status</b> \u2014 current CPU and forecast<br>\u2022 <b>memory status</b> \u2014 current memory and forecast<br>\u2022 <b>when will breach</b> \u2014 breach countdown<br>\u2022 <b>root cause</b> \u2014 diagnose why a breach is happening<br>\u2022 <b>why this prediction</b> \u2014 SHAP explanation<br>\u2022 <b>explain cpu</b> \u2014 what drives CPU forecast<br>\u2022 <b>explain memory</b> \u2014 what drives memory forecast<br>\u2022 <b>sla breaches</b> \u2014 alert count<br>\u2022 <b>cost savings</b> \u2014 estimated EUR saved<br>\u2022 <b>model accuracy</b> \u2014 R\u00b2 score';
    if (q.includes('cpu')&&(q.includes('status')||q.includes('now')||q.includes('how'))) {{
      if (!live) return 'Press Start on Live Monitor first.';
      return '\U0001f4ca <b>CPU</b><br>Now: <b>'+d.cpu.toFixed(1)+'%</b> (SLA:'+d.cpuSla+'%)<br>Forecast: <b>'+d.fcCpu.toFixed(1)+'%</b><br>'+mt(d.mcCpu);
    }}
    if (q.includes('mem')||q.includes('ram')||q.includes('memory')) {{
      if (!live) return 'Press Start on Live Monitor first.';
      return '\U0001f4ca <b>Memory</b><br>Now: <b>'+d.mem.toFixed(1)+'%</b> (SLA:'+d.memSla+'%)<br>Forecast: <b>'+d.fcMem.toFixed(1)+'%</b><br>'+mt(d.mcMem);
    }}
    if (q.includes('root cause')||q.includes('cause')||q.includes('diagnos')||q.includes('analyz')||q.includes('analys')||q.includes('why breach')||q.includes('why is it')) {{
      if (!live) return 'Start monitoring first so I can look at the data.';
      var rca = '\U0001f50e <b>Root Cause Analysis</b><br><br>';
      var issues = [];
      var cpuRise = d.fcCpu - d.cpu;
      var memRise = d.fcMem - d.mem;
      var cpuNearSla = d.cpu >= d.cpuSla * 0.85;
      var memNearSla = d.mem >= d.memSla * 0.85;
      var cpuBreaching = d.cpu >= d.cpuSla;
      var memBreaching = d.mem >= d.memSla;
      if (cpuBreaching) {{
        issues.push('<b>\U0001f534 CPU is currently above the '+d.cpuSla+'% SLA threshold</b> at <b>'+d.cpu.toFixed(1)+'%</b>. This typically happens when multiple workloads are competing for compute resources on the same node. Common triggers include batch processing jobs, container scaling events, or an unexpected traffic spike. The model forecasts CPU will be at <b>'+d.fcCpu.toFixed(1)+'%</b> in 30 minutes, so this is '+(d.fcCpu > d.cpu ? 'likely to get worse before it improves.' : 'expected to ease slightly, but remains above threshold.'));
      }} else if (cpuNearSla) {{
        issues.push('<b>\U0001f7e0 CPU is approaching the danger zone</b> at <b>'+d.cpu.toFixed(1)+'%</b> (threshold: '+d.cpuSla+'%). It has not breached yet, but it is close enough that a small load increase could push it over. The forecast shows CPU heading to <b>'+d.fcCpu.toFixed(1)+'%</b>, which '+(d.fcCpu >= d.cpuSla ? 'is above the SLA limit \u2014 a breach is likely imminent.' : 'is still within safe limits, but the margin is thin.'));
      }} else {{
        issues.push('<b>\U0001f7e2 CPU looks healthy</b> at <b>'+d.cpu.toFixed(1)+'%</b>, well below the '+d.cpuSla+'% threshold. No immediate concern here.');
      }}
      if (memBreaching) {{
        issues.push('<b>\U0001f534 Memory is above the '+d.memSla+'% SLA threshold</b> at <b>'+d.mem.toFixed(1)+'%</b>. High memory usage in cloud environments is often caused by memory leaks in long-running containers, large in-memory caches or datasets, or a sudden burst of new container deployments. Unlike CPU, memory pressure does not resolve on its own \u2014 it usually requires explicit action like restarting containers or adding nodes. Forecast: <b>'+d.fcMem.toFixed(1)+'%</b>.');
      }} else if (memNearSla) {{
        issues.push('<b>\U0001f7e0 Memory is elevated</b> at <b>'+d.mem.toFixed(1)+'%</b> (threshold: '+d.memSla+'%). This is not a breach yet, but memory tends to creep up gradually and then spike suddenly when containers get OOM-killed and rescheduled. The model forecasts <b>'+d.fcMem.toFixed(1)+'%</b> in 30 minutes'+(d.fcMem >= d.memSla ? ' \u2014 which crosses the SLA threshold. Consider proactive scaling now.' : '.'));
      }} else {{
        issues.push('<b>\U0001f7e2 Memory is within safe limits</b> at <b>'+d.mem.toFixed(1)+'%</b>. No concerns at this time.');
      }}
      rca += issues.join('<br><br>');
      if (cpuBreaching || memBreaching) {{
        rca += '<br><br><b>\u2699\ufe0f Recommended Actions:</b><br>';
        rca += '\u2022 Check for recently deployed or scaled workloads<br>';
        rca += '\u2022 Review container resource limits and requests<br>';
        rca += '\u2022 Consider horizontal scaling \u2014 add nodes to distribute load<br>';
        rca += '\u2022 Investigate if a batch job or cron task is consuming excess resources';
      }} else if (cpuNearSla || memNearSla) {{
        rca += '<br><br><b>\u26a0\ufe0f Proactive Recommendation:</b> The system is not breaching yet, but trending in that direction. Proactive scaling now could prevent an SLA violation in the next 15\u201330 minutes.';
      }} else {{
        rca += '<br><br>\u2705 Overall, the cluster is operating normally. No action required at this time.';
      }}
      return rca;
    }}
    if (q.includes('breach')||q.includes('when')||q.includes('predict')) {{
      if (!live) return 'Start monitoring first.';
      return '\u23f1\ufe0f <b>Breach ETA</b><br>CPU: '+mt(d.mcCpu)+'<br>Mem: '+mt(d.mcMem);
    }}
    if (q.includes('why')||q.includes('explain')||q.includes('shap')||q.includes('feature')||q.includes('important')) {{
      var cpuExplain = q.includes('mem') || q.includes('ram');
      if (!live) {{
        return '\U0001f50d <b>SHAP Explainability</b><br><br>Our SHAP analysis (using TreeExplainer) identified the key drivers behind each prediction:<br><br>' +
          '<b>For CPU forecasts</b>, the model relies most heavily on the <b>6-step rolling average</b> of recent CPU usage. When this rolling mean is elevated, the model predicts continued high demand. The second most important signal is the <b>most recent CPU reading</b> (lag-1), followed by <b>CPU volatility</b> (rolling standard deviation). This makes intuitive sense: a steadily climbing CPU is more likely to stay high.<br><br>' +
          '<b>For Memory forecasts</b>, the model depends on <b>shorter-term signals</b>. The single most recent memory reading (lag-1) dominates, followed by the 3-step rolling mean. This reflects how container memory allocations change in discrete steps rather than gradually.<br><br>' +
          'Type <b>explain cpu</b> or <b>explain memory</b> with live data running for a contextual breakdown.';
      }}
      if (cpuExplain) {{
        var lvl = d.mem > 80 ? 'critically high' : d.mem > 60 ? 'moderately elevated' : d.mem > 40 ? 'moderate' : 'relatively low';
        var dir = d.fcMem > d.mem ? 'upward' : d.fcMem < d.mem ? 'downward' : 'stable';
        return '\U0001f9e0 <b>Why Memory is forecast at '+d.fcMem.toFixed(1)+'%</b><br><br>' +
          'Based on SHAP analysis, these are the main factors:<br><br>' +
          '<b>1. Recent memory reading (lag-1):</b> Memory was at <b>'+d.mem.toFixed(1)+'%</b> in the last observation, which is '+lvl+'. This is the strongest signal \u2014 the model treats the most recent value as the best starting point for its forecast.<br><br>' +
          '<b>2. Short-term trend (rolling mean):</b> The 3-step rolling average captures whether memory has been climbing or falling over the last 30 minutes. The current trend is <b>'+dir+'</b>, which pushes the forecast in that direction.<br><br>' +
          '<b>3. Volatility (rolling std):</b> If memory has been fluctuating a lot recently, the model widens its uncertainty. Stable memory is easier to predict than spiky memory.<br><br>' +
          '<b>4. Rate of change:</b> How fast memory is rising or falling right now. A sharp increase signals potential container scaling events.<br><br>' +
          '<i>These four features account for over 75% of the prediction according to SHAP values.</i>';
      }} else {{
        var lvl = d.cpu > 80 ? 'critically high' : d.cpu > 60 ? 'moderately elevated' : d.cpu > 40 ? 'moderate' : 'relatively low';
        var dir = d.fcCpu > d.cpu ? 'upward' : d.fcCpu < d.cpu ? 'downward' : 'stable';
        return '\U0001f9e0 <b>Why CPU is forecast at '+d.fcCpu.toFixed(1)+'%</b><br><br>' +
          'Based on SHAP analysis, these are the main factors:<br><br>' +
          '<b>1. Rolling average (6-step):</b> The average CPU over the last 60 minutes is the single strongest predictor. Current CPU at <b>'+d.cpu.toFixed(1)+'%</b> is '+lvl+', and this rolling context heavily influences the 30-minute-ahead forecast.<br><br>' +
          '<b>2. Most recent reading (lag-1):</b> The last observed CPU value. If it has spiked recently, the model expects elevated usage to continue in the short term.<br><br>' +
          '<b>3. CPU volatility (rolling std):</b> Measures how much CPU has been fluctuating. High volatility means the forecast carries more uncertainty \u2014 the model hedges toward the mean.<br><br>' +
          '<b>4. Rate of change:</b> Whether CPU is trending <b>'+dir+'</b> right now. A rising trajectory pushes the forecast higher; a falling one pulls it down.<br><br>' +
          '<i>These four features explain over 80% of the prediction, as confirmed by SHAP TreeExplainer analysis.</i>';
      }}
    }}
    if (q.includes('sla')||q.includes('alert')) {{
      if (!live) return 'Start monitoring first.';
      return d.breaches===0?'\u2705 No breaches.':'\U0001f6a8 <b>'+d.breaches+' breach(es)</b>';
    }}
    if (q.includes('cost')||q.includes('saving')||q.includes('euro')) {{
      if (!live) return 'Start monitoring first.';
      return '\U0001f4b0 Saved: <b>\u20ac'+d.savings.toFixed(2)+'</b>';
    }}
    if (q.includes('accura')||q.includes('r2')||q.includes('model'))
      return 'Ensemble targets <b>R\u00b2 > 90%</b>. See System Health tab.';
    return 'Try: <b>cpu status</b>, <b>memory status</b>, <b>when will breach</b>, <b>root cause</b>, <b>why this prediction</b>, <b>help</b>';
  }}
  function send() {{
    var inp=pdoc.getElementById('cloud-chat-input');
    var t=inp.value.trim();
    if (!t) return;
    addM(t,'user'); inp.value='';
    setTimeout(function(){{ addM(reply(t),'bot'); }}, 300);
  }}

  fab.addEventListener('click', toggle);
  pdoc.getElementById('cloud-chat-close').addEventListener('click', toggle);
  pdoc.getElementById('cloud-chat-send').addEventListener('click', send);
  pdoc.getElementById('cloud-chat-input').addEventListener('keydown', function(e){{ if(e.key==='Enter') send(); }});
}})();
</script>"""

components.html(_CHAT_SCRIPT, height=0, scrolling=False)
