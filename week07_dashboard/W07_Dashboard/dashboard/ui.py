"""Visual language and persona guidance for the Week 7 dashboard."""

from __future__ import annotations

import streamlit as st


PERSONA_DESCRIPTIONS = {
    "Engineering Manager": (
        "Prioritizes detector quality, seed stability, failure modes, and "
        "precision/recall tradeoffs."
    ),
    "Product Manager": (
        "Prioritizes trends, algorithm and platform comparisons, adoption "
        "readiness, and tradeoffs between quality and cost."
    ),
    "Customer Success": (
        "Prioritizes unit-level status, alert severity, operational context, "
        "and the next recommended action."
    ),
}

PERSONA_CONTROLS = {
    "Engineering Manager": [
        "Anomaly method and fault-scope filters",
        "Precision, recall, F1, AUROC, and split-stability views",
        "Environment, algorithm, and seed controls",
        "Domain reliability and scenario-disagreement drill-down",
    ],
    "Product Manager": [
        "Cross-platform and algorithm comparison charts",
        "Learning-curve, wall-clock, and threshold-success controls",
        "Product-domain scorecards and reliability comparison",
        "Executive summaries and recommendation callouts",
    ],
    "Customer Success": [
        "Fleet status and alert-severity filters",
        "Unit selector with battery, connectivity, and task history",
        "Alert-history table and recommended next action",
        "High-risk scenario explorer for customer-facing escalation context",
    ],
}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --panel: rgba(255, 255, 255, 0.88);
          --line: rgba(15, 23, 42, 0.10);
          --ink: #0f172a;
          --muted: #526174;
        }
        .stApp {
          background:
            radial-gradient(circle at 8% 8%, rgba(30, 64, 175, 0.07), transparent 28rem),
            radial-gradient(circle at 92% 6%, rgba(13, 148, 136, 0.06), transparent 24rem),
            #f7f9fc;
        }
        .block-container {
          max-width: 1480px;
          padding-top: 1.35rem;
          padding-bottom: 3rem;
        }
        h1, h2, h3 {
          letter-spacing: -0.025em;
          color: var(--ink);
        }
        [data-testid="stMetric"] {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: 0.85rem 1rem;
          box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
        }
        [data-testid="stMetricLabel"] {
          color: var(--muted);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
          background: var(--panel);
          border-color: var(--line);
          border-radius: 16px;
        }
        .hero {
          padding: 1.25rem 1.35rem;
          border: 1px solid var(--line);
          border-radius: 18px;
          background: linear-gradient(135deg, rgba(255,255,255,.96), rgba(239,246,255,.88));
          box-shadow: 0 14px 36px rgba(15, 23, 42, 0.055);
          margin-bottom: 0.8rem;
        }
        .eyebrow {
          color: #475569;
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.11em;
          text-transform: uppercase;
        }
        .hero-title {
          color: #0f172a;
          font-size: 2rem;
          font-weight: 760;
          line-height: 1.08;
          margin-top: 0.25rem;
        }
        .hero-copy {
          color: #526174;
          max-width: 960px;
          margin-top: 0.45rem;
        }
        .persona-note {
          border-left: 4px solid #334155;
          background: rgba(255,255,255,.75);
          border-radius: 0 12px 12px 0;
          padding: .75rem 1rem;
          color: #334155;
          margin: .35rem 0 1rem 0;
        }
        .status-healthy {font-weight:700; color:#047857;}
        .status-warning {font-weight:700; color:#b45309;}
        .status-critical {font-weight:700; color:#b91c1c;}
        .small-note {color:#64748b; font-size:.86rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Analyst Workspace</div>
          <div class="hero-title">InGen Fleet, Policy & Decision Intelligence</div>
          <div class="hero-copy">
            One dashboard for fleet-health monitoring, reinforcement-learning
            policy comparison, and decision-evaluation reliability.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_persona_context(persona: str, view: str) -> None:
    st.markdown(
        f"""
        <div class="persona-note">
          <strong>{persona} lens · {view}</strong><br>
          {PERSONA_DESCRIPTIONS[persona]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_control_map(persona: str) -> None:
    with st.expander("Control surface for this persona"):
        st.write(PERSONA_DESCRIPTIONS[persona])
        for item in PERSONA_CONTROLS[persona]:
            st.markdown(f"- {item}")
