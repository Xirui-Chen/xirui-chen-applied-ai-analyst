from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.charts import (
    PLOTLY_CONFIG,
    anomaly_heatmap,
    anomaly_metric_bars,
    anomaly_split_box,
    exact_match_verdict,
    fleet_metric_line,
    learning_curve,
    multi_agent_comparison,
    policy_tradeoff,
    reliability_bars,
    reward_by_seed,
    verdict_distribution,
)
from dashboard.data import load_dashboard_data
from dashboard.ui import (
    PERSONA_DESCRIPTIONS,
    apply_theme,
    render_control_map,
    render_hero,
    render_persona_context,
)

st.set_page_config(
    page_title="InGen Fleet, Policy & Decision Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
data = load_dashboard_data()
render_hero()

WORKSPACES = [
    "Fleet Health",
    "Policy Benchmark",
    "Decision Evaluation",
]

PERSONA_DEFAULT_VIEW = {
    "Engineering Manager": "Fleet Health",
    "Product Manager": "Policy Benchmark",
    "Customer Success": "Fleet Health",
}

if "persona" not in st.session_state:
    st.session_state["persona"] = "Engineering Manager"

if "workspace" not in st.session_state:
    st.session_state["workspace"] = PERSONA_DEFAULT_VIEW[
        st.session_state["persona"]
    ]

if "_last_persona" not in st.session_state:
    st.session_state["_last_persona"] = st.session_state["persona"]

with st.sidebar:
    st.subheader("Audience")
    persona = st.selectbox(
        "Persona",
        ["Engineering Manager", "Product Manager", "Customer Success"],
        key="persona",
        help=(
            "Changes the recommended starting workspace, ordering, emphasis, "
            "and operational guidance."
        ),
    )
    st.caption(PERSONA_DESCRIPTIONS[persona])
    st.divider()
    st.subheader("Data status")
    st.success("Weeks 3–6 sample data loaded")
    st.caption(
        "All fleet telemetry, alerts, and operational examples are synthetic. "
        "No real customer or production data is included."
    )
    manifest = data["manifest"]
    st.metric("Sample data files", len(manifest["files"]))
    st.metric(
        "Seeded decision judgments",
        len(data["decision_judgments"]),
    )
    render_control_map(persona)

if st.session_state["_last_persona"] != persona:
    st.session_state["workspace"] = PERSONA_DEFAULT_VIEW[persona]
    st.session_state["_last_persona"] = persona

view = st.radio(
    "Workspace",
    WORKSPACES,
    horizontal=True,
    label_visibility="collapsed",
    key="workspace",
)

st.caption(
    f"Recommended starting workspace for {persona}: "
    f"{PERSONA_DEFAULT_VIEW[persona]}. "
    "You can still open any workspace manually."
)

render_persona_context(persona, view)

DOMAIN_LABELS = {
    "aido_rover": "Aido Rover",
    "sentinel_prime": "Sentinel Prime",
    "senpai": "Senpai",
    "fari": "Fari",
}


def render_fleet_status_panel() -> None:
    status = data["fleet_status"].copy()

    with st.container(border=True):
        st.subheader("Fleet operations")
        control_a, control_b = st.columns(2)
        with control_a:
            status_filter = st.multiselect(
                "Unit status",
                ["Healthy", "Warning", "Critical"],
                default=["Healthy", "Warning", "Critical"],
                key="fleet_status_filter",
            )
        with control_b:
            severity_floor = st.slider(
                "Minimum alert severity score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="fleet_severity_floor",
                help="Healthy units retain a low synthetic monitoring score.",
            )

        filtered = status[
            status["status"].isin(status_filter)
            & (status["alert_severity_score"] >= severity_floor)
        ].copy()

        display = filtered[
            [
                "unit_id",
                "status",
                "battery_soc_pct",
                "mission_mode",
                "location_zone",
                "active_alert",
                "alert_severity_score",
                "recommended_action",
            ]
        ].rename(
            columns={
                "unit_id": "Unit",
                "status": "Status",
                "battery_soc_pct": "Battery",
                "mission_mode": "Mission",
                "location_zone": "Zone",
                "active_alert": "Active alert",
                "alert_severity_score": "Alert score",
                "recommended_action": "Recommended action",
            }
        )

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Battery": st.column_config.ProgressColumn(
                    "Battery",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
                "Alert score": st.column_config.ProgressColumn(
                    "Alert score",
                    min_value=0,
                    max_value=1,
                    format="%.3f",
                ),
            },
        )
        st.caption(
            "Customer-success snapshot derived from synthetic Week 3 telemetry "
            "and Week 4 fault examples."
        )


def render_unit_drilldown() -> None:
    status = data["fleet_status"]
    telemetry = data["fleet_telemetry"]
    alerts = data["alert_history"]

    with st.container(border=True):
        st.subheader("Unit drill-down")
        unit_id = st.selectbox(
            "Fleet unit",
            sorted(status["unit_id"].unique()),
            key="fleet_unit_selector",
        )
        row = status[status["unit_id"] == unit_id].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status", row["status"])
        m2.metric("Battery", f"{row['battery_soc_pct']:.1f}%")
        m3.metric("Alert", str(row["active_alert"]).replace("_", " ").title())
        m4.metric("Task success", f"{row['task_success_probability']:.1%}")

        st.info(f"Recommended action: {row['recommended_action']}")

        power_tab, connectivity_tab, task_tab = st.tabs(
            ["Power & thermal", "Connectivity", "Task execution"]
        )
        with power_tab:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    fleet_metric_line(
                        telemetry,
                        unit_id,
                        "battery_soc_pct",
                        f"{unit_id} battery state",
                        "Battery SoC (%)",
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
            with c2:
                st.plotly_chart(
                    fleet_metric_line(
                        telemetry,
                        unit_id,
                        "motor_temp_c",
                        f"{unit_id} motor temperature",
                        "Temperature (°C)",
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
        with connectivity_tab:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    fleet_metric_line(
                        telemetry,
                        unit_id,
                        "wifi_rssi_dbm",
                        f"{unit_id} Wi-Fi signal",
                        "RSSI (dBm)",
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
            with c2:
                st.plotly_chart(
                    fleet_metric_line(
                        telemetry,
                        unit_id,
                        "gps_hdop",
                        f"{unit_id} GPS dilution",
                        "GPS HDOP",
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
        with task_tab:
            st.plotly_chart(
                fleet_metric_line(
                    telemetry,
                    unit_id,
                    "task_success_probability",
                    f"{unit_id} task-success probability",
                    "Probability",
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        unit_alerts = (
            alerts[alerts["robot_id"] == unit_id]
            .sort_values("severity", ascending=False)
            .head(12)
            .copy()
        )
        st.markdown("**Recent benchmark alert examples**")
        st.dataframe(
            unit_alerts[
                [
                    "event_time",
                    "fault_type",
                    "severity_label",
                    "severity",
                    "duration_s",
                    "notes",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "severity": st.column_config.ProgressColumn(
                    "Severity",
                    min_value=0,
                    max_value=1,
                    format="%.3f",
                )
            },
        )


def render_anomaly_benchmark() -> None:
    summary = data["anomaly_summary"]
    per_split = data["anomaly_per_split"]

    with st.container(border=True):
        st.subheader("Anomaly detector benchmark")
        scopes = sorted(summary["scope"].unique())
        default_index = scopes.index("all_faults") if "all_faults" in scopes else 0
        scope = st.selectbox(
            "Fault scope",
            scopes,
            index=default_index,
            format_func=lambda value: value.replace("_", " ").title(),
            key="anomaly_scope",
        )
        selected = summary[summary["scope"] == scope].copy()
        best = selected.sort_values("f1_mean", ascending=False).iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Best detector", best["method"])
        k2.metric("Precision", f"{best['precision_mean']:.3f}")
        k3.metric("Recall", f"{best['recall_mean']:.3f}")
        k4.metric("AUROC", f"{best['auroc_mean']:.3f}")

        c1, c2 = st.columns([1.25, 1])
        with c1:
            st.plotly_chart(
                anomaly_metric_bars(summary, scope),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        with c2:
            st.plotly_chart(
                anomaly_split_box(per_split, scope),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        st.plotly_chart(
            anomaly_heatmap(summary),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )


def fleet_health_view() -> None:
    status = data["fleet_status"]
    anomaly = data["anomaly_summary"]
    overall = anomaly[anomaly["scope"] == "all_faults"].sort_values(
        "f1_mean", ascending=False
    )
    best = overall.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fleet units", len(status))
    c2.metric(
        "Units needing attention",
        int(status["status"].isin(["Warning", "Critical"]).sum()),
    )
    c3.metric("Best anomaly F1", f"{best['f1_mean']:.3f}", best["method"])
    c4.metric("Best anomaly AUROC", f"{best['auroc_mean']:.3f}")

    if persona == "Engineering Manager":
        render_anomaly_benchmark()
        render_fleet_status_panel()
        render_unit_drilldown()
    elif persona == "Customer Success":
        render_fleet_status_panel()
        render_unit_drilldown()
        render_anomaly_benchmark()
    else:
        render_anomaly_benchmark()
        render_fleet_status_panel()
        render_unit_drilldown()


def policy_benchmark_view() -> None:
    summary = data["rl_summary"]
    seed_results = data["rl_seed"]
    learning = data["rl_learning"]
    multi = data["rl_multi_diag"]

    with st.container(border=True):
        st.subheader("Policy controls")
        environments = sorted(summary["environment"].unique())
        environment = st.selectbox(
            "Environment",
            environments,
            format_func=lambda value: value.replace("Continuous", ""),
        )
        algorithms = st.multiselect(
            "Algorithms",
            ["PPO", "SAC"],
            default=["PPO", "SAC"],
        )
        available_seeds = sorted(
            data["rl_learning"]["seed"].astype(int).unique().tolist()
        )
        seeds = st.multiselect(
            "Training seeds",
            available_seeds,
            default=available_seeds,
        )
        if not algorithms or not seeds:
            st.warning("Select at least one algorithm and one seed.")
            st.stop()

    selected_summary = summary[
        (summary["environment"] == environment)
        & (summary["algorithm"].isin(algorithms))
    ].copy()

    best_reward = selected_summary.sort_values(
        "final_reward_mean", ascending=False
    ).iloc[0]
    fastest = selected_summary.sort_values("wallclock_sec_mean").iloc[0]
    most_stable = selected_summary.sort_values("convergence_stability_cv").iloc[0]
    best_threshold = selected_summary.sort_values(
        "threshold_success_rate", ascending=False
    ).iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Best final reward",
        f"{best_reward['final_reward_mean']:.1f}",
        best_reward["algorithm"],
    )
    k2.metric(
        "Fastest wall-clock",
        f"{fastest['wallclock_sec_mean']:.1f}s",
        fastest["algorithm"],
    )
    k3.metric(
        "Best stability CV",
        f"{most_stable['convergence_stability_cv']:.3f}",
        most_stable["algorithm"],
    )
    k4.metric(
        "Threshold success",
        f"{best_threshold['threshold_success_rate']:.0%}",
        best_threshold["algorithm"],
    )

    st.plotly_chart(
        learning_curve(learning, environment, algorithms, seeds),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            reward_by_seed(seed_results, environment, algorithms, seeds),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with c2:
        st.plotly_chart(
            policy_tradeoff(summary),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with st.container(border=True):
        st.subheader("Multi-agent operational diagnostics")
        ppo = multi[multi["policy"] == "ppo"].iloc[0]
        random = multi[multi["policy"] == "random"].iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Mean agent return",
            f"{ppo['mean_agent_return_mean_across_seed_mean']:.2f}",
            f"{ppo['mean_agent_return_mean_across_seed_mean'] - random['mean_agent_return_mean_across_seed_mean']:+.2f} vs random",
        )
        m2.metric(
            "Assignment distance",
            f"{ppo['final_assignment_distance_mean_across_seed_mean']:.3f}",
            f"{ppo['final_assignment_distance_mean_across_seed_mean'] - random['final_assignment_distance_mean_across_seed_mean']:+.3f} vs random",
            delta_color="inverse",
        )
        m3.metric(
            "Collision pairs / cycle",
            f"{ppo['collision_pairs_per_cycle_mean_across_seed_mean']:.3f}",
            f"{ppo['collision_pairs_per_cycle_mean_across_seed_mean'] - random['collision_pairs_per_cycle_mean_across_seed_mean']:+.3f} vs random",
            delta_color="inverse",
        )
        st.plotly_chart(
            multi_agent_comparison(multi),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
        st.caption(
            "PPO improved return and assignment distance directionally, but "
            "collision rate increased and coordination success remained 0%."
        )


def decision_evaluation_view() -> None:
    reliability = data["decision_reliability"]
    domain = data["decision_domain"]
    scenarios = data["decision_scenarios"]
    judgments = data["decision_judgments"]

    overall = reliability[reliability["scope"] == "overall"].iloc[0]
    rule_accuracy = scenarios["exact_match"].astype(bool).mean()
    pass_rate_exact = (
        judgments[judgments["exact_match"].astype(bool)]["verdict"].eq("PASS").mean()
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rule exact-match accuracy", f"{rule_accuracy:.0%}")
    k2.metric(
        "Overall judge alpha",
        f"{overall['krippendorff_alpha_nominal']:.3f}",
    )
    k3.metric(
        "Three-run agreement",
        f"{overall['exact_three_run_agreement']:.0%}",
    )
    k4.metric("Exact-match PASS recall", f"{pass_rate_exact:.1%}")

    with st.container(border=True):
        st.subheader("Scorecard filters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            domain_filter = st.multiselect(
                "Product domain",
                sorted(scenarios["domain"].unique()),
                default=sorted(scenarios["domain"].unique()),
                format_func=lambda value: DOMAIN_LABELS.get(value, value),
            )
        with c2:
            risk_filter = st.multiselect(
                "Risk level",
                sorted(scenarios["risk_level"].unique()),
                default=sorted(scenarios["risk_level"].unique()),
            )
        with c3:
            difficulty_filter = st.multiselect(
                "Difficulty",
                sorted(scenarios["difficulty"].unique()),
                default=sorted(scenarios["difficulty"].unique()),
            )
        with c4:
            verdict_filter = st.multiselect(
                "Majority verdict",
                ["PASS", "PARTIAL", "FAIL"],
                default=["PASS", "PARTIAL", "FAIL"],
            )

    filtered = scenarios[
        scenarios["domain"].isin(domain_filter)
        & scenarios["risk_level"].isin(risk_filter)
        & scenarios["difficulty"].isin(difficulty_filter)
        & scenarios["judge_majority_verdict"].isin(verdict_filter)
    ].copy()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            reliability_bars(reliability),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with c2:
        st.plotly_chart(
            verdict_distribution(judgments),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            exact_match_verdict(judgments),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with c2:
        scorecard = domain.merge(
            reliability[reliability["scope"] == "domain"][
                [
                    "domain",
                    "exact_three_run_agreement",
                    "krippendorff_alpha_nominal",
                ]
            ],
            on="domain",
            how="left",
        ).copy()
        scorecard["domain"] = scorecard["domain"].map(DOMAIN_LABELS)
        scorecard = scorecard.rename(
            columns={
                "domain": "Domain",
                "scenario_count": "Scenarios",
                "rule_exact_accuracy": "Rule accuracy",
                "judge_mean_score": "Mean judge score",
                "exact_three_run_agreement": "3-run agreement",
                "krippendorff_alpha_nominal": "Alpha",
            }
        )
        st.markdown("#### Domain scorecard")
        st.dataframe(
            scorecard[
                [
                    "Domain",
                    "Scenarios",
                    "Rule accuracy",
                    "Mean judge score",
                    "3-run agreement",
                    "Alpha",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Rule accuracy": st.column_config.NumberColumn(format="percent"),
                "Mean judge score": st.column_config.NumberColumn(format="%.3f"),
                "3-run agreement": st.column_config.NumberColumn(format="percent"),
                "Alpha": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption(
            "Mean judge score is ordinal (FAIL=0, PARTIAL=1, PASS=2), not accuracy."
        )

    pivot = judgments.pivot(
        index="scenario_id",
        columns="judge_seed",
        values="verdict",
    )
    disagreement_ids = pivot[pivot.nunique(axis=1) > 1].index.tolist()

    with st.container(border=True):
        st.subheader("Scenario-level review")
        st.caption(
            f"{len(disagreement_ids)} scenarios show seed disagreement. "
            "These are the highest-priority cases for human review."
        )
        review = filtered.copy()
        review["seed_disagreement"] = review["scenario_id"].isin(disagreement_ids)
        review["domain"] = review["domain"].map(DOMAIN_LABELS)
        st.dataframe(
            review[
                [
                    "scenario_id",
                    "domain",
                    "risk_level",
                    "difficulty",
                    "candidate_action",
                    "target_action",
                    "exact_match",
                    "judge_majority_verdict",
                    "judge_mean_score",
                    "seed_disagreement",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "exact_match": st.column_config.CheckboxColumn("Exact match"),
                "seed_disagreement": st.column_config.CheckboxColumn(
                    "Seed disagreement"
                ),
                "judge_mean_score": st.column_config.NumberColumn(format="%.3f"),
            },
        )

        available_ids = filtered["scenario_id"].tolist()
        if available_ids:
            scenario_id = st.selectbox(
                "Scenario explorer",
                available_ids,
            )
            scenario = scenarios[scenarios["scenario_id"] == scenario_id].iloc[0]
            scenario_judgments = judgments[
                judgments["scenario_id"] == scenario_id
            ][["judge_seed", "verdict", "score", "reason"]].sort_values("judge_seed")

            left, right = st.columns(2)
            with left:
                st.markdown("**Candidate action**")
                st.code(scenario["candidate_action"])
                st.write(scenario["candidate_rationale"])
            with right:
                st.markdown("**Target action**")
                st.code(scenario["target_action"])
                st.write(scenario["target_rationale"])

            st.dataframe(
                scenario_judgments,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No scenarios match the current filters.")


if view == "Fleet Health":
    fleet_health_view()
elif view == "Policy Benchmark":
    policy_benchmark_view()
else:
    decision_evaluation_view()

st.divider()
st.caption(
    "Week 7 analyst dashboard · Synthetic and benchmark data only · "
    "Authoritative decision correctness remains exact match against target_action."
)
