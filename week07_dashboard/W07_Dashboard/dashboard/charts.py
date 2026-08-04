"""Plotly chart builders used by the Week 7 dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def polish(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=60, b=30),
        legend_title_text="",
        hoverlabel=dict(namelength=-1),
    )
    return fig


def anomaly_metric_bars(frame: pd.DataFrame, scope: str) -> go.Figure:
    filtered = frame[frame["scope"] == scope].copy()
    long = filtered.melt(
        id_vars=["method"],
        value_vars=["precision_mean", "recall_mean", "f1_mean", "auroc_mean"],
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].str.replace("_mean", "", regex=False).str.upper()
    fig = px.bar(
        long,
        x="method",
        y="value",
        color="metric",
        barmode="group",
        text_auto=".2f",
        title=f"Detector quality · {scope.replace('_', ' ').title()}",
        labels={"method": "", "value": "Score"},
    )
    fig.update_yaxes(range=[0, 1.05])
    return polish(fig)


def anomaly_heatmap(frame: pd.DataFrame) -> go.Figure:
    filtered = frame[frame["scope"] != "all_faults"].copy()
    filtered["scope_label"] = (
        filtered["scope"].str.replace("_", " ", regex=False).str.title()
    )
    matrix = filtered.pivot(
        index="scope_label",
        columns="method",
        values="auroc_mean",
    )
    fig = px.imshow(
        matrix,
        text_auto=".3f",
        aspect="auto",
        title="AUROC by fault type and detector",
        labels={"x": "Detector", "y": "Fault type", "color": "AUROC"},
    )
    return polish(fig, height=470)


def anomaly_split_box(frame: pd.DataFrame, scope: str) -> go.Figure:
    filtered = frame[frame["scope"] == scope].copy()
    fig = px.box(
        filtered,
        x="method",
        y="auroc",
        points="all",
        title="Seed-to-seed AUROC stability",
        labels={"method": "", "auroc": "AUROC"},
    )
    fig.update_yaxes(range=[0, 1.05])
    return polish(fig)


def fleet_metric_line(
    frame: pd.DataFrame,
    unit_id: str,
    metric: str,
    title: str,
    y_label: str,
) -> go.Figure:
    filtered = frame[frame["robot_id"] == unit_id].copy()
    fig = px.line(
        filtered,
        x="timestamp",
        y=metric,
        title=title,
        labels={"timestamp": "", metric: y_label},
    )
    return polish(fig, height=330)


def learning_curve(
    frame: pd.DataFrame,
    environment: str,
    algorithms: list[str],
    seeds: list[int],
) -> go.Figure:
    filtered = frame[
        (frame["environment"] == environment)
        & (frame["algorithm"].isin(algorithms))
        & (frame["seed"].isin(seeds))
    ].copy()

    grouped = (
        filtered.groupby(["algorithm", "timesteps"], as_index=False)
        .agg(mean_reward=("mean_reward", "mean"), reward_std=("mean_reward", "std"))
        .fillna({"reward_std": 0.0})
    )

    fig = go.Figure()
    for algorithm in algorithms:
        part = grouped[grouped["algorithm"] == algorithm].sort_values("timesteps")
        if part.empty:
            continue
        upper = part["mean_reward"] + part["reward_std"]
        lower = part["mean_reward"] - part["reward_std"]
        fig.add_trace(
            go.Scatter(
                x=part["timesteps"],
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=part["timesteps"],
                y=lower,
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                opacity=0.13,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=part["timesteps"],
                y=part["mean_reward"],
                mode="lines+markers",
                name=algorithm,
                customdata=part[["reward_std"]],
                hovertemplate=(
                    "Timesteps %{x:,.0f}<br>"
                    "Mean reward %{y:.1f}<br>"
                    "Across-seed SD %{customdata[0]:.1f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Mean learning curve · {environment}",
        xaxis_title="Training timesteps",
        yaxis_title="Evaluation reward",
    )
    return polish(fig, height=450)


def reward_by_seed(
    frame: pd.DataFrame,
    environment: str,
    algorithms: list[str],
    seeds: list[int],
) -> go.Figure:
    filtered = frame[
        (frame["environment_requested"] == environment)
        & (frame["algorithm"].isin(algorithms))
        & (frame["seed"].isin(seeds))
    ].copy()
    fig = px.strip(
        filtered,
        x="algorithm",
        y="final_eval_mean_reward",
        color="algorithm",
        hover_data=["seed", "threshold_reached", "wallclock_sec"],
        title="Final reward by seed",
        labels={"algorithm": "", "final_eval_mean_reward": "Final reward"},
    )
    return polish(fig, height=400)


def policy_tradeoff(frame: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        frame,
        x="wallclock_sec_mean",
        y="final_reward_mean",
        color="algorithm",
        symbol="environment",
        size="threshold_success_rate",
        size_max=28,
        hover_data={
            "convergence_stability_cv": ":.3f",
            "threshold_success_rate": ":.0%",
            "wallclock_sec_mean": ":.1f",
            "final_reward_mean": ":.1f",
        },
        title="Performance versus wall-clock cost",
        labels={
            "wallclock_sec_mean": "Mean wall-clock seconds",
            "final_reward_mean": "Mean final reward",
        },
    )
    return polish(fig, height=430)


def multi_agent_comparison(frame: pd.DataFrame) -> go.Figure:
    metric_map = {
        "mean_agent_return_mean_across_seed_mean": "Mean agent return",
        "final_assignment_distance_mean_across_seed_mean": "Final assignment distance",
        "collision_pairs_per_cycle_mean_across_seed_mean": "Collision pairs / cycle",
    }
    long = frame.melt(
        id_vars=["policy"],
        value_vars=list(metric_map),
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(metric_map)
    fig = px.bar(
        long,
        x="policy",
        y="value",
        color="policy",
        facet_col="metric",
        facet_col_wrap=3,
        text_auto=".3f",
        title="Multi-agent diagnostics · PPO versus random",
        labels={"policy": "", "value": "Measured value"},
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(
        lambda annotation: annotation.update(
            text=annotation.text.split("=")[-1]
        )
    )
    return polish(fig, height=430)


def reliability_bars(frame: pd.DataFrame) -> go.Figure:
    filtered = frame[frame["scope"] == "domain"].copy()
    filtered["domain_label"] = (
        filtered["domain"]
        .replace(
            {
                "aido_rover": "Aido Rover",
                "sentinel_prime": "Sentinel Prime",
                "senpai": "Senpai",
                "fari": "Fari",
            }
        )
    )
    fig = px.bar(
        filtered,
        x="domain_label",
        y="krippendorff_alpha_nominal",
        text_auto=".3f",
        hover_data=["exact_three_run_agreement", "interpretation"],
        title="Seeded judge reliability by product domain",
        labels={
            "domain_label": "",
            "krippendorff_alpha_nominal": "Nominal Krippendorff alpha",
        },
    )
    fig.add_hline(y=0.8, line_dash="dot", annotation_text="Strong")
    fig.add_hline(y=0.667, line_dash="dot", annotation_text="Tentative")
    fig.update_yaxes(range=[0, 1.05])
    return polish(fig, height=430)


def verdict_distribution(frame: pd.DataFrame) -> go.Figure:
    counts = (
        frame.groupby(["domain", "verdict"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    counts["domain"] = counts["domain"].replace(
        {
            "aido_rover": "Aido Rover",
            "sentinel_prime": "Sentinel Prime",
            "senpai": "Senpai",
            "fari": "Fari",
        }
    )
    fig = px.bar(
        counts,
        x="domain",
        y="count",
        color="verdict",
        barmode="stack",
        title="PASS / PARTIAL / FAIL distribution",
        labels={"domain": "", "count": "Seeded judgments"},
        category_orders={"verdict": ["PASS", "PARTIAL", "FAIL"]},
    )
    return polish(fig, height=420)


def exact_match_verdict(frame: pd.DataFrame) -> go.Figure:
    counts = (
        frame.groupby(["exact_match", "verdict"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    counts["contract_status"] = counts["exact_match"].map(
        {True: "Exact action match", False: "Non-exact action"}
    )
    fig = px.bar(
        counts,
        x="contract_status",
        y="count",
        color="verdict",
        barmode="group",
        text_auto=True,
        title="Authoritative exact match versus LLM verdict",
        labels={"contract_status": "", "count": "Judgments"},
        category_orders={"verdict": ["PASS", "PARTIAL", "FAIL"]},
    )
    return polish(fig, height=420)
