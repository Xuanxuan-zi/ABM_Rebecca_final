"""
Create polished, paper-ready visualizations for the campus cultural segregation ABM.

Input:
    batch_results_final_tick.csv

Run:
    python3 visualize_batch_results_fancy.py

Outputs:
    figures_fancy/fig_01_main_network_outcomes.png
    figures_fancy/fig_02_policy_effect_lollipop.png
    figures_fancy/fig_03_spatial_and_welfare_outcomes.png
    figures_fancy/fig_04_full_outcome_dashboard.png
    figures_fancy/fancy_summary_table.csv
    figures_fancy/fancy_intervention_effects.csv

Design logic:
    - Use batch_results_final_tick.csv instead of batch_summary_mean.csv,
      because the final-tick file contains all 20 repeated runs per condition.
    - Each point shows the mean across runs.
    - Error bars show 95% confidence intervals.
    - The paired lollipop plot shows the intervention effect:
      with intervention minus without intervention.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT_FILE = Path("batch_results_final_tick.csv")
OUTPUT_DIR = Path("figures_fancy")


# ---------------------------------------------------------------------
# Labels and plotting order
# ---------------------------------------------------------------------

SCENARIO_ORDER = ["baseline", "chinese_heavy", "balanced"]

SCENARIO_LABELS = {
    "baseline": "Baseline",
    "chinese_heavy": "Chinese-heavy",
    "balanced": "Balanced",
}

INTERVENTION_LABELS = {
    False: "No intervention",
    True: "With intervention",
}

OUTCOME_LABELS = {
    "cross_cultural_friendship_share": "Cross-cultural\nfriendship share",
    "network_homophily": "Network\nhomophily",
    "spatial_clustering": "Spatial\nclustering",
    "mean_stress": "Mean\nstress",
    "isolation_rate": "Isolation\nrate",
    "active_social_ties": "Active\nsocial ties",
    "mean_cultural_dist": "Mean cultural\ndistance",
}

PERCENT_OUTCOMES = {
    "cross_cultural_friendship_share",
    "network_homophily",
    "spatial_clustering",
    "mean_stress",
    "isolation_rate",
    "mean_cultural_dist",
}

MAIN_NETWORK_OUTCOMES = [
    "cross_cultural_friendship_share",
    "network_homophily",
]

SPATIAL_WELFARE_OUTCOMES = [
    "spatial_clustering",
    "mean_stress",
    "isolation_rate",
]

FULL_DASHBOARD_OUTCOMES = [
    "cross_cultural_friendship_share",
    "network_homophily",
    "spatial_clustering",
    "mean_stress",
    "isolation_rate",
    "active_social_ties",
    "mean_cultural_dist",
]


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_results() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Put this script in the same folder "
            "as batch_results_final_tick.csv."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"scenario", "intervention"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["intervention"] = df["intervention"].astype(bool)
    df["scenario"] = pd.Categorical(
        df["scenario"],
        categories=SCENARIO_ORDER,
        ordered=True,
    )

    return df.sort_values(["scenario", "intervention"]).reset_index(drop=True)


def summarize(df: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    rows = []

    for outcome in outcomes:
        if outcome not in df.columns:
            continue

        one = (
            df.groupby(["scenario", "intervention"], observed=True)[outcome]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        one["se"] = one["std"] / np.sqrt(one["count"])
        one["ci95"] = 1.96 * one["se"]
        one["outcome"] = outcome
        rows.append(one)

    if not rows:
        raise ValueError("No requested outcomes were found in the input file.")

    return pd.concat(rows, ignore_index=True)


def effect_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for outcome in summary["outcome"].unique():
        one = summary[summary["outcome"] == outcome]
        wide = one.pivot(index="scenario", columns="intervention", values="mean")

        if False not in wide.columns or True not in wide.columns:
            continue

        wide = wide.rename(columns={False: "no_intervention", True: "with_intervention"})
        wide["difference"] = wide["with_intervention"] - wide["no_intervention"]
        wide["percent_change"] = (
            wide["difference"] / wide["no_intervention"].replace(0, np.nan)
        ) * 100
        wide["outcome"] = outcome
        rows.append(wide.reset_index())

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=10)


def apply_y_format(ax, outcome: str):
    if outcome in PERCENT_OUTCOMES:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def panel_pointplot(ax, summary: pd.DataFrame, outcome: str, title: str):
    one = summary[summary["outcome"] == outcome].copy()

    x = np.arange(len(SCENARIO_ORDER))
    offsets = {False: -0.14, True: 0.14}
    markers = {False: "o", True: "D"}

    for intervention in [False, True]:
        sub = (
            one[one["intervention"] == intervention]
            .set_index("scenario")
            .reindex(SCENARIO_ORDER)
            .reset_index()
        )

        ax.errorbar(
            x + offsets[intervention],
            sub["mean"],
            yerr=sub["ci95"],
            marker=markers[intervention],
            linestyle="none",
            capsize=4,
            markersize=6,
            label=INTERVENTION_LABELS[intervention],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.set_title(title, fontsize=12, fontweight="bold")
    apply_y_format(ax, outcome)
    clean_axis(ax)

    lower = (one["mean"] - one["ci95"]).min()
    upper = (one["mean"] + one["ci95"]).max()
    padding = (upper - lower) * 0.25 if upper > lower else 0.05

    if outcome in PERCENT_OUTCOMES:
        ax.set_ylim(max(0, lower - padding), min(1, upper + padding))
    else:
        ax.set_ylim(max(0, lower - padding), upper + padding)


def add_intervention_annotation(ax, summary: pd.DataFrame, outcome: str):
    """Add small plus/minus labels showing intervention direction by scenario."""
    one = summary[summary["outcome"] == outcome]
    means = one.pivot(index="scenario", columns="intervention", values="mean")

    if False not in means.columns or True not in means.columns:
        return

    y_top = ax.get_ylim()[1]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    label_y = y_top - 0.08 * y_span

    for i, scenario in enumerate(SCENARIO_ORDER):
        if scenario not in means.index:
            continue
        diff = means.loc[scenario, True] - means.loc[scenario, False]
        if outcome in PERCENT_OUTCOMES:
            label = f"{diff:+.1%}"
        else:
            label = f"{diff:+.1f}"
        ax.text(i, label_y, label, ha="center", va="top", fontsize=9)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def make_main_network_figure(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    for ax, outcome in zip(axes, MAIN_NETWORK_OUTCOMES):
        panel_pointplot(
            ax,
            summary,
            outcome,
            OUTCOME_LABELS[outcome].replace("\n", " "),
        )
        add_intervention_annotation(ax, summary, outcome)

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(
        "Network-level intervention effects at tick 300",
        fontsize=15,
        fontweight="bold",
        y=1.16,
    )

    fig.savefig(OUTPUT_DIR / "fig_01_main_network_outcomes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_policy_effect_lollipop(effects: pd.DataFrame):
    selected = effects[
        effects["outcome"].isin(
            [
                "cross_cultural_friendship_share",
                "network_homophily",
                "spatial_clustering",
                "mean_stress",
                "isolation_rate",
                "mean_cultural_dist",
            ]
        )
    ].copy()

    # Use percentage-point changes for proportion outcomes.
    selected["plot_difference"] = selected["difference"]
    selected.loc[selected["outcome"].isin(PERCENT_OUTCOMES), "plot_difference"] *= 100

    selected["label"] = (
        selected["outcome"].map(lambda x: OUTCOME_LABELS.get(x, x).replace("\n", " "))
        + " | "
        + selected["scenario"].astype(str).map(SCENARIO_LABELS)
    )

    selected = selected.sort_values(["outcome", "scenario"])

    y = np.arange(len(selected))

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.axvline(0, linewidth=1)

    for i, value in enumerate(selected["plot_difference"]):
        ax.plot([0, value], [i, i], linewidth=2)
        ax.scatter(value, i, s=48)

    ax.set_yticks(y)
    ax.set_yticklabels(selected["label"], fontsize=9)
    ax.set_xlabel("Intervention effect\nwith intervention minus no intervention")
    ax.set_title(
        "Direction and size of intervention effects",
        fontsize=14,
        fontweight="bold",
    )

    for i, value in enumerate(selected["plot_difference"]):
        text = f"{value:+.2f}"
        ax.text(
            value,
            i,
            "  " + text if value >= 0 else text + "  ",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )

    clean_axis(ax)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.10)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_02_policy_effect_lollipop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_spatial_welfare_figure(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)

    for ax, outcome in zip(axes, SPATIAL_WELFARE_OUTCOMES):
        panel_pointplot(
            ax,
            summary,
            outcome,
            OUTCOME_LABELS[outcome].replace("\n", " "),
        )
        add_intervention_annotation(ax, summary, outcome)

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(
        "Spatial clustering and welfare outcomes at tick 300",
        fontsize=15,
        fontweight="bold",
        y=1.16,
    )

    fig.savefig(OUTPUT_DIR / "fig_03_spatial_and_welfare_outcomes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_dashboard(summary: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), constrained_layout=True)
    axes = axes.ravel()

    for ax, outcome in zip(axes, FULL_DASHBOARD_OUTCOMES):
        panel_pointplot(
            ax,
            summary,
            outcome,
            OUTCOME_LABELS[outcome].replace("\n", " "),
        )
        add_intervention_annotation(ax, summary, outcome)

    # Hide unused panel.
    if len(FULL_DASHBOARD_OUTCOMES) < len(axes):
        for ax in axes[len(FULL_DASHBOARD_OUTCOMES):]:
            ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle(
        "Batch-run outcome dashboard at tick 300",
        fontsize=16,
        fontweight="bold",
        y=1.10,
    )

    fig.savefig(OUTPUT_DIR / "fig_04_full_outcome_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_results()
    summary = summarize(df, FULL_DASHBOARD_OUTCOMES)
    effects = effect_table(summary)

    summary["scenario"] = summary["scenario"].astype(str)
    effects["scenario"] = effects["scenario"].astype(str)

    summary.to_csv(OUTPUT_DIR / "fancy_summary_table.csv", index=False)
    effects.to_csv(OUTPUT_DIR / "fancy_intervention_effects.csv", index=False)

    make_main_network_figure(summary)
    make_policy_effect_lollipop(effects)
    make_spatial_welfare_figure(summary)
    make_dashboard(summary)

    print("Fancy visualization finished.")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("\nGenerated figures:")
    print("  - fig_01_main_network_outcomes.png")
    print("  - fig_02_policy_effect_lollipop.png")
    print("  - fig_03_spatial_and_welfare_outcomes.png")
    print("  - fig_04_full_outcome_dashboard.png")
    print("\nGenerated tables:")
    print("  - fancy_summary_table.csv")
    print("  - fancy_intervention_effects.csv")


if __name__ == "__main__":
    main()
