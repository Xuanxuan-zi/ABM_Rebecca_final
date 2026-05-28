"""
Run batch experiments for the campus cultural segregation ABM.

This script runs each population scenario with and without the R5 university
intervention. It repeats each condition across multiple random seeds and saves
the results as CSV files.

Outputs:
1. batch_results_all_ticks.csv
   - All model outcomes for every tick of every run.

2. batch_results_final_tick.csv
   - Only the final tick from each run.

3. batch_summary_mean.csv
   - Mean outcomes by scenario and intervention condition.

4. batch_summary_mean_sd.csv
   - Mean and standard deviation by scenario and intervention condition.
"""

from model import run_batch_experiments


def main():
    # Main batch settings
    n_runs = 20
    n_ticks = 300
    n_agents = 500

    # Run all scenario and intervention combinations
    results = run_batch_experiments(
        scenarios=("baseline", "chinese_heavy", "balanced"),
        interventions=(False, True),
        n_runs=n_runs,
        n_ticks=n_ticks,
        n_agents=n_agents,
        width=50,
        height=50,
        base_seed=1000,
        verbose=True,
    )

    # Save all tick-level results
    results.to_csv("batch_results_all_ticks.csv", index=False)

    # Keep only the final tick of each run
    final_results = results[results["tick"] == n_ticks].copy()
    final_results.to_csv("batch_results_final_tick.csv", index=False)

    # Main outcome variables to summarize
    outcome_vars = [
        "spatial_clustering",
        "network_homophily",
        "cross_cultural_friendship_share",
        "mean_stress",
        "mean_stress_active",
        "isolation_rate",
        "active_social_ties",
        "mean_tie_strength",
        "mean_cultural_dist",
    ]

    # Calculate mean outcomes by scenario and intervention
    summary_mean = (
        final_results
        .groupby(["scenario", "intervention"])[outcome_vars]
        .mean()
        .reset_index()
    )
    summary_mean.to_csv("batch_summary_mean.csv", index=False)

    # Calculate mean and standard deviation for more formal reporting
    summary_mean_sd = (
        final_results
        .groupby(["scenario", "intervention"])[outcome_vars]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_mean_sd.to_csv("batch_summary_mean_sd.csv", index=False)

    print("\nBatch experiment finished.")
    print(f"Each condition was run {n_runs} times.")
    print(f"Each run stopped at tick {n_ticks}.")
    print("\nMean outcomes at final tick:")
    print(summary_mean)


if __name__ == "__main__":
    main()
