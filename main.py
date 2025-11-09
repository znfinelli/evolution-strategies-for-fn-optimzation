"""
EVOLUTION STRATEGIES FOR FUNCTION OPTIMIZATION
Master in Artificial Intelligence - Evolutionary Computation
November 2025

This script is the main entry point for our project. It orchestrates 
the entire experimental pipeline:
1.  Defines the set of experiments to run.
2.  Initializes the ExperimentRunner (which handles the 30 independent runs).
3.  Calls the runner for each experimental configuration.
4.  Collects and aggregates all results into a single pandas DataFrame.
5.  Generates and saves all required plots and CSV files to the 'outputs/' dir.
"""

import numpy as np
import pandas as pd
import warnings
import os

# Import our custom modules from the 'src' package
from src import (
    ExperimentRunner,
    TestFunctions,  # We don't strictly need this, but good for clarity
    plot_convergence,
    plot_boxplots
)

# Suppress common warnings (e.g., from matplotlib) for a cleaner console output
warnings.filterwarnings('ignore')

# Set a global random seed for reproducibility, as required by the project brief.
# This ensures our 30-run experiments are repeatable.
np.random.seed(42)


def main():
    """Main experimental pipeline"""
    
    print("="*80)
    print("EVOLUTION STRATEGIES FOR FUNCTION OPTIMIZATION")
    print("Master in AI - Evolutionary Computation")
    print("="*80)
    
    # Initialize the runner. n_runs=30 for statistical rigor, per the project brief.
    runner = ExperimentRunner(n_runs=30)
    
    # --- Define Experimental Matrix ---
    # We will compare (μ,λ)-ES ("comma") vs. (μ+λ)-ES ("plus")
    # on two function types:
    #   1. Sphere: Unimodal, simple benchmark to verify the algorithm works.
    #   2. Rastrigin: Highly multimodal, a "difficult" function as per the brief.
    # We will also test scalability by increasing dimension (n=10 vs n=20).
    experiments = [
        # --- Sphere function ---
        {'func': 'sphere', 'dim': 10, 'mu': 15, 'lambda': 100, 'strategy': 'comma'},
        {'func': 'sphere', 'dim': 10, 'mu': 15, 'lambda': 100, 'strategy': 'plus'},
        {'func': 'sphere', 'dim': 20, 'mu': 15, 'lambda': 100, 'strategy': 'comma'},
        
        # --- Rastrigin function ---
        # Using a ~1:7 ratio for μ:λ as recommended in the slides for ES
        # (specifically, the (μ,λ)[cite_start]-ES is often tuned for μ/λ ≈ 1/6 [cite: 88, 279]).
        {'func': 'rastrigin', 'dim': 10, 'mu': 20, 'lambda': 140, 'strategy': 'comma'},
        {'func': 'rastrigin', 'dim': 10, 'mu': 20, 'lambda': 140, 'strategy': 'plus'},
        # Increase μ and λ for higher dimension to maintain search power
        {'func': 'rastrigin', 'dim': 20, 'mu': 30, 'lambda': 200, 'strategy': 'comma'},
    ]
    
    all_results_list = []
    all_histories_dict = {}
    
    # --- 1. Run all experiments ---
    for exp_config in experiments:
        # run_experiment handles the 30 independent trials for this config
        df_results, list_of_histories = runner.run_experiment(
            func_name=exp_config['func'],
            dim=exp_config['dim'],
            mu=exp_config['mu'],
            lambda_=exp_config['lambda'],
            strategy=exp_config['strategy'],
            max_gen=500  # Set a generous generation limit
        )
        all_results_list.append(df_results)
        
        # Store all 30 fitness histories for this experiment config
        # We'll average these for the convergence plots
        key = (
            f"{exp_config['func']}_d{exp_config['dim']}_"
            f"μ{exp_config['mu']}_λ{exp_config['lambda']}_{exp_config['strategy']}"
        )
        all_histories_dict[key] = list_of_histories
    
    # Combine all results into a single DataFrame for analysis
    combined_df = pd.concat(all_results_list, ignore_index=True)
    
    # --- 2. Save Artifacts (CSVs and Plots) ---
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True) # Ensure dir exists

    # Save detailed, run-by-run results to CSV
    results_path = os.path.join(output_dir, 'results.csv')
    combined_df.to_csv(results_path, index=False)
    print(f"\n{'='*80}")
    print(f"Detailed run data saved to: {results_path}")
    
    # --- 3. Generate Visualizations (as required by brief) ---
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS...")
    print(f"{'='*80}")
    
    # Plot convergence for Sphere
    sphere_histories = {k: v for k, v in all_histories_dict.items() if 'sphere' in k}
    if sphere_histories:
        plot_convergence(
            sphere_histories,
            'Convergence on Sphere Function (Avg. of 30 Runs)',
            os.path.join(output_dir, 'convergence_sphere.png')
        )
    
    # Plot convergence for Rastrigin
    rastrigin_histories = {k: v for k, v in all_histories_dict.items() if 'rastrigin' in k}
    if rastrigin_histories:
        plot_convergence(
            rastrigin_histories,
            'Convergence on Rastrigin Function (Avg. of 30 Runs)',
            os.path.join(output_dir, 'convergence_rastrigin.png')
        )
    
    # Plot comparative boxplots
    plot_boxplots(combined_df, os.path.join(output_dir, 'comparison_boxplots.png'))
    
    # --- 4. Generate Summary Statistics Table (for the report) ---
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS (for final report)")
    print(f"{'='*80}")
    
    # Aggregate results for the report
    summary = combined_df.groupby(['function', 'dimension', 'mu', 'lambda', 'strategy']).agg({
        'best_fitness': ['mean', 'std', 'min'],
        'generations': ['mean', 'std'],
        'function_evals': ['mean', 'std'],
        'time': ['mean', 'std'],
        'converged': ['sum', 'mean'] # 'mean' gives success rate
    }).round(6)
    
    print(summary)
    
    # Save summary table to CSV
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(summary_path)
    print(f"\nSummary statistics saved to: {summary_path}")
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Find all outputs in: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()