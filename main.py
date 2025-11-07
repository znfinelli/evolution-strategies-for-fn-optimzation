"""
EVOLUTION STRATEGIES FOR FUNCTION OPTIMIZATION
Master in Artificial Intelligence - Evolutionary Computation
November 2025
"""

import numpy as np
import pandas as pd
import warnings
import os

from src import (
    ExperimentRunner,
    TestFunctions,
    plot_convergence,
    plot_boxplots
)

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def main():
    """Main experimental pipeline"""
    
    print("="*80)
    print("EVOLUTION STRATEGIES FOR FUNCTION OPTIMIZATION")
    print("Master in AI - Evolutionary Computation")
    print("="*80)
    
    # Create experiment runner
    runner = ExperimentRunner(n_runs=30)
    
    # Define experiments
    experiments = [
        # Sphere function - easy unimodal
        {'func': 'sphere', 'dim': 10, 'mu': 15, 'lambda': 100, 'strategy': 'comma'},
        {'func': 'sphere', 'dim': 10, 'mu': 15, 'lambda': 100, 'strategy': 'plus'},
        {'func': 'sphere', 'dim': 20, 'mu': 15, 'lambda': 100, 'strategy': 'comma'},
        
        # Rastrigin - multimodal
        {'func': 'rastrigin', 'dim': 10, 'mu': 20, 'lambda': 140, 'strategy': 'comma'},
        {'func': 'rastrigin', 'dim': 10, 'mu': 20, 'lambda': 140, 'strategy': 'plus'},
        {'func': 'rastrigin', 'dim': 20, 'mu': 30, 'lambda': 200, 'strategy': 'comma'},
    ]
    
    all_results = []
    all_histories = {}
    
    # Run all experiments
    for exp in experiments:
        df, histories = runner.run_experiment(
            func_name=exp['func'],
            dim=exp['dim'],
            mu=exp['mu'],
            lambda_=exp['lambda'],
            strategy=exp['strategy'],
            max_gen=500
        )
        all_results.append(df)
        
        # Store histories for plotting
        key = f"{exp['func']}_d{exp['dim']}_μ{exp['mu']}_λ{exp['lambda']}_{exp['strategy']}"
        all_histories[key] = histories
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Prepare output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Save results to CSV
    results_path = os.path.join(output_dir, 'results.csv')
    combined_df.to_csv(results_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_path}")
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Convergence plots for Sphere
    sphere_histories = {k: v for k, v in all_histories.items() if 'sphere' in k}
    plot_convergence(
        sphere_histories,
        'Convergence on Sphere Function',
        os.path.join(output_dir, 'convergence_sphere.png')
    )
    
    # Convergence plots for Rastrigin
    rastrigin_histories = {k: v for k, v in all_histories.items() if 'rastrigin' in k}
    plot_convergence(
        rastrigin_histories,
        'Convergence on Rastrigin Function',
        os.path.join(output_dir, 'convergence_rastrigin.png')
    )
    
    # Box plots
    plot_boxplots(combined_df, os.path.join(output_dir, 'comparison_boxplots.png'))
    
    # Summary statistics table
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    summary = combined_df.groupby(['function', 'dimension', 'mu', 'lambda', 'strategy']).agg({
        'best_fitness': ['mean', 'std', 'min'],
        'generations': ['mean', 'std'],
        'function_evals': ['mean', 'std'],
        'time': ['mean', 'std'],
        'converged': ['sum', 'mean']
    }).round(6)
    
    print(summary)
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(summary_path)
    print(f"\nSummary statistics saved to: {summary_path}")
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print("  - results.csv (detailed results)")
    print("  - summary_statistics.csv (aggregated statistics)")
    print("  - convergence_sphere.png")
    print("  - convergence_rastrigin.png")
    print("  - comparison_boxplots.png")


if __name__ == "__main__":
    main()