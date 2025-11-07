import numpy as np
import pandas as pd
from typing import List, Tuple
from src.evolution_strategy import EvolutionStrategy
from src.test_functions import TestFunctions
from src.es_params import ESParams


class ExperimentRunner:
    """Run comprehensive experiments with different configurations"""
    
    def __init__(self, n_runs: int = 30):
        self.n_runs = n_runs
        self.results = []
        
    def run_experiment(self, func_name: str, dim: int, mu: int, lambda_: int, 
                       strategy: str, max_gen: int = 500) -> Tuple[pd.DataFrame, List]:
        """
        Run multiple independent trials
        """
        print(f"\n{'='*60}")
        print(f"Running: {func_name}, dim={dim}, μ={mu}, λ={lambda_}, strategy={strategy}")
        print(f"{'='*60}")
        
        # Get function and bounds
        func = getattr(TestFunctions, func_name)
        lower, upper = TestFunctions.get_bounds(func_name, dim)
        
        # Setup parameters
        tau = 1 / np.sqrt(2 * dim)
        tau_prime = 1 / np.sqrt(2 * np.sqrt(dim))
        
        params = ESParams(
            mu=mu,
            lambda_=lambda_,
            dim=dim,
            sigma=0.5,
            tau=tau,
            tau_prime=tau_prime,
            max_generations=max_gen,
            target_fitness=1e-6,
            strategy=strategy
        )
        
        # Run multiple trials
        trial_results = []
        all_histories = []
        
        for run in range(self.n_runs):
            if run % 10 == 0:
                print(f"Run {run+1}/{self.n_runs}...", end=' ')
            
            es = EvolutionStrategy(params, func, (lower, upper))
            result = es.run(verbose=False)
            
            trial_results.append({
                'function': func_name,
                'dimension': dim,
                'mu': mu,
                'lambda': lambda_,
                'strategy': strategy,
                'run': run + 1,
                'best_fitness': result['best_fitness'],
                'generations': result['generations'],
                'function_evals': result['function_evaluations'],
                'time': result['time'],
                'converged': result['converged']
            })
            
            all_histories.append(result['best_history'])
        
        print("Done!")
        
        # Store results
        df = pd.DataFrame(trial_results)
        
        # Print summary statistics
        print(f"\nResults Summary:")
        print(f"  Mean Best Fitness: {df['best_fitness'].mean():.6e} ± {df['best_fitness'].std():.6e}")
        print(f"  Success Rate: {df['converged'].sum()}/{self.n_runs} ({100*df['converged'].mean():.1f}%)")
        print(f"  Mean Generations: {df['generations'].mean():.1f} ± {df['generations'].std():.1f}")
        print(f"  Mean Time: {df['time'].mean():.3f}s ± {df['time'].std():.3f}s")
        
        return df, all_histories