import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List


def plot_convergence(histories_dict: Dict[str, List], title: str, save_path: str):
    """Plot convergence curves for multiple configurations"""
    plt.figure(figsize=(10, 6))
    
    for label, histories in histories_dict.items():
        # Compute mean and std across runs
        max_len = max(len(h) for h in histories)
        
        # Pad histories to same length
        padded = []
        for h in histories:
            if len(h) < max_len:
                padded.append(h + [h[-1]] * (max_len - len(h)))
            else:
                padded.append(h)
        
        histories_array = np.array(padded)
        mean = np.mean(histories_array, axis=0)
        std = np.std(histories_array, axis=0)
        
        generations = np.arange(len(mean))
        
        plt.semilogy(generations, mean, label=label, linewidth=2)
        plt.fill_between(generations, mean - std, mean + std, alpha=0.2)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness (log scale)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_boxplots(df: pd.DataFrame, save_path: str):
    """Create box plots comparing different configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Best fitness comparison
    ax = axes[0, 0]
    data_to_plot = []
    labels = []
    for (func, mu, lam, strat), group in df.groupby(['function', 'mu', 'lambda', 'strategy']):
        data_to_plot.append(group['best_fitness'].values)
        labels.append(f"{func[:3]}\nμ={mu},λ={lam}\n{strat}")
    
    ax.boxplot(data_to_plot, labels=labels)
    ax.set_ylabel('Best Fitness', fontsize=11)
    ax.set_title('Best Fitness Distribution', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Generations comparison
    ax = axes[0, 1]
    data_to_plot = []
    for (func, mu, lam, strat), group in df.groupby(['function', 'mu', 'lambda', 'strategy']):
        data_to_plot.append(group['generations'].values)
    
    ax.boxplot(data_to_plot, labels=labels)
    ax.set_ylabel('Generations', fontsize=11)
    ax.set_title('Generations to Convergence', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3)
    
    # Function evaluations
    ax = axes[1, 0]
    data_to_plot = []
    for (func, mu, lam, strat), group in df.groupby(['function', 'mu', 'lambda', 'strategy']):
        data_to_plot.append(group['function_evals'].values)
    
    ax.boxplot(data_to_plot, labels=labels)
    ax.set_ylabel('Function Evaluations', fontsize=11)
    ax.set_title('Function Evaluations', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3)
    
    # Success rate
    ax = axes[1, 1]
    success_rates = []
    for (func, mu, lam, strat), group in df.groupby(['function', 'mu', 'lambda', 'strategy']):
        success_rates.append(100 * group['converged'].mean())
    
    ax.bar(range(len(labels)), success_rates, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Success Rate (Convergence < 1e-6)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")