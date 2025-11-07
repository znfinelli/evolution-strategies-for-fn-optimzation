import numpy as np
import time
from typing import Callable, Tuple, Dict
from .es_params import ESParams


class EvolutionStrategy:
    """
    Implementation of (μ,λ)-ES and (μ+λ)-ES with self-adaptive mutation
    """
    
    def __init__(self, params: ESParams, fitness_func: Callable, bounds: Tuple[np.ndarray, np.ndarray]):
        self.params = params
        self.fitness_func = fitness_func
        self.lower_bounds, self.upper_bounds = bounds
        
        # Initialize population
        self.population = None
        self.sigmas = None
        self.fitness_values = None
        
        # Statistics
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.generations_run = 0
        self.function_evaluations = 0
        
    def initialize_population(self):
        """Initialize population uniformly in search space"""
        self.population = np.random.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(self.params.mu, self.params.dim)
        )
        self.sigmas = np.full(self.params.mu, self.params.sigma)
        # Evaluate and store fitness values for the initialized population
        self.fitness_values = self.evaluate_population(self.population)
        
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness for all individuals"""
        fitness = np.array([self.fitness_func(ind) for ind in population])
        self.function_evaluations += len(population)
        return fitness
    
    def mutate(self, parent: np.ndarray, sigma: float) -> Tuple[np.ndarray, float]:
        """
        Apply self-adaptive mutation
        First mutate sigma, then use it to mutate the individual
        """
        # Self-adaptation of sigma
        new_sigma = sigma * np.exp(self.params.tau_prime * np.random.randn() + 
                                    self.params.tau * np.random.randn())
        
        # Mutate individual
        offspring = parent + new_sigma * np.random.randn(self.params.dim)
        
        # Clip to bounds
        offspring = np.clip(offspring, self.lower_bounds, self.upper_bounds)
        
        return offspring, new_sigma
    
    def select_parents(self, population: np.ndarray, fitness: np.ndarray, 
                       sigmas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select μ best individuals"""
        indices = np.argsort(fitness)[:self.params.mu]
        return population[indices], sigmas[indices]
    
    def run(self, verbose: bool = False) -> Dict:
        """
        Run the Evolution Strategy
        Returns dictionary with results
        """
        start_time = time.time()
        self.initialize_population()
        
        for generation in range(self.params.max_generations):
            # Generate offspring
            offspring_pop = []
            offspring_sigmas = []
            
            for _ in range(self.params.lambda_):
                # Select random parent
                parent_idx = np.random.randint(self.params.mu)
                parent = self.population[parent_idx]
                parent_sigma = self.sigmas[parent_idx]
                
                # Mutate
                offspring, new_sigma = self.mutate(parent, parent_sigma)
                offspring_pop.append(offspring)
                offspring_sigmas.append(new_sigma)
            
            offspring_pop = np.array(offspring_pop)
            offspring_sigmas = np.array(offspring_sigmas)
            
            # Evaluate offspring
            offspring_fitness = self.evaluate_population(offspring_pop)
            
            # Selection strategy
            if self.params.strategy == 'comma':
                # (μ,λ)-ES: select only from offspring
                self.population, self.sigmas = self.select_parents(
                    offspring_pop, offspring_fitness, offspring_sigmas
                )
                self.fitness_values = self.evaluate_population(self.population)
            else:
                # (μ+λ)-ES: select from parents + offspring
                combined_pop = np.vstack([self.population, offspring_pop])
                combined_sigmas = np.concatenate([self.sigmas, offspring_sigmas])
                combined_fitness = np.concatenate([self.fitness_values, offspring_fitness])
                
                self.population, self.sigmas = self.select_parents(
                    combined_pop, combined_fitness, combined_sigmas
                )
                self.fitness_values = self.evaluate_population(self.population)
            
            # Update statistics
            current_best_fitness = np.min(self.fitness_values)
            current_mean_fitness = np.mean(self.fitness_values)
            
            self.best_fitness_history.append(current_best_fitness)
            self.mean_fitness_history.append(current_mean_fitness)
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[np.argmin(self.fitness_values)].copy()
            
            self.generations_run = generation + 1
            
            # Verbose output
            if verbose and generation % 10 == 0:
                print(f"Gen {generation}: Best={current_best_fitness:.6f}, Mean={current_mean_fitness:.6f}")
            
            # Check convergence
            if current_best_fitness <= self.params.target_fitness:
                if verbose:
                    print(f"Converged at generation {generation}")
                break
        
        end_time = time.time()
        
        return {
            'best_fitness': self.best_fitness,
            'best_solution': self.best_solution,
            'generations': self.generations_run,
            'function_evaluations': self.function_evaluations,
            'time': end_time - start_time,
            'best_history': self.best_fitness_history,
            'mean_history': self.mean_fitness_history,
            'converged': self.best_fitness <= self.params.target_fitness
        }