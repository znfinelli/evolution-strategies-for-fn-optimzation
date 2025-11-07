from dataclasses import dataclass


@dataclass
class ESParams:
    """Parameters for Evolution Strategy"""
    mu: int  # Number of parents
    lambda_: int  # Number of offspring
    dim: int  # Problem dimension
    sigma: float  # Initial mutation strength
    tau: float  # Learning rate for sigma adaptation
    tau_prime: float  # Global learning rate
    max_generations: int  # Maximum generations
    target_fitness: float  # Target fitness to stop
    strategy: str  # 'comma' for (μ,λ) or 'plus' for (μ+λ)