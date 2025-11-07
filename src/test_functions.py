import numpy as np
from typing import Tuple


class TestFunctions:
    """Collection of benchmark optimization functions"""
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """
        Sphere function: f(x) = sum(x_i^2)
        Global minimum: f(0,...,0) = 0
        """
        return np.sum(x**2)
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """
        Rastrigin function: highly multimodal
        Global minimum: f(0,...,0) = 0
        """
        n = len(x)
        A = 10
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """
        Rosenbrock function: narrow valley
        Global minimum: f(1,...,1) = 0
        """
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """
        Ackley function: multimodal with deep valleys
        Global minimum: f(0,...,0) = 0
        """
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
    
    @staticmethod
    def get_bounds(func_name: str, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get search space bounds for each function"""
        bounds = {
            'sphere': (-5.12, 5.12),
            'rastrigin': (-5.12, 5.12),
            'rosenbrock': (-2.048, 2.048),
            'ackley': (-32.768, 32.768)
        }
        lower, upper = bounds[func_name]
        return np.full(dim, lower), np.full(dim, upper)