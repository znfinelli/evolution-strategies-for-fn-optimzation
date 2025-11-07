# Evolution Strategies for Function Optimization

**Master in Artificial Intelligence - Evolutionary Computation**  
**Date:** November 2025

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Algorithm Details](#-algorithm-details)
- [Experimental Design](#-experimental-design)
- [Results & Visualization](#-results--visualization)
- [Customization Guide](#-customization-guide)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)

---

## 🚀 Quick Start

### 1. Installation

**Recommended: Using Virtual Environment**
```bash
# Create a lightweight venv in the project folder
python3 -m venv .venv

# Activate the virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

**Alternative: Global Installation (not recommended)**
```bash
pip install numpy matplotlib pandas
```

### 2. Run Experiments
```bash
# Ensure virtual environment is active
python main.py
```

### 3. Check Results

Results will be generated in the `outputs/` directory:
- `results.csv` - Detailed results for each run
- `summary_statistics.csv` - Aggregated statistics
- `convergence_sphere.png` - Convergence plot for Sphere function
- `convergence_rastrigin.png` - Convergence plot for Rastrigin function
- `comparison_boxplots.png` - Comparative analysis plots

**Expected Runtime:** ~5-10 minutes (depending on your CPU)

---

## 📖 Project Overview

This project implements **Evolution Strategies (ES)** for continuous function optimization. We compare **(μ,λ)-ES** and **(μ+λ)-ES** strategies on different benchmark functions with varying dimensions.

### Implemented Algorithms

- **(μ,λ)-ES:** Selection only from offspring
- **(μ+λ)-ES:** Selection from both parents and offspring
- **Self-adaptive mutation** with learning rates τ and τ'

### Benchmark Functions

| Function | Type | Difficulty | Global Minimum |
|----------|------|------------|----------------|
| **Sphere** | Unimodal | Easy | f(0,...,0) = 0 |
| **Rastrigin** | Multimodal | Hard | f(0,...,0) = 0 |
| **Rosenbrock** | Valley-shaped | Medium | f(1,...,1) = 0 |
| **Ackley** | Multimodal | Hard | f(0,...,0) = 0 |

---

## 📁 Project Structure
```
evolution_strategies/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── main.py                        # Main execution script
│
├── src/                           # Source code package
│   ├── __init__.py               # Package initialization
│   ├── es_params.py              # ES parameters dataclass
│   ├── test_functions.py         # Benchmark functions
│   ├── evolution_strategy.py     # Core ES algorithm
│   ├── experiment_runner.py      # Experiment management
│   └── visualization.py          # Plotting functions
│
└── outputs/                       # Generated results (auto-created)
    ├── results.csv
    ├── summary_statistics.csv
    ├── convergence_sphere.png
    ├── convergence_rastrigin.png
    └── comparison_boxplots.png
```

### Module Descriptions

| Module | Responsibility |
|--------|----------------|
| `es_params.py` | Defines ES configuration parameters |
| `test_functions.py` | Benchmark optimization functions |
| `evolution_strategy.py` | Core ES algorithm implementation |
| `experiment_runner.py` | Manages multiple independent runs |
| `visualization.py` | Generates plots and visualizations |
| `main.py` | Orchestrates experiments and output |

---

## 🧬 Algorithm Details

### Initialization

- **Population:** Uniformly random in search space
- **Mutation Strength (σ):** Initial value = 0.5

### Self-Adaptation Mechanism

The mutation strength adapts automatically during evolution:
```
τ = 1 / √(2n)          where n = dimension
τ' = 1 / √(2√n)

σ_new = σ × exp(τ' × N(0,1) + τ × N(0,1))
x_new = x + σ_new × N(0,I)
```

### Selection Strategies

| Strategy | Description | Advantages |
|----------|-------------|------------|
| **(μ,λ)-ES** | Select μ best from λ offspring only | More exploratory, avoids stagnation |
| **(μ+λ)-ES** | Select μ best from μ parents + λ offspring | More stable, preserves best solutions |

### Reproduction

- Each offspring created by mutating a random parent
- **No recombination** (crossover) is used in this implementation

---

## 🔬 Experimental Design

### Parameters Tested

| Parameter | Values |
|-----------|--------|
| **Functions** | Sphere, Rastrigin |
| **Dimensions** | 10, 20 |
| **Population Sizes** | μ=15-30, λ=100-200 |
| **Strategies** | comma, plus |
| **Independent Runs** | 30 per configuration |
| **Max Generations** | 500 |
| **Target Fitness** | 1e-6 |

### Performance Metrics

✅ **Best fitness achieved**  
✅ **Generations to convergence**  
✅ **Function evaluations**  
✅ **Execution time**  
✅ **Success rate** (% reaching target fitness)

---

## 📊 Results & Visualization

### Output Files

#### 1. `results.csv`
Detailed results for each individual run.

**Columns:**
- `function` - Benchmark function name
- `dimension` - Problem dimension
- `mu` - Number of parents
- `lambda` - Number of offspring
- `strategy` - Selection strategy (comma/plus)
- `run` - Run number (1-30)
- `best_fitness` - Best fitness achieved
- `generations` - Generations to convergence
- `function_evals` - Total function evaluations
- `time` - Execution time (seconds)
- `converged` - Boolean: reached target fitness

#### 2. `summary_statistics.csv`
Aggregated statistics per configuration (mean, std, min).

#### 3. Convergence Plots
- **convergence_sphere.png** - Shows fitness evolution over generations for Sphere function
- **convergence_rastrigin.png** - Shows fitness evolution for Rastrigin function
- Mean ± standard deviation across 30 runs
- Log scale on y-axis for better visualization

#### 4. `comparison_boxplots.png`
Four subplots comparing all configurations:
- Best fitness distribution
- Generations to convergence
- Function evaluations
- Success rate

### Statistical Analysis

Each configuration is run **30 times** to ensure statistical significance.

**Metrics Computed:**
- Mean and standard deviation
- Success rate (% reaching target fitness)
- Minimum achieved fitness
- Median performance

**Comparisons:**
- (μ,λ) vs (μ+λ) strategies
- Different dimensions (10 vs 20)
- Easy (Sphere) vs Hard (Rastrigin) functions
- Impact of population size

---

## 🎛️ Customization Guide

### Modify Experiment Configurations

Edit the `experiments` list in `main.py`:
```python
experiments = [
    {'func': 'sphere', 'dim': 10, 'mu': 15, 'lambda': 100, 'strategy': 'comma'},
    {'func': 'ackley', 'dim': 20, 'mu': 25, 'lambda': 150, 'strategy': 'plus'},
    # Add your custom configurations here
]
```

**Available Functions:**
- `'sphere'`
- `'rastrigin'`
- `'rosenbrock'`
- `'ackley'`

**Recommended λ/μ Ratio:** 5-10

### Change Number of Runs
```python
runner = ExperimentRunner(n_runs=30)  # Change to 10, 20, etc.
```

### Add New Benchmark Functions

Add to `src/test_functions.py`:
```python
@staticmethod
def your_function(x: np.ndarray) -> float:
    """Your custom optimization function"""
    return np.sum(x**4)  # Example

# Also add bounds in get_bounds() method
```

---

## 🔧 Troubleshooting

### Common Issues

#### ❌ `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Make sure you're in the project root directory
cd path/to/evolution_strategies/
python main.py
```

#### ❌ `ModuleNotFoundError: No module named 'numpy'`

**Solution:**
```bash
pip install -r requirements.txt
```

#### ⏱️ Experiments Running Too Slow

**Solutions:**
- Reduce `n_runs` from 30 to 10-15
- Reduce `max_generations` from 500 to 200-300
- Test on smaller dimensions first

#### 💾 Memory Error

**Solutions:**
- Reduce dimension size
- Reduce population size (μ and λ)
- Close other applications

#### 📉 Poor Convergence on Difficult Functions

**Expected Behavior:** Rastrigin and Ackley are intentionally difficult with many local optima.

**Solutions to Improve:**
- Increase `max_generations`
- Increase population size (larger λ)
- Adjust λ/μ ratio (try 7-10)
- Try different random seeds

---

## 📚 References

### Evolution Strategies

1. **Rechenberg, I. (1973).** *Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution.* Stuttgart: Frommann-Holzboog.

2. **Schwefel, H.-P. (1995).** *Evolution and Optimum Seeking.* New York: Wiley.

3. **Hansen, N., & Ostermeier, A. (2001).** "Completely derandomized self-adaptation in evolution strategies." *Evolutionary Computation*, 9(2), 159-195.

4. **Beyer, H.-G., & Schwefel, H.-P. (2002).** "Evolution strategies – A comprehensive introduction." *Natural Computing*, 1(1), 3-52.

### Benchmark Functions

5. **Jamil, M., & Yang, X. S. (2013).** "A literature survey of benchmark functions for global optimization problems." *International Journal of Mathematical Modelling and Numerical Optimisation*, 4(2), 150-194.

6. **Surjanovic, S., & Bingham, D.** *Virtual Library of Simulation Experiments: Test Functions and Datasets.* Retrieved from https://www.sfu.ca/~ssurjano/optimization.html

---

## 👥 Contact & Submission

**Master in Artificial Intelligence**  
**Course:** Evolutionary Computation - Practical Work  
**Submission Deadline:** November 10, 2025

For technical questions about the implementation, refer to:
- Inline documentation in source code
- Module docstrings
- This README

---

## 📝 License

This project is developed for educational purposes as part of the Master in AI curriculum.

---

## ✨ Features

✅ **Modular Design** - Clean separation of concerns  
✅ **Easy to Extend** - Add new functions or strategies easily  
✅ **Well-Documented** - Comprehensive docstrings and comments  
✅ **Statistical Rigor** - 30 independent runs per configuration  
✅ **Professional Visualizations** - Publication-quality plots  
✅ **Reproducible** - Fixed random seed for consistency

---

**Happy Experimenting! 🧪🧬**