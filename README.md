# Sampled Elite Local Search Genetic Algorithm for the p-Median Problem

A high-performance Python implementation of a Genetic Algorithm with hybrid operators for solving the p-Median facility location problem.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
- [Performance](#performance)

## 🎯 Overview

The **p-Median Problem** is a classical facility location optimization problem where the goal is to select `p` facilities from a set of `n` candidates to minimize the total distance between clients and their nearest facility.

This project implements a sophisticated Genetic Algorithm with:
- Unique combination-based population generation
- Tournament and roulette selection methods
- Specialized crossover operators preserving feasibility
- Local search mutation for elite individuals
- Adaptive stopping criteria based on coefficient of variation
- Vectorized fitness evaluation using NumPy

## ✨ Features

- **Efficient Data Structures**: Custom `Combinaciones` class using Python sets for O(1) uniqueness checking
- **Vectorized Operations**: NumPy-based fitness evaluation for all individuals simultaneously
- **Hybrid Mutation**: Simple mutation for population diversity + local search for elite individuals
- **Smart Stopping Criteria**: CV-based convergence detection to avoid unnecessary iterations
- **Hyperparameter Tuning**: Automated optimization using Hyperopt with checkpoint support
- **Batch Testing**: Run multiple replicas across all benchmark instances with statistical reporting

## 📁 Project Structure

```
├── combinaciones.py           # Unique combination generator and manager
├── ExtractData.py            # OR-Library format parser and Floyd-Warshall
├── Genetic_pMP.py            # Core GA operators for p-Median
├── GeneticAlgorithm_V2.py    # Generic GA framework
├── funtion_pMP.py            # High-level wrapper for pMP GA
├── HiperparametrizacionV2.py # Hyperopt-based parameter tuning
├── test_runner.py            # Batch experiment runner
├── pmed/                     # Raw OR-Library dataset files (.txt)
├── datasets/                 # Processed distance matrices (.npy)
└── resultados/               # Output directory for Excel results
```

## 🔧 Installation

### Requirements

```bash
pip install numpy numba hyperopt openpyxl pandas
```

### Python Version

Python 3.8 or higher recommended.

## 🚀 Usage

### 1. Prepare Datasets

Convert OR-Library format files to NumPy matrices:

```python
from ExtractData import generar_datasets_pmedian

# Reads all .txt files from pmed/ and saves matrices to datasets/
generar_datasets_pmedian("pmed", "datasets")
```

### 2. Run a Single Instance

```python
import numpy as np
from funtion_pMP import genetic_algorithm_pMP
from Genetic_pMP import selecciona_torneo, cruzamiento_intercambio, mutacion_simple

# Load distance matrix
cost_matrix = np.load('datasets/pmed1.npy')
p = 5  # Number of facilities to select

# Execute GA
tiempo, mejor_individuo, mejor_fitness = genetic_algorithm_pMP(
    cost_matrix=cost_matrix,
    p=p,
    num_iteraciones=500,
    pop_size=250,
    seleccion=selecciona_torneo,
    cruzamiento=cruzamiento_intercambio,
    mutacion=mutacion_simple,
    para_seleccion={},
    para_cruzamiento={},
    para_mutacion={},
    prob_cruzamiento=0.4,
    prob_mutacion=0.6,
    usar_criterio_parada_cv=True,
    sample_frac=0.37,
    frac_mejores=0.2,
    umbral_cv=0.0001,
    min_gener=50,
    maximizar=False
)

print(f"Best fitness: {mejor_fitness}")
print(f"Selected facilities: {mejor_individuo}")
print(f"Execution time: {tiempo} minutes")
```

### 3. Run Batch Experiments

Execute multiple replicas across all benchmark instances:

```bash
python test_runner.py
```

Results are saved to `resultados/resultados_GA_pmed.xlsx` with statistics (best, mean, median, std, avg time).

### 4. Hyperparameter Optimization

```bash
python HiperparametrizacionV2.py
```

Uses TPE (Tree-structured Parzen Estimator) to search optimal parameters. Progress is checkpointed to `hyperopt_trials_pmed.pkl`.

## 📊 Datasets

The project uses benchmark instances from the [OR-Library p-Median collection](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/pmedinfo.html):

| Instance | Nodes (n) | Facilities (p) | Known Optimum |
|----------|-----------|----------------|---------------|
| pmed1    | 100       | 5              | 5819          |
| pmed5    | 100       | 33             | 1255          |
| pmed10   | 200       | 67             | 1255          |
| pmed25   | 500       | 167            | 1828          |
| pmed40   | 900       | 90             | 5128          |

## ⚙️ Configuration

### Key Parameters

**Population & Iterations:**
- `pop_size`: Population size (50-500 recommended)
- `num_iteraciones`: Maximum generations (500-1000 typical)

**Genetic Operators:**
- `seleccion`: Selection method (`'selecciona_torneo'` or `'ruleta'`)
- `cruzamiento`: Crossover (`'cruzamiento_intercambio'`)
- `mutacion`: Standard mutation (`'mutacion_simple'` or `'mutation_local_search_sample'`)
- `prob_cruzamiento`: Crossover probability (0.4-0.95)
- `prob_mutacion`: Mutation probability (0.1-0.6)

**Elite Mutation (always applied to best individual):**
- `sample_frac`: Fraction of neighborhood to sample (0.2-0.5)

**Stopping Criteria:**
- `usar_criterio_parada_cv`: Enable CV-based stopping
- `frac_mejores`: Top fraction to compute CV (0.1-0.3)
- `umbral_cv`: CV threshold for convergence (0.0001-0.005)
- `min_gener`: Minimum generations before stopping (50-100)

## 🔬 Hyperparameter Optimization

The `HiperparametrizacionV2.py` script searches over:

```python
space = {
    'tam': [10, 200],                    # Population size
    'seleccion': ['selecciona_torneo', 'ruleta'],
    'prob_mutacion': [0.1, 0.95],
    'prob_cruzamiento': [0.15, 0.975],
    'num_competidores': [4, 20],         # Tournament size
}
```

Evaluates each configuration across 10 test instances and optimizes the average gap from known optima.

## 📈 Results

Results are exported to Excel with columns:

- `test`: Instance name
- `mejor_fitness`: Best solution across replicas
- `mean_fitness`: Average solution quality
- `median_fitness`: Median solution
- `std_fitness`: Standard deviation
- `tiempo_promedio_min`: Average runtime in minutes

## ⚡ Performance

**Optimizations:**
- Numba JIT compilation for selection operators
- Vectorized fitness evaluation (O(n×p) per generation)
- Set-based unique population management
- Efficient NumPy broadcasting for distance calculations

**Typical Performance:**
- Small instances (n=100): < 1 minute
- Medium instances (n=200-300): 1-5 minutes
- Large instances (n=500-900): 5-30 minutes

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional crossover operators (PMX, OX)
- Parallel population evaluation
- GPU acceleration with CuPy
- More sophisticated local search heuristics
- Integration with CPLEX/Gurobi for comparison

## 📝 License

This project is provided as-is for academic and research purposes.

## 📚 References

- OR-Library: [https://people.brunel.ac.uk/~mastjjb/jeb/orlib/pmedinfo.html](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/pmedinfo.html)

- Hyperopt: [https://github.com/hyperopt/hyperopt](https://github.com/hyperopt/hyperopt)

## 👤 Authors

- Fermín Rivero Sotelo
- Nelson Montes Villalba
- Jorge Lopez Pereira
- Helman Hernández Riaño

---

