# Job Scheduling Optimizer - Genetic Algorithm

A Python-based project for optimizing job scheduling across multiple machines using Genetic Algorithm approach. This project demonstrates how evolutionary algorithms can solve complex optimization problems efficiently.

## Project Status

**Completed:**
- Job class with `__init__` and `__repr__` methods
- `create_sample_jobs()` function for generating random jobs
- `calculate_makespan()` function for evaluating solutions
- `calculate_fitness()` function (inverse of makespan)
- Chromosome representation (job-to-machine assignment array)
- Population initialization with random solutions
- Population evaluation (batch fitness calculation)
- Tournament selection for parent selection
- Selection pressure analysis and demonstration
- Manual optimization examples demonstrating 63.4% improvement
- Random search comparison (100 solutions)
- Comprehensive test cases with detailed output

**In Development:**
- Crossover operator (single-point crossover)
- Mutation operator (random reassignment)
- Elitism (preserve best solutions)
- Main evolution loop (integrate all operators)
- Result visualization (fitness evolution, Gantt chart)
- CSV input support for custom job datasets

## Problem Description

The Job Scheduling Problem involves assigning N jobs to M machines to minimize the makespan (total completion time). Each job has a specific processing time, and machines can process jobs in parallel.

**Example:**
```
Jobs: Job 0 (10 min), Job 1 (25 min), Job 2 (15 min)
Machines: 3

Poor distribution:
Machine 0: Jobs 0,1,2 = 50 min
Machine 1: (empty) = 0 min
Machine 2: (empty) = 0 min
Makespan = 50 min (worst case)

Optimal distribution:
Machine 0: Job 0 = 10 min
Machine 1: Job 1 = 25 min
Machine 2: Job 2 = 15 min
Makespan = 25 min (best case)
```

**Makespan** = Maximum completion time across all machines (the bottleneck machine).

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)
- Virtual environment support

### Setup Instructions

1. **Clone the repository**
git clone https://github.com/BaykushevI/job-scheduler-ga.git
cd job-scheduler-ga

2. **Create virtual environment**
python -m venv venv
venv\Scripts\activate

3. **Install dependencies**
pip install --upgrade pip
pip install -r requirements.txt

4. **Verify installation**
python -c "import numpy; import matplotlib; print('All dependencies installed successfully!')"

## Usage

### Running the Program
# Ensure virtual environment is activated
python job_scheduler_ga.py


### Expected Output

The program demonstrates:
1. Manual job creation and automatic generation
2. Chromosome representation examples
3. Makespan calculation for different solutions
4. Fitness conversion (1/makespan)
5. Selection probability analysis
6. Random search experiment (100 solutions)
7. Larger problem demonstration (20 jobs)
8. Tournament selection process
9. Selection pressure visualization

**Sample output:**
```
======================================================================
Testing Job Scheduling - Step 2: Chromosome and Makespan
======================================================================

1. Creating test jobs:
  Job(id=0, time=10)
  Job(id=1, time=25)
  Job(id=2, time=15)
  Job(id=3, time=20)
  Job(id=4, time=12)

  Solution 4: Optimal distribution
  Chromosome: [2 0 1 2 1]
  Machine loads:
    Machine 0: J1 = 25 time units
    Machine 1: J2, J4 = 27 time units
    Machine 2: J0, J3 = 30 time units
  Makespan: 30 time units

  Best solution: 30 time units
  Improvement from worst: 63.4%
```

## Current Implementation

### 1. Chromosome Representation

Solutions are encoded as NumPy arrays where each index represents a job and the value represents the assigned machine.

**Example:**
```python
# 5 jobs, 3 machines
chromosome = np.array([2, 0, 1, 2, 1])

# Interpretation:
# Job 0 → Machine 2
# Job 1 → Machine 0
# Job 2 → Machine 1
# Job 3 → Machine 2
# Job 4 → Machine 1
```

**Why this encoding?**
- Simple and intuitive
- Easy to crossover (cut and combine)
- Easy to mutate (change single value)
- Compact representation

### 2. Makespan Calculation

Makespan is the maximum completion time across all machines.
```python
def calculate_makespan(chromosome, jobs, num_machines):
    machine_loads = np.zeros(num_machines)  # Initialize loads
    
    for job_idx, machine_idx in enumerate(chromosome):
        machine_loads[machine_idx] += jobs[job_idx].processing_time
    
    makespan = np.max(machine_loads)  # Bottleneck machine
    return makespan
```

**Example:**
```
Chromosome: [0, 1, 2, 0, 1]
Jobs: [10, 25, 15, 20, 12]

Machine 0: Jobs 0,3 = 10+20 = 30
Machine 1: Jobs 1,4 = 25+12 = 37  ← Maximum
Machine 2: Job 2 = 15

Makespan = 37 (Machine 1 is the bottleneck)
```

### 3. Fitness Function

Fitness is the inverse of makespan to convert minimization to maximization.
```python
fitness = 1.0 / makespan
```

**Why inverse?**
- Genetic Algorithms maximize fitness (higher = better)
- We want to minimize makespan (lower = better)
- Using `1/makespan` aligns both goals

**Example:**
```
Solution A: makespan=30 → fitness=0.0333 (better)
Solution B: makespan=50 → fitness=0.0200 (worse)

In selection:
Solution A: 33.3% chance of being selected
Solution B: 20.0% chance of being selected
```

### 4. Population Initialization

Creates diverse starting population of random solutions.
```python
population = np.random.randint(0, num_machines, size=(population_size, num_jobs))
```

**Example with 100 solutions for 5 jobs:**
```
Population shape: (100, 5)
[[0, 2, 1, 0, 2],  ← Solution 1
 [1, 0, 2, 1, 0],  ← Solution 2
 ...
 [0, 0, 1, 2, 1]]  ← Solution 100
```

### 5. Tournament Selection

Selects parents for next generation based on fitness.

**Process:**
1. Randomly pick 3 individuals (tournament size)
2. Compare their fitness values
3. Winner (highest fitness) becomes parent
4. Repeat to fill parent pool

**Key benefit:** Balances exploitation (choosing best) with exploration (diversity).
```python
# Example tournament:
Candidates: Solution 5 (fitness=0.028), Solution 23 (0.031), Solution 67 (0.022)
Winner: Solution 23 (highest fitness)
```

**Selection pressure:**
- Better solutions: Selected more often (but not always)
- Weaker solutions: Still have small chance (maintains diversity)

## Project Structure
```
job-scheduler-ga/
├── data/                       # Input data files (optional)
├── results/                    # Output results and visualizations
├── venv/                       # Virtual environment (not tracked)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── job_scheduler_ga.py        # Main implementation (current: 450+ lines)
```

## Key Algorithms Implemented

### Current Components

1. **Job Class**
   - Represents individual jobs with ID and processing time
   - Pretty printing with `__repr__`

2. **Chromosome Encoding**
   - Array-based representation of job assignments
   - Enables efficient GA operations

3. **Fitness Evaluation**
   - Makespan calculation
   - Inverse fitness for maximization
   - Batch evaluation for entire population

4. **Tournament Selection**
   - Probabilistic parent selection
   - Configurable selection pressure
   - Maintains population diversity

### Upcoming Components

5. **Crossover (Next)**
   - Single-point crossover
   - Combines parent solutions

6. **Mutation**
   - Random gene changes
   - Prevents premature convergence

7. **Evolution Loop**
   - Integrates all operators
   - Iterative improvement over generations

## Technologies

- **Python 3.8+**: Core programming language
- **NumPy 1.21+**: Numerical computations, random generation, array operations
- **Matplotlib 3.4+**: Data visualization and result plotting
- **Plotly 5.0+** (optional): Interactive visualizations

### NumPy Functions Used

- `np.random.randint()`: Generate random chromosomes
- `np.random.choice()`: Tournament selection
- `np.zeros()`: Initialize machine loads
- `np.max()`: Calculate makespan
- `np.mean()`: Average fitness
- `np.argmax()`: Find best solution index
- `np.array()`: Array creation
- `np.random.seed()`: Reproducible results

## Performance Benchmarks

### Small Problem (5 jobs, 3 machines)
- **Manual optimization**: 30 time units (optimal)
- **Random search (100 attempts)**: 30 time units (lucky)
- **Worst case**: 82 time units
- **Improvement**: 63.4%

### Larger Problem (20 jobs, 5 machines)
- **Random search (100 attempts)**: 95 time units
- **Ideal (perfect balance)**: 69.4 time units
- **Gap from ideal**: 37.4%
- **Conclusion**: Random search insufficient for larger problems

**This motivates the need for evolutionary approach!**

## Development

This project is developed incrementally for educational purposes. Each component is thoroughly documented with inline comments explaining implementation details.

### Code Style
- Extensive comments in English
- Docstrings for all functions
- Type hints for clarity
- Educational focus with examples

### Testing
Current tests embedded in main script using `if __name__ == "__main__"` pattern.

**Test coverage:**
- Job creation and representation
- Makespan calculation (4 test cases)
- Fitness conversion
- Random search (100 solutions)
- Population initialization
- Tournament selection (visual demonstration)
- Selection pressure analysis

## Educational Goals

This project demonstrates:
1. How to encode optimization problems for GA
2. Why inverse fitness is needed for minimization problems
3. How selection pressure affects evolution
4. Difference between random search and evolutionary search
5. Importance of population diversity
6. NumPy for efficient computations

## Contributing

This is an educational project. Feel free to fork and experiment with:
- Different selection methods (roulette wheel, rank selection)
- Varying tournament sizes
- Alternative chromosome encodings
- Multi-objective optimization
- Constraint handling

## Future Enhancements

- [ ] Complete GA implementation (crossover, mutation, evolution)
- [ ] Visualization of evolution progress
- [ ] Gantt chart for final schedule
- [ ] CSV file input for custom datasets
- [ ] Command-line arguments for parameters
- [ ] Performance comparison with other algorithms
- [ ] Unit tests with pytest
- [ ] GUI interface (optional)
- [ ] Parallel evaluation (multiprocessing)

## License

This project is created for educational purposes.

## Author

Educational Project - Genetic Algorithm for Job Scheduling Optimization

**Repository:** https://github.com/BaykushevI/job-scheduler-ga

## Acknowledgments

- Genetic Algorithm concepts from evolutionary computation theory
- NumPy documentation for efficient array operations
- Tournament selection from standard GA literature

---

**Last Updated:** October 2025  
**Status:** Active Development - Tournament Selection Complete