# Job Scheduling Optimizer - Genetic Algorithm

A Python-based project for optimizing job scheduling across multiple machines using Genetic Algorithm approach.

## Project Status

**Completed:**
- Job class with `__init__` and `__repr__` methods
- `create_sample_jobs()` function for generating random jobs
- Test cases for functionality verification

**In Development:**
- Chromosome representation
- Fitness function
- Genetic Algorithm operators (selection, crossover, mutation)
- Result visualization

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
Makespan = 50 min

Optimal distribution:
Machine 0: Job 0 = 10 min
Machine 1: Job 1 = 25 min
Machine 2: Job 2 = 15 min
Makespan = 25 min
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/job-scheduler-ga.git
cd job-scheduler-ga

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
```bash
# Run the main script
python job_scheduler_ga.py
```

### Expected Output
```
============================================================
Testing Job Scheduling - Step 1: Job Class
============================================================

1. Creating jobs manually:
  Job(id=0, time=10)
  Job(id=1, time=25)
  Job(id=2, time=15)

2. Generating jobs automatically:
  Created 10 jobs:
    Job(id=0, time=23)
    Job(id=1, time=14)
    ...

3. Total processing time:
  If all jobs run one after another: 167 time units
  If we have 3 machines, best case: ~55.7 time units
```

## Project Structure
```
job-scheduler-ga/
├── data/                       # Input data files (optional)
├── results/                    # Output results and visualizations
├── venv/                       # Virtual environment (not tracked)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── job_scheduler_ga.py        # Main implementation
```

## Technologies

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and random number generation
- **Matplotlib**: Data visualization and result plotting
- **Plotly** (optional): Interactive visualizations

## Genetic Algorithm Components

The project implements the following GA components:

1. **Chromosome Encoding**: Job-to-machine assignment representation
2. **Fitness Function**: Makespan calculation (lower is better)
3. **Selection**: Tournament selection for parent selection
4. **Crossover**: Single-point crossover for generating offspring
5. **Mutation**: Random machine reassignment for diversity
6. **Elitism**: Preservation of best solutions across generations

## Development

This project is developed incrementally for educational purposes. Each component is thoroughly documented with inline comments explaining the implementation details.

### Running Tests
```bash
python job_scheduler_ga.py
```

Currently, tests are embedded in the main script using the `if __name__ == "__main__"` block.

## Contributing

This is an educational project. Feel free to fork and experiment with different GA parameters and improvements.

## License

This project is created for educational purposes.

## Author

Educational Project - Genetic Algorithm for Job Scheduling Optimization

## Roadmap

- [x] Basic Job class implementation
- [x] Sample job generator
- [ ] Chromosome representation
- [ ] Fitness calculation
- [ ] GA operators (selection, crossover, mutation)
- [ ] Evolution loop
- [ ] Result visualization
- [ ] CSV input support
- [ ] Performance metrics
- [ ] Unit tests