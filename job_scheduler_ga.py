"""
Job Scheduling Optimizer using Genetic Algorithm
=================================================
This module implements a Genetic Algorithm to optimize job scheduling.
"""
import numpy as np
from typing import List
class Job:
    """
    Represents a single job with an ID and processing time.
    
    Attributes:
        job_id (int): Unique identifier for the job
        processing_time (int): Time required to complete the job
    
    Example:
        job1 = Job(job_id=0, processing_time=10)
        print(job1.job_id)          # Output: 0
        print(job1.processing_time) # Output: 10
    """
    
    def __init__(self, job_id: int, processing_time: int):
        # Store the job ID and processing time
        self.job_id = job_id
        self.processing_time = processing_time
        
    def __repr__(self):
        """Returns a string representation of the Job instance."""
        return f"Job(id={self.job_id}, time={self.processing_time})"
    
def create_sample_jobs(num_jobs: int = 10, min_time: int = 5, max_time: int = 30) -> List['Job']:
    """
    Generate sample jobs with random processing times.

    This function creates multiple Job objects at once with random
    processing times within the specified range.
    
    Args:
        num_jobs (int): Number of jobs to create (default: 10)
        min_time (int): Minimum processing time (default: 5)
        max_time (int): Maximum processing time (default: 30)
        
    Returns:
        List[Job]: List of generated Job objects
        
    Example:
        # Create 5 jobs with times between 10 and 50
        jobs = create_sample_jobs(num_jobs=5, min_time=10, max_time=50)
        
        # Might generate:
        # [Job(id=0, time=23), Job(id=1, time=45), Job(id=2, time=12), ...]
    """
    jobs = [] # Empty list to store jobs

    # Loop num_jobs times
    for i in range(num_jobs):
        # Generate random processing time between min_time and max_time (inclusive)
        processing_time = np.random.randint(min_time, max_time + 1)
        # Create a Job with id = 1 and the random processing time
        job = Job(job_id=i, processing_time=processing_time)
        # Add to our list
        jobs.append(job)
    return jobs

def calculate_makespan(chromosome: np.ndarray, jobs: List[Job], num_machines: int) -> float:
    """
    Calculate the makespan (total completion time) for a given chromosome.
    
    The makespan is the maximum completion time across all machines.
    Lower makespan = better solution.
    
    Args:
        chromosome (np.ndarray): Job-to-machine assignment array
                                 Example: [0, 2, 1, 0, 2] means:
                                 Job 0 -> Machine 0
                                 Job 1 -> Machine 2
                                 Job 2 -> Machine 1
                                 Job 3 -> Machine 0
                                 Job 4 -> Machine 2
        jobs (List[Job]): List of Job objects
        num_machines (int): Number of available machines
        
    Returns:
        float: Makespan value (maximum machine load)
        
    Example:
        jobs = [Job(0, 10), Job(1, 25), Job(2, 15)]
        chromosome = [0, 1, 2]  # Each job on different machine
        makespan = calculate_makespan(chromosome, jobs, 3)
        # Result: 25 (Machine 1 has the longest job)
    """
    # Initialize machine loads (cumulative time for each machine)
    # Example: [0, 0, 0] for 3 machines - all start empty
    machine_loads = np.zeros(num_machines)

    # For each job, add its processing time to the assigned machine
    for job_idx, machine_idx in enumerate(chromosome):
        # job_idx: index in chromosome (0, 1, 2, ...)
        # machine_idx: which machine this job is assigned to
        
        # Get the job's processing time
        job_processing_time = jobs[job_idx].processing_time

        # Add this time to the machine's total load
        machine_loads[machine_idx] += job_processing_time

    # Makespan is the maximum load (the busiest machine determines total time)
    makespan = np.max(machine_loads)
    
    return makespan

def initialize_population(num_jobs: int, num_machines: int, population_size: int) -> np.ndarray:
    """
    Create initial random population of chromosomes.
    
    Each chromosome is a random assignment of jobs to machines.
    This creates diverse starting solutions for the GA to evolve.
    
    Args:
        num_jobs (int): Number of jobs to schedule
        num_machines (int): Number of available machines
        population_size (int): Number of solutions in the population
        
    Returns:
        np.ndarray: Population matrix of shape (population_size, num_jobs)
                    Each row is a chromosome (solution)
                    
    Example:
        # 5 jobs, 3 machines, population of 4
        population = initialize_population(5, 3, 4)
        
        # Might generate:
        # [[0, 2, 1, 0, 2],  ← Solution 1
        #  [1, 0, 2, 1, 0],  ← Solution 2
        #  [2, 1, 0, 2, 1],  ← Solution 3
        #  [0, 0, 1, 2, 1]]  ← Solution 4
    """
    # Generate random integers between 0 and num_machines-1
    # Shape: (population_size rows, num_jobs columns)
    population = np.random.randint(
        low = 0,  # Minimum machine ID (inclusive)
        high = num_machines,  # Maximum machine ID (exclusive, so we use num_machines)
        size = (population_size, num_jobs)    # Matrix dimensions
    )

    return population

def evaluate_population(population: np.ndarray, jobs: List[Job], num_machines: int) -> np.ndarray:
    """
    Calculate fitness for all chromosomes in the population.
    
    This function evaluates how good each solution is by calculating
    its fitness value (1/makespan).
    
    Args:
        population (np.ndarray): Population matrix (population_size, num_jobs)
        jobs (List[Job]): List of Job objects
        num_machines (int): Number of available machines
        
    Returns:
        np.ndarray: Array of fitness values, one for each chromosome
        
    Example:
        population = [[0, 2, 1, 0, 2], [1, 0, 2, 1, 0]]
        fitness_values = evaluate_population(population, jobs, 3)
        # Returns: [0.0270, 0.0333] (example values)
    """
    # Create empty array to store fitness values
    fitness_values = np.zeros(len(population))

    # calculate fitness for each chromosome
    for i, chromosome in enumerate(population):
        fitness_values[i] = calculate_fitness(chromosome, jobs, num_machines)

    return fitness_values

def calculate_fitness(chromosome: np.ndarray, jobs: List[Job], num_machines: int) -> float:
    """
    Calculate the fitness value for a given chromosome.
    
    Fitness is the inverse of makespan (1/makespan).
    Higher fitness = better solution.
    
    Why inverse?
    - Our goal: minimize makespan (lower is better)
    - GA goal: maximize fitness (higher is better)
    - Solution: fitness = 1/makespan converts minimization to maximization
    
    Args:
        chromosome (np.ndarray): Job-to-machine assignment array
        jobs (List[Job]): List of Job objects
        num_machines (int): Number of available machines
        
    Returns:
        float: Fitness value (1/makespan)
        
    Example:
        Solution A: makespan = 30 → fitness = 1/30 = 0.0333 (better)
        Solution B: makespan = 50 → fitness = 1/50 = 0.0200 (worse)
    """
    # Calculate makespan for this chromosome
    makespan = calculate_makespan(chromosome, jobs, num_machines)
    
    # Convert to fitness (inverse relationship)
    # Lower makespan → Higher fitness
    fitness = 1.0 / makespan
    
    return fitness

#########TESTS######################   
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Job Scheduling - Step 2: Chromosome and Makespan")
    print("=" * 70)
    
    # Create some test jobs
    print("\n1. Creating test jobs:")
    jobs = [
        Job(job_id=0, processing_time=10),
        Job(job_id=1, processing_time=25),
        Job(job_id=2, processing_time=15),
        Job(job_id=3, processing_time=20),
        Job(job_id=4, processing_time=12)
    ]
    
    for job in jobs:
        print(f"  {job}")
    
    total_time = sum(job.processing_time for job in jobs)
    print(f"\n  Total processing time: {total_time} time units")
    
    # Test different chromosomes (solutions)
    print("\n" + "=" * 70)
    print("2. Testing different solutions (chromosomes):")
    print("=" * 70)
    
    num_machines = 3
    
    # Solution 1: All jobs on Machine 0 (WORST CASE)
    print("\n  Solution 1: All jobs on Machine 0")
    chromosome1 = np.array([0, 0, 0, 0, 0])
    print(f"  Chromosome: {chromosome1}")
    makespan1 = calculate_makespan(chromosome1, jobs, num_machines)
    print(f"  Machine loads:")
    
    # Calculate and display load per machine
    loads1 = np.zeros(num_machines)
    for job_idx, machine_idx in enumerate(chromosome1):
        loads1[machine_idx] += jobs[job_idx].processing_time
    
    for i, load in enumerate(loads1):
        jobs_on_machine = [j for j, m in enumerate(chromosome1) if m == i]
        job_ids = ', '.join([f"J{j}" for j in jobs_on_machine])
        print(f"    Machine {i}: {job_ids if job_ids else '(empty)'} = {load:.0f} time units")
    
    print(f"  Makespan: {makespan1:.0f} time units")
    
    # Solution 2: Balanced distribution
    print("\n  Solution 2: Balanced distribution")
    chromosome2 = np.array([0, 1, 2, 0, 1])
    print(f"  Chromosome: {chromosome2}")
    makespan2 = calculate_makespan(chromosome2, jobs, num_machines)
    print(f"  Machine loads:")
    
    loads2 = np.zeros(num_machines)
    for job_idx, machine_idx in enumerate(chromosome2):
        loads2[machine_idx] += jobs[job_idx].processing_time
    
    for i, load in enumerate(loads2):
        jobs_on_machine = [j for j, m in enumerate(chromosome2) if m == i]
        job_ids = ', '.join([f"J{j}" for j in jobs_on_machine])
        print(f"    Machine {i}: {job_ids if job_ids else '(empty)'} = {load:.0f} time units")
    
    print(f"  Makespan: {makespan2:.0f} time units")
    
    # Solution 3: Your turn to create!
    print("\n  Solution 3: Different distribution")
    chromosome3 = np.array([2, 0, 1, 2, 0])
    print(f"  Chromosome: {chromosome3}")
    makespan3 = calculate_makespan(chromosome3, jobs, num_machines)
    print(f"  Machine loads:")
    
    loads3 = np.zeros(num_machines)
    for job_idx, machine_idx in enumerate(chromosome3):
        loads3[machine_idx] += jobs[job_idx].processing_time
    
    for i, load in enumerate(loads3):
        jobs_on_machine = [j for j, m in enumerate(chromosome3) if m == i]
        job_ids = ', '.join([f"J{j}" for j in jobs_on_machine])
        print(f"    Machine {i}: {job_ids if job_ids else '(empty)'} = {load:.0f} time units")
    
    print(f"  Makespan: {makespan3:.0f} time units")

    # Solution 4: Optimal distribution (found manually)
    print("\n  Solution 4: Optimal distribution")
    chromosome4 = np.array([2, 0, 1, 2, 1])
    print(f"  Chromosome: {chromosome4}")
    makespan4 = calculate_makespan(chromosome4, jobs, num_machines)
    print(f"  Machine loads:")
    
    loads4 = np.zeros(num_machines)
    for job_idx, machine_idx in enumerate(chromosome4):
        loads4[machine_idx] += jobs[job_idx].processing_time
    
    for i, load in enumerate(loads4):
        jobs_on_machine = [j for j, m in enumerate(chromosome4) if m == i]
        job_ids = ', '.join([f"J{j}" for j in jobs_on_machine])
        print(f"    Machine {i}: {job_ids if job_ids else '(empty)'} = {load:.0f} time units")
    
    print(f"  Makespan: {makespan4:.0f} time units")

    # Compare solutions
    print("\n" + "=" * 70)
    print("3. Comparison:")
    print("=" * 70)
    print(f"  Solution 1 (all on one machine): {makespan1:.0f} time units")
    print(f"  Solution 2 (balanced):            {makespan2:.0f} time units")
    print(f"  Solution 3 (different):           {makespan3:.0f} time units")
    print(f"  Solution 4 (optimal):             {makespan4:.0f} time units")
    
    best_makespan = min(makespan1, makespan2, makespan3, makespan4)
    print(f"\n  Best solution: {best_makespan:.0f} time units")
    print(f"  Improvement from worst: {((makespan1 - best_makespan) / makespan1 * 100):.1f}%")

    # Test fitness calculation
    print("\n" + "=" * 70)
    print("4. Fitness Calculation (for Genetic Algorithm):")
    print("=" * 70)
    print("\n  Why fitness = 1/makespan?")
    print("  - GA maximizes fitness (higher = better)")
    print("  - We want to minimize makespan (lower = better)")
    print("  - Solution: Use inverse (1/makespan)")
    print()
    
    # Calculate fitness for all solutions
    fitness1 = calculate_fitness(chromosome1, jobs, num_machines)
    fitness2 = calculate_fitness(chromosome2, jobs, num_machines)
    fitness3 = calculate_fitness(chromosome3, jobs, num_machines)
    fitness4 = calculate_fitness(chromosome4, jobs, num_machines)
    
    print("  Makespan → Fitness conversion:")
    print(f"    Solution 1: makespan={makespan1:.0f} → fitness={fitness1:.6f}")
    print(f"    Solution 2: makespan={makespan2:.0f} → fitness={fitness2:.6f}")
    print(f"    Solution 3: makespan={makespan3:.0f} → fitness={fitness3:.6f}")
    print(f"    Solution 4: makespan={makespan4:.0f} → fitness={fitness4:.6f} ← HIGHEST (best)")
    
    print("\n  Notice: Lower makespan = Higher fitness!")
    print(f"  Best solution has fitness {fitness4:.6f} (makespan {makespan4:.0f})")
    
    # Show selection probability (simplified)
    print("\n" + "=" * 70)
    print("5. Selection Probability (simplified example):")
    print("=" * 70)
    
    total_fitness = fitness1 + fitness2 + fitness3 + fitness4
    prob1 = (fitness1 / total_fitness) * 100
    prob2 = (fitness2 / total_fitness) * 100
    prob3 = (fitness3 / total_fitness) * 100
    prob4 = (fitness4 / total_fitness) * 100
    
    print("\n  In GA, better solutions have higher chance to be selected:")
    print(f"    Solution 1: {prob1:.1f}% chance")
    print(f"    Solution 2: {prob2:.1f}% chance")
    print(f"    Solution 3: {prob3:.1f}% chance")
    print(f"    Solution 4: {prob4:.1f}% chance ← HIGHEST (best solution)")
    
    print("\n  This is how GA evolves toward better solutions!")

    # Random Search Experiment
    print("\n" + "=" * 70)
    print("6. Random Search Experiment (100 random solutions):")
    print("=" * 70)
    print("\n  Goal: Find best solution among 100 random attempts")
    print("  This simulates what happens WITHOUT evolution")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate 100 random solutions
    num_jobs = len(jobs)
    population_size = 100
    population = initialize_population(num_jobs, num_machines, population_size)
    
    print(f"  Generated {population_size} random solutions...")
    print(f"  Example chromosomes:")
    for i in range(5):
        print(f"    Solution {i+1}: {population[i]}")
    print(f"    ...")
    print()
    
    # Evaluate all solutions
    fitness_values = evaluate_population(population, jobs, num_machines)
    
    # Find best solution
    best_idx = np.argmax(fitness_values)  # Index of highest fitness
    best_chromosome = population[best_idx]
    best_fitness = fitness_values[best_idx]
    best_makespan = 1.0 / best_fitness  # Convert back to makespan
    
    # Find worst solution
    worst_idx = np.argmin(fitness_values)
    worst_chromosome = population[worst_idx]
    worst_fitness = fitness_values[worst_idx]
    worst_makespan = 1.0 / worst_fitness
    
    # Calculate statistics
    avg_fitness = np.mean(fitness_values)
    avg_makespan = 1.0 / avg_fitness
    
    print("  Results:")
    print(f"    Best solution found:")
    print(f"      Chromosome: {best_chromosome}")
    print(f"      Makespan: {best_makespan:.0f} time units")
    print(f"      Fitness: {best_fitness:.6f}")
    
    # Show machine distribution for best solution
    loads_best = np.zeros(num_machines)
    for job_idx, machine_idx in enumerate(best_chromosome):
        loads_best[machine_idx] += jobs[job_idx].processing_time
    
    print(f"      Machine loads:")
    for i, load in enumerate(loads_best):
        jobs_on_machine = [j for j, m in enumerate(best_chromosome) if m == i]
        job_ids = ', '.join([f"J{j}" for j in jobs_on_machine])
        print(f"        Machine {i}: {job_ids if job_ids else '(empty)'} = {load:.0f} time units")
    
    print()
    print(f"    Worst solution found:")
    print(f"      Makespan: {worst_makespan:.0f} time units")
    
    print()
    print(f"    Average makespan: {avg_makespan:.1f} time units")
    
    # Compare with our manual optimal solution
    print("\n" + "=" * 70)
    print("7. Random Search vs Manual Optimization:")
    print("=" * 70)
    print(f"    Manual optimal solution (Solution 4): {makespan4:.0f} time units")
    print(f"    Best random solution (from 100):      {best_makespan:.0f} time units")
    
    if best_makespan <= makespan4:
        print(f"\n    Random search found equal or better solution!")
        print(f"    Difference: {makespan4 - best_makespan:.0f} time units")
    else:
        print(f"\n    Manual solution is still better!")
        print(f"    Difference: {best_makespan - makespan4:.0f} time units worse")
    
    print(f"\n    Note: With only 5 jobs, random search can get lucky.")
    print(f"    With 50+ jobs, random search fails completely!")
    print(f"    That's why we need Genetic Algorithm evolution.")

    # Experiment with larger problem
    print("\n" + "=" * 70)
    print("8. Larger Problem: Random Search Struggles")
    print("=" * 70)
    
    # Create larger problem
    np.random.seed(123)
    large_jobs = create_sample_jobs(num_jobs=20, min_time=5, max_time=30)
    large_num_machines = 5
    
    print(f"\n  Problem: {len(large_jobs)} jobs, {large_num_machines} machines")
    print(f"  Total processing time: {sum(j.processing_time for j in large_jobs)} time units")
    print(f"  Ideal makespan (if perfectly balanced): {sum(j.processing_time for j in large_jobs) / large_num_machines:.1f} time units")
    print()
    
    # Try random search
    large_population = initialize_population(len(large_jobs), large_num_machines, population_size=100)
    large_fitness = evaluate_population(large_population, large_jobs, large_num_machines)
    
    large_best_idx = np.argmax(large_fitness)
    large_best_makespan = 1.0 / large_fitness[large_best_idx]
    large_avg_makespan = 1.0 / np.mean(large_fitness)
    
    print(f"  Random Search Results (100 attempts):")
    print(f"    Best makespan found: {large_best_makespan:.0f} time units")
    print(f"    Average makespan: {large_avg_makespan:.0f} time units")
    
    ideal_makespan = sum(j.processing_time for j in large_jobs) / large_num_machines
    gap = large_best_makespan - ideal_makespan
    print(f"\n    Gap from ideal: {gap:.0f} time units ({(gap/ideal_makespan*100):.1f}%)")
    print(f"\n    This is where Genetic Algorithm will shine!")
    print(f"    GA will evolve these 100 solutions over 200 generations")
    print(f"    and find MUCH better solutions!")