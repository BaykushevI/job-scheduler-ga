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