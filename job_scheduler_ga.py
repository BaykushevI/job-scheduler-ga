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
    
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Job Scheduling - Step 1: Job Class")
    print("=" * 60)
    
    # Test 1: Manual job creation
    print("\n1. Creating jobs manually:")
    job1 = Job(job_id=0, processing_time=10)
    job2 = Job(job_id=1, processing_time=25)
    job3 = Job(job_id=2, processing_time=15)
    
    print(f"  {job1}")
    print(f"  {job2}")
    print(f"  {job3}")
    
    # Test 2: Automatic job generation
    print("\n2. Generating jobs automatically:")
    np.random.seed(42)  # For reproducible results
    jobs = create_sample_jobs(num_jobs=10, min_time=5, max_time=25)
    
    print(f"  Created {len(jobs)} jobs:")
    for job in jobs:
        print(f"    {job}")
    
    # Test 3: Calculate total time if all jobs run sequentially
    print("\n3. Total processing time:")
    total_time = sum(job.processing_time for job in jobs)
    print(f"  If all jobs run one after another: {total_time} time units")
    print(f"  If we have 3 machines, best case: ~{total_time / 3:.1f} time units")