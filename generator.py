from operation import Operation
from jobshop import JobShopInstance

import random
import json

def generate_instances(seed = 400):
    random.seed(seed)

    instance_list = []
    for job_number in [5,10,20,40,60]:
        for op_per_job in [5,10,15,20,25]:
            for m in [5,10,20]:
                for i in range(30):
                    jobs = []
                    for j in range(job_number):
                        op = []
                        for k in range(op_per_job):
                            op.append(Operation(machine_id=random.randint(0, m-1), duration=random.randint(1, 100)))
                        jobs.append(op)
                    instance = JobShopInstance(jobs=jobs)
                    instance_list.append(instance)
                    
    return instance_list

import random

def generate_general_instances(
    num_instances=100,
    num_jobs_range=(5, 60),
    num_machines_range=(5, 20),
    num_op_range=(5, 25),
    seed=400
):
    random.seed(seed)
    instance_list = []

    for _ in range(num_instances):
        num_jobs = random.randint(*num_jobs_range)
        num_machines = random.randint(*num_machines_range)

        jobs = []

        for _ in range(num_jobs):
            n_op = random.randint(*num_op_range)
            operations = []

            for _ in range(n_op):
                operations.append(
                    Operation(
                        machine_id=random.randint(0, num_machines - 1),
                        duration=random.randint(1, 100)
                    )
                )

            jobs.append(operations)

        instance = JobShopInstance(jobs=jobs)
        instance_list.append(instance)

    return instance_list

# Function to save instances to JSON
def save_instances_to_json(instances, file_path):
    """
    Save job shop instances to a JSON file.

    Args:
        instances: List of JobShopInstance objects.
        file_path: Path to the JSON file.
    """
    data = []
    for instance in instances:
        jobs_data = []
        for job in instance.jobs:
            job_data = []
            for op in job:
                job_data.append({
                    'machine_id': op.machine_id,
                    'duration': op.duration
                })
            jobs_data.append(job_data)
        data.append({'jobs': jobs_data})

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Function to load instances from JSON
def load_instances_from_json(file_path):
    """
    Load job shop instances from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        List of JobShopInstance objects.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    instances = []
    for instance_data in data:
        jobs = []
        for job_data in instance_data['jobs']:
            job = []
            for op_data in job_data:
                job.append(Operation(
                    machine_id=op_data['machine_id'],
                    duration=op_data['duration']
                ))
            jobs.append(job)
        instances.append(JobShopInstance(jobs=jobs))

    return instances

if __name__ == "__main__":
    # Example usage
    instances = generate_instances()
    save_instances_to_json(instances, "instances.json")
    loaded_instances = load_instances_from_json("instances.json")
    print(f"Saved and loaded {len(loaded_instances)} instances successfully.")
