from operation import Operation
from jobshop import JobShopInstance

import random

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

if __name__ == "__main__":
    instances = generate_general_instances()
    print(f"Generated {len(instances)} instances.")

    print("\nSample instance:")
    if instances:
        print(instances[0].jobs)