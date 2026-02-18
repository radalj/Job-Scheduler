import random
from jobshop import JobShopInstance
from schedule import Schedule
from operation import Operation

def random_schedule(instance: JobShopInstance):
    """
    Generate a random schedule for a given JobShopInstance.
    
    Args:
        instance: JobShopInstance to schedule
    Returns:
        Schedule object with random start times for each operation
    """
    schedule = Schedule(instance)
    
    all_operations_job_ids = []

    for job_id, job in enumerate(instance.jobs):
        all_operations_job_ids.extend([job_id] * len(job))
    
    random.shuffle(all_operations_job_ids)

    machine_available_time = [0] * instance.num_machines
    job_next_op_index = [0] * instance.num_jobs

    for job_id in all_operations_job_ids:
        op_index = job_next_op_index[job_id]
        if op_index >= len(instance.jobs[job_id]):
            continue  # No more operations for this job

        operation = instance.jobs[job_id][op_index]
        machine_id = operation.machine_id
        duration = operation.duration

        # Determine the earliest start time based on machine and job constraints
        earliest_start_time = max(
            machine_available_time[machine_id],
            schedule.get_operation_start_time(instance.jobs[job_id][op_index - 1].operation_id) + instance.jobs[job_id][op_index - 1].duration if op_index > 0 else 0
        )

        # Schedule the operation
        schedule.add_operation(operation.operation_id, earliest_start_time)

        # Update machine availability and job's next operation index
        machine_available_time[machine_id] = earliest_start_time + duration
        job_next_op_index[job_id] += 1

    return schedule

if __name__ == "__main__":
    # Example usage
    instance = JobShopInstance(jobs=[
        [Operation(machine_id=0, duration=3), Operation(machine_id=1, duration=2)],
        [Operation(machine_id=1, duration=4), Operation(machine_id=0, duration=1)]
    ], name="ExampleInstance")

    schedule = random_schedule(instance)
    print(schedule)
    schedule.plot()