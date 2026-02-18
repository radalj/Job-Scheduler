class Operation:
    """
    Represents a single operation in a job.

    Assumption:
        Each operation has exactly ONE machine.
    """

    def __init__(self, machine_id: int, duration: int):
        self.machine_id = machine_id
        self.duration = duration

        # These will be set by JobShopInstance
        self.job_id = None
        self.position_in_job = None
        self.operation_id = None

    def __repr__(self):
        return (
            f"Op(j={self.job_id}, "
            f"pos={self.position_in_job}, "
            f"m={self.machine_id}, "
            f"d={self.duration})"
        )
