class JobShopInstance:
    """
    Basic Job Shop instance for RL / GNN usage.

    jobs: list[list[Operation]]
    """

    def __init__(self, jobs, name="JobShopInstance"):
        self.jobs = jobs
        self.name = name

        self._set_operation_attributes()

    # ------------------------------------------------
    # Assign unique IDs and metadata
    # ------------------------------------------------
    def _set_operation_attributes(self):
        op_id = 0
        for job_id, job in enumerate(self.jobs):
            for pos, op in enumerate(job):
                op.job_id = job_id
                op.position_in_job = pos
                op.operation_id = op_id
                op_id += 1

    # ------------------------------------------------
    # Properties
    # ------------------------------------------------
    @property
    def num_jobs(self):
        return len(self.jobs)

    @property
    def num_operations(self):
        return sum(len(job) for job in self.jobs)

    @property
    def num_machines(self):
        max_machine = 0
        for job in self.jobs:
            for op in job:
                max_machine = max(max_machine, op.machine_id)
        return max_machine + 1

    @property
    def duration_matrix(self):
        return [[op.duration for op in job] for job in self.jobs]

    @property
    def machines_matrix(self):
        return [[op.machine_id for op in job] for job in self.jobs]

    def __repr__(self):
        return (
            f"JobShopInstance("
            f"name={self.name}, "
            f"jobs={self.num_jobs}, "
            f"machines={self.num_machines}, "
            f"operations={self.num_operations})"
        )
