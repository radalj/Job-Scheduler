import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Schedule:
    """
    Represents a schedule for a job shop instance.
    """
    def __init__(self, job_shop_instance):
        self.job_shop_instance = job_shop_instance
        self.schedule = {}  # Dictionary to hold the schedule (operation_id: start_time)

    def add_operation(self, operation_id, start_time):
        self.schedule[operation_id] = start_time

    def get_operation_start_time(self, operation_id):
        return self.schedule.get(operation_id, None)

    def __str__(self):
        return f"Schedule for {self.job_shop_instance.name}: {self.schedule}"
    
    def plot(self):
        # Plot the schedule using a Gantt chart for each machine
        machine_schedules = {}
        
        # Generate colors for each job
        colors = list(mcolors.TABLEAU_COLORS.values())
        if self.job_shop_instance.num_jobs > len(colors):
            # Extend colors if there are more jobs than available colors
            colors = colors * ((self.job_shop_instance.num_jobs // len(colors)) + 1)
        
        # Organize operations by machine and collect operation details
        for job in self.job_shop_instance.jobs:
            for op in job:
                machine_id = op.machine_id
                if machine_id not in machine_schedules:
                    machine_schedules[machine_id] = []
                start_time = self.get_operation_start_time(op.operation_id)
                if start_time is not None:
                    machine_schedules[machine_id].append({
                        'start_time': start_time,
                        'duration': op.duration,
                        'job_id': op.job_id,
                        'position': op.position_in_job,
                        'color': colors[op.job_id]
                    })
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for machine_id, ops in machine_schedules.items():
            for op_info in ops:
                # Draw the bar with job-specific color
                bar = ax.barh(machine_id, op_info['duration'], 
                             left=op_info['start_time'], 
                             color=op_info['color'],
                             edgecolor='black',
                             linewidth=0.5)
                
                # Add text annotation "job:i op:j"
                text = f"job:{op_info['job_id']} op:{op_info['position']}"
                ax.text(op_info['start_time'] + op_info['duration']/2, 
                       machine_id, 
                       text,
                       ha='center', 
                       va='center',
                       fontsize=8,
                       fontweight='bold')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine ID')
        ax.set_title('Job Shop Schedule Gantt Chart')
        ax.set_yticks(range(self.job_shop_instance.num_machines))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()