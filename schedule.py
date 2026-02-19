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
    
    def makespan(self):
        # Calculate the makespan of the schedule
        max_completion_time = 0
        for job in self.job_shop_instance.jobs:
            for op in job:
                start_time = self.get_operation_start_time(op.operation_id)
                if start_time is not None:
                    completion_time = start_time + op.duration
                    max_completion_time = max(max_completion_time, completion_time)
        return max_completion_time
    
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

    def plot_all_results():
        # Open gnn_results.txt and random_results.txt, read lines, extract makespan values, and plot them as line chart
        # X-axis: instance number, Y-axis: makespan, two lines for random vs GNN
        gnn_makespans = []
        random_makespans = []
        
        # Read GNN results
        try:
            with open("gnn_results.txt", "r") as f:
                for line in f:
                    if "Makespan" in line:
                        makespan = int(line.split("Makespan:")[1].strip())
                        gnn_makespans.append(makespan)
        except FileNotFoundError:
            print("Warning: gnn_results.txt not found")
            
        # Read Random results  
        try:
            with open("random_results.txt", "r") as f:
                for line in f:
                    if "Makespan" in line:
                        makespan = int(line.split("Makespan:")[1].strip())
                        random_makespans.append(makespan)
        except FileNotFoundError:
            print("Warning: random_results.txt not found")
        
        if not gnn_makespans and not random_makespans:
            print("No results found to plot")
            return
            
        # Create line chart
        plt.figure(figsize=(12, 7))
        
        # Determine the maximum number of instances to plot
        max_instances = max(len(gnn_makespans), len(random_makespans))
        instance_numbers = range(1, max_instances + 1)
        
        # Plot GNN results
        if gnn_makespans:
            gnn_instances = range(1, len(gnn_makespans) + 1)
            plt.plot(gnn_instances, gnn_makespans, 'b-o', label='GNN Scheduler', linewidth=2, markersize=6, alpha=0.8)
        
        # Plot Random results
        if random_makespans:
            random_instances = range(1, len(random_makespans) + 1)
            plt.plot(random_instances, random_makespans, 'r-s', label='Random Scheduler', linewidth=2, markersize=6, alpha=0.8)
        
        plt.xlabel('Instance Number', fontsize=12)
        plt.ylabel('Makespan', fontsize=12)
        plt.title('Makespan Comparison: GNN vs Random Scheduler', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Add some styling
        plt.tight_layout()
        
        # Show statistics if both results are available
        if gnn_makespans and random_makespans:
            min_len = min(len(gnn_makespans), len(random_makespans))
            gnn_subset = gnn_makespans[:min_len]
            random_subset = random_makespans[:min_len]
            
            avg_improvement = sum(r - g for r, g in zip(random_subset, gnn_subset)) / min_len
            print(f"Average makespan improvement (Random - GNN): {avg_improvement:.2f}")
            print(f"GNN better in {sum(1 for r, g in zip(random_subset, gnn_subset) if g < r)}/{min_len} instances")
        
        plt.show()

if __name__ == "__main__":
    Schedule.plot_all_results()