## compare per instance results of random vs muxgnn and random vs gnn and muxgnn vs gnn and plot them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

def parse_makespans(filename):
    makespans = []
    with open(filename, 'r') as f:
        for line in f:
            if "Makespan:" in line:
                try:
                    val = float(line.split("Makespan:")[1].strip())
                    makespans.append(val)
                except ValueError:
                    continue
    return makespans

files = [
    'random_results.txt',
    'small_gnn_result.txt',
    'muxGNN_result.txt',
]

results = {}
for file in files:
    if os.path.exists(file):
        results[file] = parse_makespans(file)
    else:
        print(f"Warning: {file} not found.")

# Check if we have data
if not results:
    print("No result files found or parsed.")
    exit()

# Ensure all lists are the same length for direct comparison if possible
# Or just plot what we have.
# Assuming they correspond to the same instances in order.

plt.figure(figsize=(12, 6))

# Plot Makespans per Instance
for label, data in results.items():
    plt.plot(data, label=label.replace('.txt', '').replace('_', ' '), alpha=0.7)

plt.xlabel('Instance Index')
plt.ylabel('Makespan')
plt.title('Makespan Comparison per Instance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('makespan_comparison_line.png')
print("Saved makespan_comparison_line.png")

# Box Plot for Distribution
plt.figure(figsize=(10, 6))
labels = [label.replace('.txt', '').replace('_', ' ') for label in results.keys()]
data_list = [results[file] for file in files if file in results]

plt.boxplot(data_list, labels=labels)
plt.ylabel('Makespan')
plt.title('Makespan Distribution Comparison')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('makespan_comparison_boxplot.png')
print("Saved makespan_comparison_boxplot.png")

