# Job Shop Scheduler

This project is about solving job shop scheduling problems with graph neural networks and reinforcement learning.

At a high level, it includes:

- code for representing job shop instances and schedules
- scripts for generating datasets
- GNN-based scheduling models
- training and evaluation code
- a random baseline for comparison
- utilities for saving results and plotting them

## Main Idea

The scheduler treats each instance as a graph of operations and learns to choose which operation should be scheduled next.

The repository is mainly for experimentation and comparison between different scheduling approaches, especially GNN-based ones.

## Main Files

- [new_ppo.py](/Users/baharbarghbani/Documents/Uni/embedded/project/new_ppo.py): main training logic
- [GNN.py](/Users/baharbarghbani/Documents/Uni/embedded/project/GNN.py): base GNN model
- [muxGNN.py](/Users/baharbarghbani/Documents/Uni/embedded/project/muxGNN.py): multiplex GNN model
- [generator.py](/Users/baharbarghbani/Documents/Uni/embedded/project/generator.py): dataset generation and loading
- [random_scheduler.py](/Users/baharbarghbani/Documents/Uni/embedded/project/random_scheduler.py): random baseline
- [plot.py](/Users/baharbarghbani/Documents/Uni/embedded/project/plot.py): plotting utilities

## Setup

Dependencies are listed in [req.txt](/Users/baharbarghbani/Documents/Uni/embedded/project/req.txt).

A typical setup looks like:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r req.txt
```

## Usage

Common things you may want to do:

- generate or load job shop instances
- train a model
- compare a trained model with the random scheduler
- plot the results

The exact commands may change over time, so it is best to check the relevant script before running it.

## Output

The project may produce:

- model checkpoints
- text files with evaluation results
- plot images for comparisons

## Notes

This repository looks like an experiment/project workspace, so some scripts may be older than others. The main idea and structure are stable, but smaller implementation details may change as the project evolves.
