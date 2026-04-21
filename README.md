# Job Shop Scheduler with GNN + PPO

This project explores job shop scheduling with reinforcement learning. It includes:

- job shop instance generation and JSON dataset loading
- a graph-based scheduling environment
- two policy models: a standard GNN and a multiplex GNN (`MuxGNN`)
- PPO-based training
- random-baseline comparison scripts
- result logging and plotting utilities

The codebase is centered around representing each scheduling instance as an operation graph, then learning a dispatching policy that chooses the next valid operation to schedule.

## Project Layout

- [new_ppo.py](/Users/baharbarghbani/Documents/Uni/embedded/project/new_ppo.py) contains the main environment (`JobShopEnv`), PPO trainer, checkpoint save/load flow, and the default training entry point.
- [GNN.py](/Users/baharbarghbani/Documents/Uni/embedded/project/GNN.py) defines the base graph attention network policy.
- [muxGNN.py](/Users/baharbarghbani/Documents/Uni/embedded/project/muxGNN.py) defines the multiplex GNN policy with separate precedence and machine relation channels.
- [generator.py](/Users/baharbarghbani/Documents/Uni/embedded/project/generator.py) generates synthetic job shop instances and saves/loads them as JSON.
- [jobshop.py](/Users/baharbarghbani/Documents/Uni/embedded/project/jobshop.py), [operation.py](/Users/baharbarghbani/Documents/Uni/embedded/project/operation.py), and [schedule.py](/Users/baharbarghbani/Documents/Uni/embedded/project/schedule.py) hold the core scheduling data structures.
- [random_scheduler.py](/Users/baharbarghbani/Documents/Uni/embedded/project/random_scheduler.py) provides a simple random baseline.
- [compare_muxgnn_random.py](/Users/baharbarghbani/Documents/Uni/embedded/project/compare_muxgnn_random.py) compares a trained model against the random scheduler.
- [plot.py](/Users/baharbarghbani/Documents/Uni/embedded/project/plot.py) generates comparison charts from text result files.
- [instances.json](/Users/baharbarghbani/Documents/Uni/embedded/project/instances.json) is the bundled dataset of serialized job shop instances.
- [checkpoints](/Users/baharbarghbani/Documents/Uni/embedded/project/checkpoints) contains saved model weights committed with the project.
- [plots](/Users/baharbarghbani/Documents/Uni/embedded/project/plots) contains example output figures.

## Environment and State Representation

Each job shop instance is modeled as a graph whose nodes are operations.

- precedence edges connect operations within the same job
- machine edges connect operations competing for the same machine
- node features encode duration, machine id, job id, operation position, completion status, availability, ready time, and estimated start time

The environment exposes a mask over currently valid operations, and the policy learns to pick the next operation to dispatch.

## Requirements

The repository includes pinned Python dependencies in [req.txt](/Users/baharbarghbani/Documents/Uni/embedded/project/req.txt).

Create an environment and install them with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r req.txt
```

If you want to generate plots, install `matplotlib` as well:

```bash
pip install matplotlib
```

## Quick Start

Generate or refresh the dataset:

```bash
python3 generator.py
```

Run the default training entry point:

```bash
python3 new_ppo.py
```

At the moment, running `new_ppo.py` executes the compact `train_gnn()` path by default, which trains the smaller GNN configuration and saves to `checkpoints/small_gnn_ppo.pt` unless you override the arguments.

Train a small `MuxGNN` model instead:

```bash
python3 new_ppo.py \
  --model-type muxgnn \
  --hidden-dim 32 \
  --num-heads 2 \
  --num-layers 2 \
  --checkpoint-path checkpoints/small_muxgnn_ppo.pt
```

Train on more data:

```bash
python3 new_ppo.py \
  --max-instances 100 \
  --epochs 30 \
  --checkpoint-path checkpoints/experiment.pt
```

## Comparing Against the Random Baseline

Use a trained checkpoint and compare it with the random scheduler:

```bash
python3 compare_muxgnn_random.py \
  --checkpoint checkpoints/muxgnn_full.pt \
  --instances-file instances.json
```

To generate random-only results:

```bash
python3 random_scheduler.py
```

This writes output to [random_results.txt](/Users/baharbarghbani/Documents/Uni/embedded/project/random_results.txt).

## Plotting Results

The plotting script expects three text files containing lines of the form `Makespan: <value>`.

Example:

```bash
python3 plot.py \
  --model1 random_results.txt \
  --model2 small_gnn_result.txt \
  --model3 small_muxGNN_result.txt \
  --name1 Random \
  --name2 GNN \
  --name3 MuxGNN \
  --outdir plots_compare
```

Example plots already included in the repository:

- [plots/bar_compare_jobs_m10_ops10.png](/Users/baharbarghbani/Documents/Uni/embedded/project/plots/bar_compare_jobs_m10_ops10.png)
- [plots/bar_compare_machines_j20_ops10.png](/Users/baharbarghbani/Documents/Uni/embedded/project/plots/bar_compare_machines_j20_ops10.png)
- [plots/bar_compare_ops_j20_m10.png](/Users/baharbarghbani/Documents/Uni/embedded/project/plots/bar_compare_ops_j20_m10.png)

## Checkpoints and Outputs

Typical artifacts produced by the project:

- `checkpoints/*.pt` for trained model weights and metadata
- `*_result.txt` files for evaluation logs
- `plots/*.png` for comparison figures

Saved checkpoints include metadata such as:

- model type
- node feature dimension
- hidden dimension
- number of heads
- number of layers
- learning rate
- seed

## Notes and Caveats

- The active environment implementation lives inside [new_ppo.py](/Users/baharbarghbani/Documents/Uni/embedded/project/new_ppo.py), not in a standalone `jobshop_env.py` file.
- Some older helper scripts, including [evaluate_model.py](/Users/baharbarghbani/Documents/Uni/embedded/project/evaluate_model.py) and [test_integration.py](/Users/baharbarghbani/Documents/Uni/embedded/project/test_integration.py), still import `jobshop_env`, so they need a small import cleanup before they can run as-is.
- In this workspace, dependencies were not installed at README-writing time, so the commands above were documented from code inspection rather than a full runtime verification.

## Repository Purpose

This is a compact research-style codebase for experimenting with graph neural networks on job shop scheduling. It is especially useful if you want to:

- compare plain GNN and multiplex GNN dispatching policies
- test PPO-based learning on synthetic job shop instances
- benchmark learned schedulers against a random baseline
- visualize makespan trends across jobs, machines, and operation counts
