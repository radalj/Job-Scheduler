# compare_3_models_barcharts.py
import re
import os
import argparse
from collections import defaultdict
import math

import matplotlib.pyplot as plt

# دقیقا مطابق generator شما
JOBS_LIST = [5, 10, 20, 40, 60]
OPS_LIST  = [5, 10, 15, 20, 25]
M_LIST    = [5, 10, 20]
REPS = 30

TOTAL_EXPECTED = len(JOBS_LIST) * len(OPS_LIST) * len(M_LIST) * REPS


def parse_makespans(path: str):
    """Extract makespan ints from lines containing 'Makespan: <int>'."""
    pat = re.compile(r"Makespan:\s*(\d+)")
    ms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                ms.append(int(m.group(1)))
    return ms


def idx_to_params(idx: int):
    """
    Map flat index -> (jobs, ops_per_job, machines, rep)
    Using the same nesting:
      for job in JOBS_LIST:
        for ops in OPS_LIST:
          for m in M_LIST:
            for rep in range(30):
    """
    reps_block = REPS
    m_block = len(M_LIST) * reps_block
    ops_block = len(OPS_LIST) * m_block

    j_i = idx // ops_block
    rem = idx % ops_block

    op_i = rem // m_block
    rem = rem % m_block

    m_i = rem // reps_block
    rep = rem % reps_block

    return JOBS_LIST[j_i], OPS_LIST[op_i], M_LIST[m_i], rep


def bucket_results(makespans):
    """buckets[(jobs, ops, m)] = list of makespans (up to 30)."""
    buckets = defaultdict(list)
    for idx, ms in enumerate(makespans):
        jobs, ops, m, _rep = idx_to_params(idx)
        buckets[(jobs, ops, m)].append(ms)
    return buckets


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def grouped_bar_chart(x_labels, series, title, xlabel, ylabel, out_path):
    """
    x_labels: list[str] or list[int]
    series: list of (name, y_values_list)
    """
    n_groups = len(x_labels)
    n_series = len(series)

    plt.figure()

    # bar positions
    x = list(range(n_groups))
    total_width = 0.8
    bar_w = total_width / max(1, n_series)
    start = -total_width / 2 + bar_w / 2

    for i, (name, y) in enumerate(series):
        pos = [xi + start + i * bar_w for xi in x]
        plt.bar(pos, y, width=bar_w, label=name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, [str(v) for v in x_labels])
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_series_for_jobs(buckets_by_model, model_names, fixed_m=10, fixed_ops=10):
    # x = JOBS_LIST
    series = []
    for name in model_names:
        b = buckets_by_model[name]
        y = [mean(b.get((j, fixed_ops, fixed_m), [])) for j in JOBS_LIST]
        series.append((name, y))
    return JOBS_LIST, series


def compute_series_for_machines(buckets_by_model, model_names, fixed_jobs=20, fixed_ops=10):
    # x = M_LIST
    series = []
    for name in model_names:
        b = buckets_by_model[name]
        y = [mean(b.get((fixed_jobs, fixed_ops, m), [])) for m in M_LIST]
        series.append((name, y))
    return M_LIST, series


def compute_series_for_ops(buckets_by_model, model_names, fixed_jobs=20, fixed_m=10):
    # x = OPS_LIST
    series = []
    for name in model_names:
        b = buckets_by_model[name]
        y = [mean(b.get((fixed_jobs, ops, fixed_m), [])) for ops in OPS_LIST]
        series.append((name, y))
    return OPS_LIST, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1", required=True, help="Result txt for model 1")
    ap.add_argument("--model2", required=True, help="Result txt for model 2")
    ap.add_argument("--model3", required=True, help="Result txt for model 3")
    ap.add_argument("--name1", default="Model-1")
    ap.add_argument("--name2", default="Model-2")
    ap.add_argument("--name3", default="Model-3")
    ap.add_argument("--outdir", default="plots_compare")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model_files = {
        args.name1: args.model1,
        args.name2: args.model2,
        args.name3: args.model3,
    }

    buckets_by_model = {}
    for name, path in model_files.items():
        ms = parse_makespans(path)
        if not ms:
            raise SystemExit(f"No makespans found in {path}")

        print(f"{name}: parsed {len(ms)} makespans (expected full grid = {TOTAL_EXPECTED})")
        if len(ms) < TOTAL_EXPECTED:
            print(f"  Warning: {name} seems partial; means will be computed from available reps.")
        buckets_by_model[name] = bucket_results(ms)

    model_names = [args.name1, args.name2, args.name3]

    # 1) jobs vary, m=10, ops=10  (همون چیزی که گفتی: 5/10/20/... با 10,10 ثابت)
    x, series = compute_series_for_jobs(buckets_by_model, model_names, fixed_m=10, fixed_ops=10)
    grouped_bar_chart(
        x_labels=x,
        series=series,
        title="Mean Makespan vs #Jobs (m=10, ops/job=10)",
        xlabel="#Jobs",
        ylabel="Mean Makespan",
        out_path=os.path.join(args.outdir, "bar_compare_jobs_m10_ops10.png"),
    )

    # 2) machines vary, jobs=20, ops=10
    x, series = compute_series_for_machines(buckets_by_model, model_names, fixed_jobs=20, fixed_ops=10)
    grouped_bar_chart(
        x_labels=x,
        series=series,
        title="Mean Makespan vs #Machines (jobs=20, ops/job=10)",
        xlabel="#Machines",
        ylabel="Mean Makespan",
        out_path=os.path.join(args.outdir, "bar_compare_machines_j20_ops10.png"),
    )

    # 3) ops/job vary, jobs=20, m=10
    x, series = compute_series_for_ops(buckets_by_model, model_names, fixed_jobs=20, fixed_m=10)
    grouped_bar_chart(
        x_labels=x,
        series=series,
        title="Mean Makespan vs Ops/Job (jobs=20, m=10)",
        xlabel="Ops per Job",
        ylabel="Mean Makespan",
        out_path=os.path.join(args.outdir, "bar_compare_ops_j20_m10.png"),
    )

    print("Done. Saved charts in:", args.outdir)


if __name__ == "__main__":
    main()