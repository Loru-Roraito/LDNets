"""Trains several settings, several seeds each, and collects the results in two tables.

    python run_experiments.py                      # all configs below, seeds 0 1 2
    python run_experiments.py --seeds 0 1 --configs no_profile pca5
    python run_experiments.py --summary            # only rebuild the tables from the finished runs
    python run_experiments.py --force              # train again even where results.json exists

Every run is a separate "python train.py" (so the GPU memory is freed), in thesis/model/runs/<config>_s<seed>/ with:
trained_model/, training_history.npz, results.json, config.json and train.log.
A finished run is skipped, so after an interruption you can just start it again.
Look at a run with:  LDNETS_RUN=pca5_s0 python visualize.py
Tables: thesis/model/runs/experiments_runs.csv (one row per run) and experiments_summary.csv (mean and std over the seeds)."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "model" / "runs"

# name: settings that differ from the defaults in common.py (num_latent_states, N_PROFILE_POINTS, PROFILE_FEATURES)
CONFIGS = {
    "no_profile": dict(N_PROFILE_POINTS=0),
    "points5": dict(N_PROFILE_POINTS=5, PROFILE_FEATURES="points"),
    "points11": dict(N_PROFILE_POINTS=11, PROFILE_FEATURES="points"),
    "pca3": dict(N_PROFILE_POINTS=3, PROFILE_FEATURES="pca"),
    "pca5": dict(N_PROFILE_POINTS=5, PROFILE_FEATURES="pca"),
}
SEEDS = [0, 1, 2]


def train(config_name, seed, force):
    run = f"{config_name}_s{seed}"
    run_dir = RUNS / run
    if (run_dir / "results.json").exists() and not force:
        print(f"[skip] {run}: already done")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").unlink(missing_ok=True)
    settings = CONFIGS[config_name]
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2))
    env = dict(os.environ, LDNETS_RUN=run, LDNETS_SEED=str(seed))
    env.update({f"LDNETS_{key}": str(value) for key, value in settings.items()})
    print(f"[run ] {run}  {settings}  ->  {run_dir / 'train.log'}", flush=True)
    start = time.time()
    with open(run_dir / "train.log", "w") as log:
        code = subprocess.run([sys.executable, "train.py"], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[{'done' if code == 0 else 'FAIL'}] {run}  {(time.time() - start) / 60:.1f} min" + ("" if code == 0 else "  (see train.log)"), flush=True)


def summarize(show=True):
    rows = []
    for path in sorted(RUNS.glob("*/results.json")):
        r = json.loads(path.read_text())
        rep = r["report"]
        rows.append({
            "run": path.parent.name, "config": path.parent.name.rsplit("_s", 1)[0], "seed": r["seed"],
            "best_valid_loss": r["best_valid_loss"], "final_valid_loss": r["final_valid_loss"], "final_train_loss": r["final_train_loss"],
            "train_vs_mean": rep["train"]["vs_mean"], "valid_vs_mean": rep["valid"]["vs_mean"], "test_vs_mean": rep["test"]["vs_mean"],
            "valid_vs_zero": rep["valid"]["vs_zero"], "test_vs_zero": rep["test"]["vs_zero"],
            "test_top_vs_zero": rep["test"]["top_vs_zero"], "minutes": r["minutes"],
        })
    if not rows:
        if show:
            print("no finished runs yet")
        return
    runs = pd.DataFrame(rows)
    runs.to_csv(RUNS / "experiments_runs.csv", index=False)
    metrics = ["final_valid_loss", "train_vs_mean", "valid_vs_mean", "test_vs_mean", "test_vs_zero", "test_top_vs_zero"]
    grouped = runs.groupby("config")[metrics]
    summary = pd.concat([grouped.mean().add_suffix("_mean"), grouped.std().add_suffix("_std"), runs.groupby("config").size().rename("seeds")], axis=1)
    summary.to_csv(RUNS / "experiments_summary.csv")
    if not show:
        return
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:8.3f}")
    print("\nmean over the seeds (std in the second table); vs_mean > 0 = better than the training-mean baseline")
    print(summary[[c + "_mean" for c in metrics] + ["seeds"]].to_string())
    print(summary[[c + "_std" for c in metrics]].to_string())
    print(f"\nsaved: {RUNS / 'experiments_runs.csv'}\n       {RUNS / 'experiments_summary.csv'}")
    print("Choose the setting with valid_vs_mean, then look at test_vs_mean once. A difference smaller than the std between seeds is noise.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if not args.summary:
        for seed in args.seeds:                      # seeds in the outer loop: if you stop early, every config has the same seeds
            for name in args.configs:
                train(name, seed, args.force)
                summarize(show=False)          # keeps the csv files up to date while it runs
    summarize()