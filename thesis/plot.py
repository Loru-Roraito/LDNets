"""Plots of the runs made by run_experiments.py.   python plot_experiments.py
Figure 1: skill against the training-mean baseline (validation and test), one dot per seed, dash = mean over the seeds.
Figure 2: training and validation loss of every run, one colour per setting.
Both are also saved as png in thesis/model/runs/. Needs run_experiments.py next to it."""
import os
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use(os.environ.get("LDNETS_BACKEND", "TkAgg"))
import matplotlib.pyplot as plt

from runall import CONFIGS, RUNS

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # one colour per setting, in the order of CONFIGS
color_of = {name: COLORS[i % len(COLORS)] for i, name in enumerate(CONFIGS)}
INK, MUTED = "#0b0b0b", "#52514e"

runs = []
for path in sorted(RUNS.glob("*/results.json")):
    results = json.loads(path.read_text())
    name = path.parent.name
    history = np.load(path.parent / "training_history.npz")
    runs.append(dict(name=name, config=name.rsplit("_s", 1)[0], results=results, history=history))
if not runs:
    raise SystemExit(f"no finished runs in {RUNS}: run run_experiments.py first")
configs = [c for c in CONFIGS if any(r["config"] == c for r in runs)]

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": "#e3e2dc", "grid.linewidth": 0.6})

# ---- figure 1: skill against the baseline ----
fig1, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, split in zip(axs, ("valid", "test")):
    for k, config in enumerate(configs):
        values = np.array([r["results"]["report"][split]["vs_mean"] for r in runs if r["config"] == config])
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(k + jitter, values, s=48, color=color_of[config], edgecolor="white", linewidth=1.5, zorder=3)
        ax.hlines(values.mean(), k - 0.3, k + 0.3, color=color_of[config], linewidth=2.5, zorder=2)
        ax.text(k + 0.34, values.mean(), f"{values.mean():.2f}", va="center", ha="left", fontsize=9, color=MUTED)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(range(len(configs)), configs, rotation=20)
    ax.set_xlim(-0.6, len(configs) - 0.2)
    ax.set_title(f"{split} years", loc="left", fontsize=11)
axs[0].set_ylabel("1 - MSE(model) / MSE(training-mean baseline)\n0 = same as the baseline, higher is better")
fig1.suptitle("Skill against the training-mean baseline (dot = seed, dash = mean)", x=0.01, ha="left", fontsize=12)
fig1.tight_layout()
fig1.savefig(RUNS / "experiments_skill.png", dpi=150)

# ---- figure 2: loss curves ----
fig2, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, key, title in zip(axs, ("loss_train_history", "loss_valid_history"), ("training loss", "validation loss")):
    for r in runs:
        h = r["history"]
        ax.loglog(np.maximum(h["iterations_history"], 1), h[key], color=color_of[r["config"]], linewidth=1.4, alpha=0.9)
    ax.axvline(max(int(runs[0]["history"]["num_epochs_Adam"]), 1), color=MUTED, linestyle="--", linewidth=0.8)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("epochs (dashed: Adam | BFGS)")
axs[0].set_ylabel("loss")
for k, config in enumerate(configs):
    axs[1].plot([], [], color=color_of[config], linewidth=2, label=config)
axs[1].legend(frameon=False, fontsize=9, loc="lower left")
fig2.suptitle("Loss of every run (one colour per setting)", x=0.01, ha="left", fontsize=12)
fig2.tight_layout()
fig2.savefig(RUNS / "experiments_loss.png", dpi=150)
print(f"saved: {RUNS / 'experiments_skill.png'}\n       {RUNS / 'experiments_loss.png'}")
plt.show()