#!/usr/bin/env python3
"""Three-panel figure for docs/hw3-report: accuracy per seed, E2 per-channel gate, accuracy vs points.

Reads only results/hw3/summary.json and results/hw3/diagnostics.json (no data, no checkpoints):

    python scripts/hw3/report_figure.py            # writes results/hw3/report_figure.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results" / "hw3"
ORDER = ("e0_max", "e1_fixed_dual", "e2_gated_dual", "e3_max_h69")
SHORT = {"e0_max": "E0 max", "e1_fixed_dual": "E1 fixed", "e2_gated_dual": "E2 gated", "e3_max_h69": "E3 h=69"}
COLORS = {"e0_max": "#5b6b7f", "e1_fixed_dual": "#e07b00", "e2_gated_dual": "#2a6fdb", "e3_max_h69": "#8a5cc2"}


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    diagnostics = json.loads((RESULTS / "diagnostics.json").read_text())
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig, (ax_acc, ax_gate, ax_pts) = plt.subplots(1, 3, figsize=(7.2, 1.95), constrained_layout=True,
                                                  gridspec_kw={"width_ratios": [1, 1.15, 1]})

    for x, key in enumerate(ORDER):
        values = 100 * np.asarray(summary["conditions"][key]["test_accuracy"]["values"])
        ax_acc.scatter(x + np.array([-0.1, 0, 0.1]), values, s=10, color=COLORS[key], zorder=3)
        ax_acc.hlines(values.mean(), x - 0.25, x + 0.25, color=COLORS[key], lw=1.6)
    ax_acc.set_xticks(range(len(ORDER)), [SHORT[k].split()[0] for k in ORDER])
    ax_acc.set(ylabel="test accuracy (%)", title="(a) Test accuracy: seeds (dots), mean (bar)")
    ax_acc.grid(axis="y", alpha=0.3)

    initial = next(iter(diagnostics["gates"].values()))["initial_gate"]
    for seed in range(3):
        gate = diagnostics["gates"][f"e2_gated_dual_seed{seed}"]["test"]["per_channel_mean"]
        ax_gate.plot(np.sort(gate)[::-1], lw=1.1, label=f"seed {seed}")
    ax_gate.axhline(initial, color="#888", ls=":", lw=1, label=f"init {initial:.3f}")
    ax_gate.set(xlabel="channel (sorted)", ylabel="mean gate g over test objects", ylim=(0, 1),
                title="(b) E2 gate per channel (0 = max, 1 = mean)")
    ax_gate.grid(alpha=0.3)
    ax_gate.legend(frameon=False, loc="upper right")

    robustness = diagnostics["aggregates"]["point_count_robustness"]
    for key in ORDER:
        counts = sorted(robustness[key], key=int)
        means = [100 * robustness[key][n]["test_accuracy"]["mean"] for n in counts]
        sds = [100 * robustness[key][n]["test_accuracy"]["std"] for n in counts]
        ax_pts.errorbar([int(n) for n in counts], means, yerr=sds, marker="o", ms=2.5, lw=1, capsize=2,
                        color=COLORS[key], label=SHORT[key].split()[0])
    ax_pts.set_xscale("log", base=2)
    ax_pts.set_xticks([256, 512, 1024], ["256", "512", "1024"])
    ax_pts.set(xlabel="points per test object (no retraining)", ylabel="test accuracy (%)",
               title="(c) Fewer input points (mean ± SD)")
    ax_pts.grid(alpha=0.3)
    ax_pts.legend(frameon=False, ncols=2, loc="lower right")
    fig.savefig(RESULTS / "report_figure.png", dpi=220)
    print(RESULTS / "report_figure.png")


if __name__ == "__main__":
    main()
