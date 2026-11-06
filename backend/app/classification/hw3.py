"""HW3 analysis: aggregate the E0-E3 runs and diagnose the pooling.

    python -m app.classification.hw3 summarize --runs DIR --out DIR
    python -m app.classification.hw3 diagnose  --runs DIR --out DIR [--data-dir DIR]
    python -m app.classification.hw3 collect   --runs DIR --dest DIR
    python -m app.classification.hw3 probe     --out DIR [--data-dir DIR]

``--runs`` holds one folder per training run, named ``<condition>_seed<k>``
(``scripts/hw3/run_matrix.sh`` writes exactly this layout; the released copy is
``results/hw3/runs``).

- ``summarize`` reads only the saved ``metrics.json``/``history.json`` and writes
  ``summary.json``, ``summary_runs.csv``, ``summary.md`` and the result figures.
- ``diagnose`` reloads every ``best.pt`` and, on the fixed validation/test points,
  re-scores the test split (and checks it against the recorded accuracy),
  compares paired runs with an exact McNemar test, measures the E2 gate, and
  runs the two P1 diagnostics: gradient participation at the final per-point
  features, and deterministic evaluation with 512 and 256 points.
- ``collect`` copies the compact artifacts of every run into the repository.
- ``probe`` (post-hoc, no network involved) checks two alternative explanations
  for the confused pairs: how well absolute object size (raw mesh units, removed
  by the unit-sphere normalisation) and bounding-box aspect ratios (kept by the
  normalisation, perturbed by the per-axis scale augmentation) separate
  dresser/night_stand and table/desk on their own.

Nothing here trains or selects anything; the test split is only re-scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import dataset
from .geometry import normalize_points
from .models import MiniPointNet, ResidualGatedDualPool
from .training import git_dirty, git_sha, load_checkpoint

# id -> (label, model name, hidden width)
CONDITIONS = {
    "e0_max": ("E0 max pooling (HW2)", "mini_pointnet", 64),
    "e1_fixed_dual": ("E1 fixed dual (0.5 max + 0.5 mean)", "mini_pointnet_fixed_dual", 64),
    "e2_gated_dual": ("E2 residual gated dual (proposed)", "mini_pointnet_gated_dual", 64),
    "e3_max_h69": ("E3 max pooling, h=69 (parameter-matched)", "mini_pointnet", 69),
}
SHORT = {"e0_max": "E0", "e1_fixed_dual": "E1", "e2_gated_dual": "E2", "e3_max_h69": "E3"}
SEEDS = (0, 1, 2)
PAIRS = (("table", "desk"), ("desk", "table"), ("dresser", "night_stand"), ("night_stand", "dresser"))
PAIR_KEYS = tuple(f"{a}->{b}" for a, b in PAIRS)
COMPARISONS = (("e1_fixed_dual", "e0_max"), ("e2_gated_dual", "e0_max"), ("e3_max_h69", "e0_max"),
               ("e2_gated_dual", "e1_fixed_dual"), ("e2_gated_dual", "e3_max_h69"))
ARTIFACTS = ("best.pt", "config.json", "history.json", "metrics.json", "curves.png", "confusion.png")
HW2_REFERENCE = {  # stored HW2 Mini-PointNet results (backend/models/modelnet10, docs/hw2-experiments.md)
    "test_accuracy": [0.9361233480176211, 0.9229074889867841, 0.9240088105726872],
    "best_epoch": [45, 40, 49],
    "best_val_accuracy": [0.9774436090225563, 0.9774436090225563, 0.9761904761904762],
}
COLORS = {"e0_max": "#5b6b7f", "e1_fixed_dual": "#e07b00", "e2_gated_dual": "#2a6fdb", "e3_max_h69": "#8a5cc2"}


def run_dir(runs: Path, condition: str, seed: int) -> Path:
    return runs / f"{condition}_seed{seed}"


def stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "sd_sample": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max()), "values": values.tolist()}


def critical_pairs(confusion, classes: list[str]) -> dict:
    confusion = np.asarray(confusion)
    index = {name: i for i, name in enumerate(classes)}
    out = {f"{a}->{b}": int(confusion[index[a], index[b]]) for a, b in PAIRS}
    out["critical_total"] = int(sum(out[k] for k in PAIR_KEYS))
    return out


# ----------------------------------------------------------------------------- summarize

def load_runs(runs: Path) -> dict:
    records = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            folder = run_dir(runs, condition, seed)
            if not (folder / "metrics.json").exists():
                continue
            metrics = json.loads((folder / "metrics.json").read_text())
            config = json.loads((folder / "config.json").read_text())
            history = json.loads((folder / "history.json").read_text())
            records[(condition, seed)] = {"metrics": metrics, "config": config, "history": history}
    return records


def summarize(runs: Path, out: Path) -> dict:
    records = load_runs(runs)
    if not records:
        raise SystemExit(f"No runs under {runs}")
    rows = []
    for (condition, seed), record in sorted(records.items(), key=lambda kv: (list(CONDITIONS).index(kv[0][0]), kv[0][1])):
        m, c = record["metrics"], record["config"]
        rows.append({
            "condition": condition, "id": SHORT[condition], "model": m["model"], "hidden_dim": c["hidden_dim"],
            "seed": seed, "parameters": m["parameters"], "best_epoch": m["best_epoch"],
            "best_val_accuracy": m["best_val_accuracy"], "test_accuracy": m["test_accuracy"],
            "test_loss": m["test_loss"], **critical_pairs(m["confusion_matrix"], m["classes"]),
            "train_seconds": m["train_seconds"], "seconds_per_epoch": float(np.mean([r["seconds"] for r in record["history"]])),
            "git_sha": m.get("git_sha"), "git_dirty": m.get("git_dirty"),
            "host": (m.get("runtime") or {}).get("host"), "slurm_job": (m.get("runtime") or {}).get("slurm_job"),
            "threads": (m.get("runtime") or {}).get("threads"),
        })
    conditions = {}
    for condition, (label, model, hidden) in CONDITIONS.items():
        mine = [r for r in rows if r["condition"] == condition]
        if not mine:
            continue
        classes = records[(condition, mine[0]["seed"])]["metrics"]["classes"]
        recall = np.mean([[p["recall"] for p in records[(condition, r["seed"])]["metrics"]["per_class"]] for r in mine], axis=0)
        conditions[condition] = {
            "id": SHORT[condition], "label": label, "model": model, "hidden_dim": hidden,
            "parameters": mine[0]["parameters"], "seeds": [r["seed"] for r in mine],
            "test_accuracy": stats([r["test_accuracy"] for r in mine]),
            "best_val_accuracy": stats([r["best_val_accuracy"] for r in mine]),
            "best_epoch": [r["best_epoch"] for r in mine],
            "train_seconds": stats([r["train_seconds"] for r in mine]),
            "seconds_per_epoch": stats([r["seconds_per_epoch"] for r in mine]),
            **{key: stats([r[key] for r in mine]) for key in (*PAIR_KEYS, "critical_total")},
            "per_class_recall_mean": dict(zip(classes, recall.tolist())),
        }
    paired = {}
    for a, b in COMPARISONS:
        diffs = [records[(a, s)]["metrics"]["test_accuracy"] - records[(b, s)]["metrics"]["test_accuracy"]
                 for s in SEEDS if (a, s) in records and (b, s) in records]
        crit = [critical_pairs(records[(a, s)]["metrics"]["confusion_matrix"], records[(a, s)]["metrics"]["classes"])["critical_total"]
                - critical_pairs(records[(b, s)]["metrics"]["confusion_matrix"], records[(b, s)]["metrics"]["classes"])["critical_total"]
                for s in SEEDS if (a, s) in records and (b, s) in records]
        if diffs:
            paired[f"{SHORT[a]}-{SHORT[b]}"] = {"test_accuracy_diff": stats(diffs), "wins": int(sum(d > 0 for d in diffs)),
                                                "losses": int(sum(d < 0 for d in diffs)), "critical_total_diff": stats(crit)}
    reproduction = None
    if all(("e0_max", s) in records for s in SEEDS):
        e0 = [records[("e0_max", s)]["metrics"] for s in SEEDS]
        reproduction = {"hw2": HW2_REFERENCE,
                        "hw3_e0": {"test_accuracy": [m["test_accuracy"] for m in e0], "best_epoch": [m["best_epoch"] for m in e0],
                                   "best_val_accuracy": [m["best_val_accuracy"] for m in e0]}}
        reproduction["identical"] = all(np.isclose(reproduction["hw2"][k], reproduction["hw3_e0"][k]).all()
                                        for k in ("test_accuracy", "best_val_accuracy")) and \
            reproduction["hw2"]["best_epoch"] == reproduction["hw3_e0"]["best_epoch"]
    summary = {"conditions": conditions, "paired_vs": paired, "runs": rows, "e0_reproduces_hw2": reproduction,
               "generated_with": {"git_sha": git_sha(), "git_dirty": git_dirty()},
               "notes": {"std": "population SD over seeds (numpy ddof=0), as in HW2; sd_sample uses ddof=1",
                         "test_objects": 908, "one_object_pp": 100 / 908}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "summary_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.md").write_text(markdown_summary(summary))
    try:
        plot_summary(summary, records, out)
    except ImportError:
        pass
    return summary


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def markdown_summary(summary: dict) -> str:
    c = summary["conditions"]
    lines = ["# HW3 results (generated by `python -m app.classification.hw3 summarize`)", "",
             "Mean ± SD over seeds 0, 1, 2 (population SD, as in HW2). Critical-pair columns are test-set object "
             "counts, mean over seeds (per-seed values in brackets). One test object = 0.11 pp.", "",
             "| Model | Params | Val acc | Test acc | table→desk | desk→table | dresser→night | night→dresser | 4-pair total | s/epoch |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in c.items():
        def cnt(k):
            return f"{row[k]['mean']:.1f} [{', '.join(str(int(v)) for v in row[k]['values'])}]"
        lines.append(f"| {row['label']} | {row['parameters']:,} | {pct(row['best_val_accuracy']['mean'])} ± {100 * row['best_val_accuracy']['std']:.2f} "
                     f"| **{pct(row['test_accuracy']['mean'])} ± {100 * row['test_accuracy']['std']:.2f}** | "
                     + " | ".join(cnt(k) for k in (*PAIR_KEYS, "critical_total"))
                     + f" | {row['seconds_per_epoch']['mean']:.1f} |")
    lines += ["", "Per-seed test accuracy:", "", "| Model | seed 0 | seed 1 | seed 2 | selected epochs |", "|---|---:|---:|---:|---|"]
    for key, row in c.items():
        lines.append(f"| {row['id']} | " + " | ".join(pct(v) for v in row["test_accuracy"]["values"])
                     + f" | {', '.join(map(str, row['best_epoch']))} |")
    lines += ["", "Paired by seed (same batches, augmentation and — for E0/E1/E2 — the same initial shared weights):", "",
              "| Comparison | mean Δ test acc | per-seed Δ (pp) | wins/losses | Δ 4-pair errors (mean) |", "|---|---:|---|---:|---:|"]
    for name, p in summary["paired_vs"].items():
        d = p["test_accuracy_diff"]
        lines.append(f"| {name} | {100 * d['mean']:+.2f} pp | {', '.join(f'{100 * v:+.2f}' for v in d['values'])} | "
                     f"{p['wins']}/{p['losses']} | {p['critical_total_diff']['mean']:+.1f} |")
    rep = summary.get("e0_reproduces_hw2")
    if rep:
        lines += ["", f"E0 rerun vs stored HW2 Mini-PointNet: test {', '.join(pct(v) for v in rep['hw3_e0']['test_accuracy'])} vs "
                  f"{', '.join(pct(v) for v in rep['hw2']['test_accuracy'])}; selected epochs {rep['hw3_e0']['best_epoch']} vs "
                  f"{rep['hw2']['best_epoch']} — {'identical' if rep['identical'] else 'not identical'}."]
    return "\n".join(lines) + "\n"


def plot_summary(summary: dict, records: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(summary["conditions"])
    fig, (ax_acc, ax_pairs) = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True,
                                           gridspec_kw={"width_ratios": [1, 1.5]})
    for x, key in enumerate(conds):
        row = summary["conditions"][key]
        values = 100 * np.asarray(row["test_accuracy"]["values"])
        ax_acc.scatter(np.full(len(values), x) + np.linspace(-0.08, 0.08, len(values)), values, color=COLORS[key], zorder=3, s=22)
        ax_acc.errorbar(x + 0.22, values.mean(), yerr=values.std(), fmt="_", color=COLORS[key], markersize=14, capsize=3)
    ax_acc.set_xticks(range(len(conds)), [SHORT[k] for k in conds])
    ax_acc.set(ylabel="test accuracy (%)", title="Test accuracy, seeds 0–2 (dots) and mean ± SD")
    ax_acc.grid(axis="y", alpha=0.3)
    width = 0.8 / len(conds)
    for j, key in enumerate(conds):
        row = summary["conditions"][key]
        means = [row[k]["mean"] for k in PAIR_KEYS]
        sds = [row[k]["std"] for k in PAIR_KEYS]
        ax_pairs.bar(np.arange(len(PAIR_KEYS)) + (j - (len(conds) - 1) / 2) * width, means, width, yerr=sds,
                     color=COLORS[key], label=SHORT[key], capsize=2)
    ax_pairs.set_xticks(range(len(PAIR_KEYS)), ["table→desk", "desk→table", "dresser→\nnight stand", "night stand→\ndresser"])
    ax_pairs.set(ylabel="test objects (mean over seeds)", title="Critical directional errors")
    ax_pairs.legend(frameon=False, ncols=4, fontsize=8)
    ax_pairs.grid(axis="y", alpha=0.3)
    fig.savefig(out / "results.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
    for key in conds:
        for seed in SEEDS:
            if (key, seed) not in records:
                continue
            h = records[(key, seed)]["history"]
            epochs = [r["epoch"] for r in h]
            axes[0].plot(epochs, [r["val_accuracy"] for r in h], color=COLORS[key], alpha=0.7, lw=1,
                         label=SHORT[key] if seed == 0 else None)
            axes[1].plot(epochs, [r["train_loss"] for r in h], color=COLORS[key], alpha=0.7, lw=1,
                         label=SHORT[key] if seed == 0 else None)
    axes[0].set(xlabel="epoch", ylabel="validation accuracy", ylim=(0.85, 1.0), title="Validation accuracy (all seeds)")
    axes[1].set(xlabel="epoch", ylabel="training loss", title="Training loss (all seeds)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, ncols=4, fontsize=8)
    fig.savefig(out / "curves_all.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- diagnose

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


@torch.no_grad()
def predict(model, points: torch.Tensor, batch_size: int = 128) -> np.ndarray:
    model.eval()
    return torch.cat([model(points[i:i + batch_size]).argmax(1) for i in range(0, len(points), batch_size)]).numpy()


@torch.no_grad()
def gate_values(model: MiniPointNet, points: torch.Tensor, batch_size: int = 128) -> np.ndarray:
    return torch.cat([model.pooling_diagnostics(points[i:i + batch_size])["gate"]
                      for i in range(0, len(points), batch_size)]).numpy()


def gate_stats(gates: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict:
    flat = gates.ravel()
    quantiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    per_channel = gates.mean(axis=0)
    return {
        "mean": float(flat.mean()), "std": float(flat.std()), "min": float(flat.min()), "max": float(flat.max()),
        "quantiles": {str(q): float(np.quantile(flat, q)) for q in quantiles},
        "fraction_below_0.1": float((flat < 0.1).mean()), "fraction_above_0.9": float((flat > 0.9).mean()),
        "fraction_above_0.5": float((flat > 0.5).mean()),
        "per_channel_mean": per_channel.tolist(),
        "per_channel_summary": {"mean_of_channel_means": float(per_channel.mean()), "std_of_channel_means": float(per_channel.std()),
                                "channels_mean_below_0.1": int((per_channel < 0.1).sum()),
                                "channels_mean_above_0.5": int((per_channel > 0.5).sum()),
                                "channels_mean_above_0.9": int((per_channel > 0.9).sum())},
        "within_object_std_mean": float(gates.std(axis=0).mean()),  # how much the gate varies across objects per channel
        "per_class_mean": {name: float(gates[labels == k].mean()) for k, name in enumerate(classes) if (labels == k).any()},
    }


def gradient_participation(model: MiniPointNet, points: torch.Tensor, labels: torch.Tensor, batch_size: int = 64) -> dict:
    """How the loss gradient at the final per-point features is spread over points (eval mode)."""
    model.eval()
    per_object = {"nonzero_point_fraction": [], "critical_point_fraction": [], "participation_ratio": [],
                  "nonmax_gradient_share": [], "argmax_point_gradient_share": []}
    for i in range(0, len(points), batch_size):
        x, y = points[i:i + batch_size], labels[i:i + batch_size]
        features = model.point_features(x).detach().requires_grad_(True)       # (B, C, N), post-ReLU
        loss = F.cross_entropy(model.head(model.pool(features)), y, reduction="sum")
        (grad,) = torch.autograd.grad(loss, features)
        b, c, n = grad.shape
        argmax = features.detach().argmax(dim=2)                                 # (B, C): the max-pooling winner per channel
        is_max = torch.zeros_like(grad, dtype=torch.bool).scatter_(2, argmax.unsqueeze(2), True)
        magnitude = grad.abs()
        point_norm = grad.norm(dim=1)                                            # (B, N)
        critical = is_max.any(dim=1)                                             # (B, N): argmax of >= 1 channel
        total = magnitude.sum(dim=(1, 2)).clamp_min(1e-30)
        per_object["nonzero_point_fraction"] += (point_norm > 0).float().mean(1).tolist()
        per_object["critical_point_fraction"] += critical.float().mean(1).tolist()
        per_object["participation_ratio"] += (point_norm.sum(1) ** 2 / (n * point_norm.pow(2).sum(1).clamp_min(1e-30))).tolist()
        per_object["nonmax_gradient_share"] += ((magnitude * ~is_max).sum(dim=(1, 2)) / total).tolist()
        per_object["argmax_point_gradient_share"] += ((point_norm * critical).sum(1) / point_norm.sum(1).clamp_min(1e-30)).tolist()
    return {key: {"mean": float(np.mean(v)), "median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}
            for key, v in per_object.items()}


def subsample(points: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """A fixed random subset of each object's cached points, re-normalised exactly like inference."""
    if count >= points.shape[1]:
        return points
    rng = np.random.default_rng(seed)
    return np.stack([normalize_points(p[rng.choice(len(p), count, replace=False)]) for p in points]).astype(np.float32)


def diagnose(runs: Path, out: Path, data_dir: Path | None = None, point_counts=(1024, 512, 256)) -> dict:
    torch.set_grad_enabled(True)
    data = dataset.load(data_dir, download=False)
    classes = data.classes
    test_x, test_y = torch.from_numpy(data.test_points).float(), torch.from_numpy(data.test_labels).long()
    val_x = torch.from_numpy(data.val_points).float()
    subsets = {n: torch.from_numpy(subsample(data.test_points, n)) for n in point_counts}
    predictions, report = {}, {"rescored": {}, "mcnemar": {}, "gates": {}, "gradient_participation": {},
                               "point_count_robustness": {}}
    gate_arrays = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            folder = run_dir(runs, condition, seed)
            if not (folder / "best.pt").exists():
                continue
            name = f"{condition}_seed{seed}"
            model, checkpoint = load_checkpoint(folder / "best.pt")
            predicted = predict(model, test_x)
            predictions[(condition, seed)] = predicted
            recorded = json.loads((folder / "metrics.json").read_text())["test_accuracy"]
            accuracy = float((predicted == data.test_labels).mean())
            report["rescored"][name] = {"test_accuracy": accuracy, "recorded": recorded,
                                        "matches_record": bool(abs(accuracy - recorded) < 1e-9)}
            robustness = {}
            for n, points in subsets.items():
                p = predict(model, points)
                confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
                np.add.at(confusion, (data.test_labels, p), 1)
                robustness[str(n)] = {"test_accuracy": float((p == data.test_labels).mean()), **critical_pairs(confusion, classes)}
            report["point_count_robustness"][name] = robustness
            report["gradient_participation"][name] = gradient_participation(model, test_x, test_y)
            if isinstance(model.pool, ResidualGatedDualPool):
                g_val, g_test = gate_values(model, val_x), gate_values(model, test_x)
                gate_arrays[seed] = g_test
                last = model.pool.gate[2]
                report["gates"][name] = {
                    "validation": gate_stats(g_val, data.val_labels, classes),
                    "test": gate_stats(g_test, data.test_labels, classes),
                    "initial_gate": float(torch.sigmoid(torch.tensor(checkpoint["model_config"]["gate_bias"]))),
                    "final_bias_b2": {"mean": float(last.bias.detach().mean()), "min": float(last.bias.detach().min()), "max": float(last.bias.detach().max())},
                    "final_W2_frobenius": float(last.weight.detach().norm()),
                }
            print(f"{name}: test {accuracy:.4f} (recorded {recorded:.4f})", flush=True)
    for a, b in COMPARISONS:
        rows = []
        for seed in SEEDS:
            if (a, seed) in predictions and (b, seed) in predictions:
                ok_a = predictions[(a, seed)] == data.test_labels
                ok_b = predictions[(b, seed)] == data.test_labels
                only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
                rows.append({"seed": seed, f"only_{SHORT[a]}_correct": only_a, f"only_{SHORT[b]}_correct": only_b,
                             "p_exact": mcnemar_exact(only_a, only_b)})
        if rows:
            report["mcnemar"][f"{SHORT[a]}_vs_{SHORT[b]}"] = rows
    report["aggregates"] = aggregate_diagnostics(report)
    report["generated_with"] = {"git_sha": git_sha(), "git_dirty": git_dirty(), "torch": torch.__version__,
                                "subsample_seed": 0, "point_counts": list(point_counts)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        if gate_arrays:
            plot_gates(gate_arrays, report, out)
        plot_robustness(report["aggregates"]["point_count_robustness"], out)
    except ImportError:
        pass
    (out / "diagnostics.md").write_text(markdown_diagnostics(report))
    return report


def aggregate_diagnostics(report: dict) -> dict:
    agg = {"gradient_participation": {}, "point_count_robustness": {}}
    for condition in CONDITIONS:
        names = [f"{condition}_seed{s}" for s in SEEDS if f"{condition}_seed{s}" in report["gradient_participation"]]
        if not names:
            continue
        agg["gradient_participation"][condition] = {
            key: stats([report["gradient_participation"][n][key]["mean"] for n in names])
            for key in report["gradient_participation"][names[0]]}
        agg["point_count_robustness"][condition] = {
            count: {"test_accuracy": stats([report["point_count_robustness"][n][count]["test_accuracy"] for n in names]),
                    "critical_total": stats([report["point_count_robustness"][n][count]["critical_total"] for n in names])}
            for count in report["point_count_robustness"][names[0]]}
    gate_names = list(report["gates"])
    if gate_names:
        agg["gates_test"] = {key: stats([report["gates"][n]["test"][key] for n in gate_names])
                             for key in ("mean", "std", "fraction_below_0.1", "fraction_above_0.9", "fraction_above_0.5")}
        agg["gates_validation"] = {key: stats([report["gates"][n]["validation"][key] for n in gate_names])
                                   for key in ("mean", "std", "fraction_below_0.1", "fraction_above_0.9", "fraction_above_0.5")}
    return agg


def markdown_diagnostics(report: dict) -> str:
    agg = report["aggregates"]
    lines = ["# HW3 diagnostics (generated by `python -m app.classification.hw3 diagnose`)", ""]
    ok = all(r["matches_record"] for r in report["rescored"].values())
    lines += [f"Re-scored test accuracy from every `best.pt` matches the recorded value: **{ok}** "
              f"({len(report['rescored'])} checkpoints).", ""]
    if report["gates"]:
        lines += ["## E2 gate g (per channel and object)", "",
                  "| Run | split | mean | SD | min | median | max | g<0.1 | g>0.5 | g>0.9 | channels mean>0.5 | final b2 mean | ‖W2‖ |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, g in report["gates"].items():
            for split in ("validation", "test"):
                s = g[split]
                lines.append(f"| {name} | {split} | {s['mean']:.3f} | {s['std']:.3f} | {s['min']:.3f} | {s['quantiles']['0.5']:.3f} | "
                             f"{s['max']:.3f} | {100 * s['fraction_below_0.1']:.1f}% | {100 * s['fraction_above_0.5']:.1f}% | "
                             f"{100 * s['fraction_above_0.9']:.1f}% | {s['per_channel_summary']['channels_mean_above_0.5']} | "
                             f"{g['final_bias_b2']['mean']:.2f} | {g['final_W2_frobenius']:.2f} |")
        lines += ["", f"Initial gate: {next(iter(report['gates'].values()))['initial_gate']:.3f} for every channel and object.", ""]
    lines += ["## Exact McNemar tests on paired test predictions (same seed)", "",
              "| Comparison | seed | only first correct | only second correct | p |", "|---|---:|---:|---:|---:|"]
    for name, rows in report["mcnemar"].items():
        for r in rows:
            values = [v for k, v in r.items() if k.startswith("only_")]
            lines.append(f"| {name.replace('_vs_', ' vs ')} | {r['seed']} | {values[0]} | {values[1]} | {r['p_exact']:.3f} |")
    lines += ["", "## P1: gradient participation at the final per-point features (test split, eval mode)", "",
              "| Condition | points with non-zero gradient | critical (argmax) points | participation ratio | gradient share off the argmax |",
              "|---|---:|---:|---:|---:|"]
    for condition, g in agg["gradient_participation"].items():
        lines.append(f"| {SHORT[condition]} | {100 * g['nonzero_point_fraction']['mean']:.1f}% | {100 * g['critical_point_fraction']['mean']:.1f}% | "
                     f"{g['participation_ratio']['mean']:.3f} | {100 * g['nonmax_gradient_share']['mean']:.1f}% |")
    lines += ["", "## P1: deterministic evaluation with fewer points (fixed subset, re-normalised)", "",
              "| Condition | 1024 pts | 512 pts | 256 pts | 4-pair errors at 1024 / 512 / 256 |", "|---|---:|---:|---:|---:|"]
    for condition, r in agg["point_count_robustness"].items():
        counts = list(r)
        lines.append(f"| {SHORT[condition]} | " + " | ".join(
            f"{100 * r[n]['test_accuracy']['mean']:.2f} ± {100 * r[n]['test_accuracy']['std']:.2f}" for n in counts)
            + " | " + " / ".join(f"{r[n]['critical_total']['mean']:.1f}" for n in counts) + " |")
    return "\n".join(lines) + "\n"


def plot_gates(gate_arrays: dict, report: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial = next(iter(report["gates"].values()))["initial_gate"]
    fig, (ax_hist, ax_chan) = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
    bins = np.linspace(0, 1, 51)
    for seed, gates in sorted(gate_arrays.items()):
        ax_hist.hist(gates.ravel(), bins=bins, histtype="step", lw=1.4, label=f"seed {seed}", density=True)
        ax_chan.plot(np.sort(gates.mean(axis=0))[::-1], lw=1.4, label=f"seed {seed}")
    ax_hist.axvline(initial, color="#888", ls=":", label=f"initial g = {initial:.3f}")
    ax_hist.set(xlabel="gate value g (all channels × test objects)", ylabel="density", title="E2 gate distribution (test)")
    ax_hist.set_yscale("log")
    ax_chan.axhline(initial, color="#888", ls=":", label=f"initial g = {initial:.3f}")
    ax_chan.set(xlabel="channel (sorted by mean gate)", ylabel="mean gate over test objects", ylim=(0, 1),
                title="Per-channel mean gate (test)")
    for ax in (ax_hist, ax_chan):
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "gates.png", dpi=150)
    plt.close(fig)


def plot_robustness(robustness: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    for condition, rows in robustness.items():
        counts = sorted(rows, key=int)
        means = [100 * rows[n]["test_accuracy"]["mean"] for n in counts]
        sds = [100 * rows[n]["test_accuracy"]["std"] for n in counts]
        ax.errorbar([int(n) for n in counts], means, yerr=sds, marker="o", ms=4, capsize=3, color=COLORS[condition],
                    label=SHORT[condition])
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024], ["256", "512", "1024"])
    ax.set(xlabel="points per test object (no retraining)", ylabel="test accuracy (%)",
           title="Fewer input points (mean ± SD, 3 seeds)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "robustness.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- probe (post-hoc)

PROBE_PAIRS = (("dresser", "night_stand"), ("table", "desk"))


def read_off_vertices(path: Path) -> np.ndarray:
    """Only the vertex block of an OFF file (handles ModelNet's merged 'OFF<n> <m> 0' header)."""
    lines = [line for line in (raw.split("#", 1)[0].strip() for raw in path.read_text().splitlines()) if line]
    rest = lines[0][3:].strip()
    counts, body = (rest.split(), lines[1:]) if rest else (lines[1].split(), lines[2:])
    n = int(counts[0])
    return np.array(" ".join(body[:n]).split(), dtype=np.float64).reshape(n, 3)


def best_stump(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Threshold and direction of the 1-D decision stump with the best training balanced accuracy."""
    values = np.unique(x)
    candidates = (values[:-1] + values[1:]) / 2 if len(values) > 1 else values
    best = (-1.0, 0.0, 1)
    for t in candidates:
        for sign in (1, -1):
            score = balanced_accuracy(y, (sign * (x - t) > 0).astype(int))
            if score > best[0]:
                best = (score, float(t), sign)
    return best[1], best[2]


def balanced_accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean([(predicted[y == k] == k).mean() for k in (0, 1)]))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    a, b = x[y == 0], x[y == 1]
    pooled = math.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled)


def augment_extents(extents: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The training augmentation's effect on bounding-box extents: an independent U(0.8, 1.25) scale per axis
    (shift does not change extents; the ±0.03 jitter is ignored)."""
    return extents * rng.uniform(0.8, 1.25, size=extents.shape)


def probe(out: Path, data_dir: Path | None = None, augment_draws: int = 20) -> dict:
    data_dir = (data_dir or dataset.data_dir_from_env()).resolve()
    with np.load(dataset.cache_path(data_dir), allow_pickle=False) as cache:
        classes = json.loads(str(cache["meta"]))["classes"]
        splits = {split: {"points": cache[f"{split}_points"], "labels": cache[f"{split}_labels"],
                          "files": cache[f"{split}_files"]} for split in ("train", "test")}
    mesh_root = data_dir / "ModelNet10"
    report = {"note": "post-hoc probe (not pre-registered); train = official training pool (train + validation), "
                      "test = official test split; balanced accuracy of the best 1-D threshold chosen on train",
              "pairs": {}}
    rng = np.random.default_rng(0)
    for a, b in PROBE_PAIRS:
        pair = {}
        features = {}
        for split, d in splits.items():
            keep = np.isin(d["labels"], [classes.index(a), classes.index(b)])
            y = (d["labels"][keep] == classes.index(b)).astype(int)            # 0 = first class, 1 = second
            pts = d["points"][keep]
            extent = pts.max(axis=1) - pts.min(axis=1)                          # normalised input: (x, y, z) extents
            raw_extent = np.stack([np.ptp(read_off_vertices(mesh_root / f), axis=0) for f in d["files"][keep]])
            features[split] = {"y": y, "extent": extent, "raw_extent": raw_extent}
        def feats(e: np.ndarray, raw_e: np.ndarray | None = None) -> dict:
            # ModelNet10 axes (checked on per-class median extents): x = depth, y = width, z = height (up).
            f = {"log height/depth (z/x)": np.log(e[:, 2] / e[:, 0]), "log height/width (z/y)": np.log(e[:, 2] / e[:, 1]),
                 "log depth/width (x/y)": np.log(e[:, 0] / e[:, 1])}
            if raw_e is not None:
                f["log absolute size (raw bbox diagonal)"] = np.log(np.linalg.norm(raw_e, axis=1))
                f["log absolute height (raw z extent)"] = np.log(raw_e[:, 2])
            return f
        train, test = features["train"], features["test"]
        clean_train, clean_test = feats(train["extent"], train["raw_extent"]), feats(test["extent"], test["raw_extent"])
        for name in clean_train:
            t, sign = best_stump(clean_train[name], train["y"])
            row = {"test_balanced_accuracy": balanced_accuracy(test["y"], (sign * (clean_test[name] - t) > 0).astype(int)),
                   "cohens_d_train": cohens_d(clean_train[name], train["y"]),
                   "class_means_train": [float(clean_train[name][train["y"] == k].mean()) for k in (0, 1)]}
            if not name.startswith("log absolute"):
                scores, ds = [], []
                for _ in range(augment_draws):  # train and test both see the augmentation, as the network's training data does
                    aug_train = feats(augment_extents(train["extent"], rng))[name]
                    aug_test = feats(augment_extents(test["extent"], rng))[name]
                    ta, sa = best_stump(aug_train, train["y"])
                    scores.append(balanced_accuracy(test["y"], (sa * (aug_test - ta) > 0).astype(int)))
                    ds.append(cohens_d(aug_train, train["y"]))
                row["augmented_test_balanced_accuracy"] = float(np.mean(scores))
                row["augmented_cohens_d_train"] = float(np.mean(ds))
            pair[name] = row
        pair["counts"] = {"train": [int((train["y"] == k).sum()) for k in (0, 1)], "test": [int((test["y"] == k).sum()) for k in (0, 1)]}
        report["pairs"][f"{a} vs {b}"] = pair
    noise = np.log(rng.uniform(0.8, 1.25, 100_000)) - np.log(rng.uniform(0.8, 1.25, 100_000))
    report["augmentation_log_ratio_noise_sd"] = float(noise.std())
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# HW3 post-hoc probe: size and proportion cues for the confused pairs", "",
             "Not pre-registered; no network is involved. Best single-threshold classifier per feature, chosen on the "
             "official training pool, scored on the official test split (balanced accuracy; 50% = chance). "
             f"The per-axis scale augmentation adds noise with SD {report['augmentation_log_ratio_noise_sd']:.3f} "
             "to every log aspect ratio.", "",
             "| Pair | Feature | Cohen's d (train) | test bal. acc. | with augmentation: d | with augmentation: bal. acc. |",
             "|---|---|---:|---:|---:|---:|"]
    for pair_name, pair in report["pairs"].items():
        for name, row in pair.items():
            if name == "counts":
                continue
            aug = (f"{row['augmented_cohens_d_train']:+.2f} | {100 * row['augmented_test_balanced_accuracy']:.1f}%"
                   if "augmented_cohens_d_train" in row else "n/a (removed by normalisation) | n/a")
            lines.append(f"| {pair_name} | {name} | {row['cohens_d_train']:+.2f} | {100 * row['test_balanced_accuracy']:.1f}% | {aug} |")
    (out / "probe.md").write_text("\n".join(lines) + "\n")
    return report


# ----------------------------------------------------------------------------- collect

def collect(runs: Path, dest: Path) -> list[str]:
    copied = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            source = run_dir(runs, condition, seed)
            if not (source / "metrics.json").exists():
                continue
            target = dest / source.name
            target.mkdir(parents=True, exist_ok=True)
            for artifact in ARTIFACTS:
                if (source / artifact).exists():
                    shutil.copy2(source / artifact, target / artifact)
            copied.append(source.name)
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("summarize", "diagnose"):
        p = sub.add_parser(name)
        p.add_argument("--runs", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "diagnose":
            p.add_argument("--data-dir", type=Path, default=None)
    p = sub.add_parser("collect")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p = sub.add_parser("probe")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "probe":
        probe(args.out, args.data_dir)
        print((args.out / "probe.md").read_text())
        return 0
    if args.command == "summarize":
        summary = summarize(args.runs, args.out)
        print((args.out / "summary.md").read_text())
        return 0 if summary else 1
    if args.command == "diagnose":
        diagnose(args.runs, args.out, args.data_dir)
        print((args.out / "diagnostics.md").read_text())
        return 0
    print(json.dumps({"collected": collect(args.runs, args.dest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
