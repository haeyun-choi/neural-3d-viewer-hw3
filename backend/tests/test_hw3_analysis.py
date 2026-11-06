"""HW3 analysis: aggregation, McNemar, gate statistics and the P1 diagnostics on tiny fixture runs."""

import csv
import json

import numpy as np
import pytest
import torch

from app.classification import dataset, hw3
from app.classification.models import build_model
from app.classification.training import TrainConfig, train

from classification_fixtures import write_fixture_dataset


def test_mcnemar_exact_known_values():
    assert hw3.mcnemar_exact(0, 0) == 1.0
    assert hw3.mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert hw3.mcnemar_exact(10, 2) == pytest.approx(2 * (1 + 12 + 66) / 4096)
    assert hw3.mcnemar_exact(3, 3) == 1.0


def test_critical_pairs_reads_directional_errors():
    classes = list(dataset.CLASSES)
    confusion = np.zeros((10, 10), dtype=int)
    i = classes.index
    confusion[i("table"), i("desk")] = 12
    confusion[i("desk"), i("table")] = 4
    confusion[i("dresser"), i("night_stand")] = 11
    confusion[i("night_stand"), i("dresser")] = 7
    pairs = hw3.critical_pairs(confusion, classes)
    assert pairs == {"table->desk": 12, "desk->table": 4, "dresser->night_stand": 11, "night_stand->dresser": 7,
                     "critical_total": 34}


def test_subsample_is_deterministic_and_renormalised():
    points = np.random.default_rng(0).normal(size=(3, 1024, 3)).astype(np.float32)
    a, b = hw3.subsample(points, 256), hw3.subsample(points, 256)
    assert a.shape == (3, 256, 3) and np.array_equal(a, b)
    assert np.allclose(a.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(np.linalg.norm(a, axis=2).max(axis=1), 1, atol=1e-5)
    assert hw3.subsample(points, 1024) is points


def test_gradient_participation_separates_max_from_dual_pooling():
    torch.manual_seed(0)
    x, y = torch.randn(6, 128, 3), torch.randint(0, 10, (6,))
    results = {}
    for name in ("mini_pointnet", "mini_pointnet_fixed_dual"):
        model = build_model(name, hidden_dim=16).eval()
        results[name] = hw3.gradient_participation(model, x, y)
    maximum, dual = results["mini_pointnet"], results["mini_pointnet_fixed_dual"]
    assert maximum["nonmax_gradient_share"]["max"] == 0  # max pooling: gradient only at the argmax
    assert maximum["nonzero_point_fraction"]["mean"] == pytest.approx(maximum["critical_point_fraction"]["mean"])
    assert dual["nonzero_point_fraction"]["min"] == 1.0  # the mean reaches every point
    assert dual["nonmax_gradient_share"]["mean"] > 0
    assert dual["participation_ratio"]["mean"] > maximum["participation_ratio"]["mean"]


@pytest.fixture(scope="module")
def fixture_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("hw3")
    write_fixture_dataset(root / "data")
    data = dataset.load(root / "data", download=False)
    for condition, (_, model, hidden) in hw3.CONDITIONS.items():
        for seed in hw3.SEEDS:
            config = TrainConfig(model=model, hidden_dim=hidden // 4, epochs=2, batch_size=10, learning_rate=5e-3,
                                 seed=seed, device="cpu")
            train(config, data, root / "runs" / f"{condition}_seed{seed}")
    return root


def test_summarize_writes_tables_and_paired_comparisons(fixture_runs, tmp_path):
    summary = hw3.summarize(fixture_runs / "runs", tmp_path)
    assert set(summary["conditions"]) == set(hw3.CONDITIONS)
    e2 = summary["conditions"]["e2_gated_dual"]
    assert e2["seeds"] == [0, 1, 2] and len(e2["test_accuracy"]["values"]) == 3
    assert set(summary["paired_vs"]) == {"E1-E0", "E2-E0", "E3-E0", "E2-E1", "E2-E3"}
    assert summary["e0_reproduces_hw2"]["identical"] is False  # fixture runs are not the HW2 runs
    rows = list(csv.DictReader((tmp_path / "summary_runs.csv").open()))
    assert len(rows) == 12 and {"table->desk", "critical_total", "parameters"} <= set(rows[0])
    text = (tmp_path / "summary.md").read_text()
    assert "table→desk" in text and "E2-E3" in text
    assert (tmp_path / "results.png").exists() and (tmp_path / "curves_all.png").exists()


def test_diagnose_rescored_gates_mcnemar_and_robustness(fixture_runs, tmp_path):
    report = hw3.diagnose(fixture_runs / "runs", tmp_path, fixture_runs / "data", point_counts=(1024, 256))
    assert len(report["rescored"]) == 12 and all(r["matches_record"] for r in report["rescored"].values())
    gates = report["gates"]["e2_gated_dual_seed0"]
    for split in ("validation", "test"):
        g = gates[split]
        assert 0 <= g["min"] <= g["mean"] <= g["max"] <= 1
        assert len(g["per_channel_mean"]) == 64  # 4 * hidden (16)
    assert gates["initial_gate"] == pytest.approx(0.1192, abs=1e-4)
    assert {r["seed"] for r in report["mcnemar"]["E2_vs_E0"]} == {0, 1, 2}
    assert set(report["point_count_robustness"]["e0_max_seed0"]) == {"1024", "256"}
    assert report["point_count_robustness"]["e0_max_seed0"]["1024"]["test_accuracy"] == \
        report["rescored"]["e0_max_seed0"]["test_accuracy"]
    assert (tmp_path / "gates.png").exists()
    assert "McNemar" in (tmp_path / "diagnostics.md").read_text()
    assert json.loads((tmp_path / "diagnostics.json").read_text())["aggregates"]["gates_test"]["mean"]["values"]


def test_collect_copies_compact_artifacts(fixture_runs, tmp_path):
    copied = hw3.collect(fixture_runs / "runs", tmp_path / "released")
    assert len(copied) == 12
    assert sorted(p.name for p in (tmp_path / "released" / "e2_gated_dual_seed1").iterdir()) == sorted(hw3.ARTIFACTS)


def test_probe_helpers(tmp_path):
    (tmp_path / "m.off").write_text("OFF3 1 0\n0 0 0\n2 0 0\n0 1 5\n3 0 1 2\n")
    assert np.array_equal(np.ptp(hw3.read_off_vertices(tmp_path / "m.off"), axis=0), [2, 1, 5])
    x = np.array([0.1, 0.2, 0.3, 1.1, 1.2, 1.3])
    y = np.array([1, 1, 1, 0, 0, 0])
    threshold, sign = hw3.best_stump(x, y)
    assert hw3.balanced_accuracy(y, (sign * (x - threshold) > 0).astype(int)) == 1.0
    assert hw3.cohens_d(x, y) > 5
    extents = np.ones((1000, 3))
    augmented = hw3.augment_extents(extents, np.random.default_rng(0))
    assert augmented.min() >= 0.8 and augmented.max() <= 1.25
