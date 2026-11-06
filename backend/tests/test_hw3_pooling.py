"""HW3: pooling variants (max, fixed dual, residual gated dual), their controls and checkpoints."""

import json
import math
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from app.classification import dataset, training
from app.classification.inference import ModelRegistry, parse_ply, preprocess
from app.classification.models import (GATE_BIAS, MINI_POINTNET_POOLING, FixedDualPool, GlobalMaxPool, MiniPointNet,
                                       ResidualGatedDualPool, build_model, parameter_count)
from app.classification.routes import ClassificationSettings
from app.classification.training import TrainConfig, load_checkpoint, save_run, train
from app.main import create_app

from classification_fixtures import write_fixture_dataset, write_fixture_models, write_fixture_samples

HW3_MODELS = ("mini_pointnet", "mini_pointnet_fixed_dual", "mini_pointnet_gated_dual")
# The four HW3 conditions: (model name, hidden width).
CONDITIONS = {"E0": ("mini_pointnet", 64), "E1": ("mini_pointnet_fixed_dual", 64),
              "E2": ("mini_pointnet_gated_dual", 64), "E3": ("mini_pointnet", 69)}


def features(batch=4, channels=256, points=1024, seed=0):
    return torch.relu(torch.randn(batch, channels, points, generator=torch.Generator().manual_seed(seed)))


# ----------------------------------------------------------------------------- shapes and invariance

@pytest.mark.parametrize("name", ("shallow", *HW3_MODELS))
@pytest.mark.parametrize("points", (1024, 256))
def test_every_variant_maps_points_to_ten_logits(name, points):
    model = build_model(name).eval()
    assert model(torch.randn(3, points, 3)).shape == (3, 10)


@pytest.mark.parametrize("name", HW3_MODELS)
def test_every_variant_is_permutation_invariant_in_eval_mode(name):
    model = build_model(name).eval()
    x = torch.randn(4, 1024, 3)
    logits = model(x)
    for seed in range(3):
        shuffled = x[:, torch.randperm(1024, generator=torch.Generator().manual_seed(seed))]
        assert torch.allclose(model(shuffled), logits, atol=1e-5)


def test_pooling_modules_are_selected_by_model_name():
    assert MINI_POINTNET_POOLING == {"mini_pointnet": "max", "mini_pointnet_fixed_dual": "fixed_dual",
                                     "mini_pointnet_gated_dual": "gated_dual"}
    for name, kind in ((HW3_MODELS[0], GlobalMaxPool), (HW3_MODELS[1], FixedDualPool),
                       (HW3_MODELS[2], ResidualGatedDualPool)):
        model = build_model(name)
        assert isinstance(model, MiniPointNet) and isinstance(model.pool, kind)
    with pytest.raises(ValueError):
        build_model("mini_pointnet_attention")


# ----------------------------------------------------------------------------- formulas

def test_max_pooling_is_the_unchanged_hw2_formula():
    torch.manual_seed(0)
    model = build_model("mini_pointnet").eval()
    x = torch.randn(4, 1024, 3)
    expected = model.head(model.point_mlp(x.transpose(1, 2)).max(dim=2).values)  # HW2 forward, spelled out
    assert torch.equal(model(x), expected)


def test_fixed_dual_pooling_is_half_max_plus_half_mean():
    h = features()
    expected = 0.5 * h.max(dim=2).values + 0.5 * h.mean(dim=2)
    assert torch.allclose(FixedDualPool()(h), expected, atol=1e-7)
    assert list(FixedDualPool().parameters()) == []
    model = build_model("mini_pointnet_fixed_dual").eval()
    diagnostics = model.pooling_diagnostics(torch.randn(2, 1024, 3))
    assert torch.allclose(diagnostics["pooled"], 0.5 * diagnostics["max"] + 0.5 * diagnostics["mean"], atol=1e-6)
    assert torch.all(diagnostics["gate"] == 0.5)


def test_gated_pooling_gate_shape_range_and_initial_value():
    pool = ResidualGatedDualPool(256)
    h = features()
    m, a = h.max(dim=2).values, h.mean(dim=2)
    g = pool.gate_values(m, a)
    assert g.shape == (4, 256) and torch.isfinite(g).all()
    # Baseline-preserving start: W2 = 0, b2 = -2, so g = sigmoid(-2) ~ 0.119 for every channel and object.
    assert torch.allclose(g, torch.full_like(g, 1 / (1 + math.exp(2))), atol=1e-6)
    assert abs(float(g.detach().mean()) - 0.1192) < 1e-3
    assert torch.allclose(pool(h), m + g * (a - m), atol=1e-6)
    # After arbitrary weights, and for huge inputs, the gate stays a finite value in [0, 1]
    # and the pooled feature stays between the mean and the max of each channel.
    with torch.no_grad():
        for p in pool.parameters():
            p.normal_(0, 3)
    for scale in (1.0, 1e4):
        hs = h * scale
        g = pool.gate_values(hs.max(dim=2).values, hs.mean(dim=2))
        assert torch.isfinite(g).all() and (g >= 0).all() and (g <= 1).all()
        z = pool(hs)
        assert torch.isfinite(z).all()
        assert (z <= hs.max(dim=2).values + 1e-3 * scale).all() and (z >= hs.mean(dim=2) - 1e-3 * scale).all()


def test_gate_bias_is_configurable_and_recorded():
    model = build_model("mini_pointnet_gated_dual", gate_hidden=8, gate_bias=0.0).eval()
    g = model.pooling_diagnostics(torch.randn(2, 128, 3))["gate"]
    assert torch.allclose(g, torch.full_like(g, 0.5))
    assert model.pool.gate[0].out_features == 8


# ----------------------------------------------------------------------------- parameter counts

def test_parameter_counts_of_the_four_conditions():
    counts = {cid: parameter_count(build_model(name, hidden_dim=h)) for cid, (name, h) in CONDITIONS.items()}
    assert counts == {"E0": 76_362, "E1": 76_362, "E2": 88_922, "E3": 88_537}
    gate = build_model("mini_pointnet_gated_dual").pool
    assert parameter_count(gate) == 512 * 16 + 16 + 16 * 256 + 256 == 12_560
    # E3 is the parameter-matched control: within 0.5% of E2.
    assert abs(counts["E3"] - counts["E2"]) / counts["E2"] < 0.005


# ----------------------------------------------------------------------------- controls

def test_shared_layers_and_random_stream_are_identical_across_poolings():
    """For one seed, E0/E1/E2 start from the same shared weights and the same dropout stream."""
    states, streams = {}, {}
    for name in HW3_MODELS:
        torch.manual_seed(123)
        model = build_model(name)
        states[name] = {k: v for k, v in model.state_dict().items() if not k.startswith("pool.")}
        streams[name] = torch.rand(8)  # what dropout would draw next
    reference = states["mini_pointnet"]
    for name in HW3_MODELS[1:]:
        assert states[name].keys() == reference.keys()
        assert all(torch.equal(states[name][k], reference[k]) for k in reference)
        assert torch.equal(streams[name], streams["mini_pointnet"])


# ----------------------------------------------------------------------------- gradients

@pytest.mark.parametrize("cid", CONDITIONS)
def test_backward_gives_every_parameter_a_finite_gradient(cid):
    name, hidden = CONDITIONS[cid]
    torch.manual_seed(0)
    model = build_model(name, hidden_dim=hidden).train()
    x, y = torch.randn(8, 256, 3), torch.randint(0, 10, (8,))
    torch.nn.functional.cross_entropy(model(x), y).backward()
    for parameter_name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), parameter_name


def test_gate_learns_from_its_zero_initialised_output_layer():
    """W2 = 0 blocks W1's gradient on the first step only; W2 and b2 learn immediately."""
    torch.manual_seed(0)
    model = build_model("mini_pointnet_gated_dual", hidden_dim=16).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x, y = torch.randn(8, 256, 3), torch.randint(0, 10, (8,))
    first, hidden, last = model.pool.gate
    torch.nn.functional.cross_entropy(model(x), y).backward()
    assert torch.count_nonzero(first.weight.grad) == 0
    assert torch.count_nonzero(last.weight.grad) > 0 and torch.count_nonzero(last.bias.grad) > 0
    optimizer.step()
    optimizer.zero_grad()
    torch.nn.functional.cross_entropy(model(x), y).backward()
    assert torch.count_nonzero(first.weight.grad) > 0


def test_gradient_reaches_one_point_per_channel_with_max_and_every_point_with_dual():
    h = features(batch=1, channels=8, points=64).requires_grad_(True)
    GlobalMaxPool()(h).sum().backward()
    assert torch.equal((h.grad != 0).sum(dim=2), torch.ones(1, 8, dtype=torch.long))
    for pool in (FixedDualPool(), ResidualGatedDualPool(8, gate_hidden=4)):
        h.grad = None
        pool(h).sum().backward()
        assert (h.grad != 0).all()


# ----------------------------------------------------------------------------- checkpoints and training

@pytest.mark.parametrize("cid", CONDITIONS)
def test_checkpoint_round_trip_reproduces_logits(cid, tmp_path):
    name, hidden = CONDITIONS[cid]
    torch.manual_seed(1)
    config = TrainConfig(model=name, hidden_dim=hidden, epochs=1)
    model = build_model(name, 10, **config.model_kwargs())
    with torch.no_grad():  # move the gate away from its initial value so the round trip is not trivial
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    model.eval()
    metrics = {"model": name, "parameters": parameter_count(model), "best_epoch": 1, "best_val_accuracy": 0.5,
               "test_accuracy": 0.5, "git_sha": None}
    save_run(tmp_path / cid, model, config, list(dataset.CLASSES), [], metrics)
    loaded, checkpoint = load_checkpoint(tmp_path / cid / "best.pt")
    assert checkpoint["model"] == name and not loaded.training
    expected = {"hidden_dim": hidden, "activation": "relu", "dropout": 0.3}
    if name == "mini_pointnet_gated_dual":
        expected.update(gate_hidden=16, gate_bias=GATE_BIAS)
    assert checkpoint["model_config"] == expected
    assert json.loads((tmp_path / cid / "config.json").read_text())["model"] == name
    x = torch.randn(5, 1024, 3)
    assert torch.allclose(loaded(x), model(x), atol=1e-6)


@pytest.fixture
def fixture_data(tmp_path):
    write_fixture_dataset(tmp_path / "data")
    return tmp_path / "data"


@pytest.mark.parametrize("name", HW3_MODELS[1:])
def test_training_smoke_for_the_new_variants(name, fixture_data, tmp_path):
    data = dataset.load(fixture_data, download=False)
    config = TrainConfig(model=name, epochs=3, learning_rate=5e-3, batch_size=10, hidden_dim=16, device="cpu")
    result = train(config, data, tmp_path / "run")
    metrics = result["metrics"]
    assert metrics["model"] == name and 0 <= metrics["test_accuracy"] <= 1
    assert metrics["runtime"]["threads"] >= 1 and "git_dirty" in metrics
    model, _ = load_checkpoint(tmp_path / "run" / "best.pt")
    x = torch.from_numpy(data.test_points).float()
    result["model"].eval()
    assert torch.allclose(model(x), result["model"](x), atol=1e-5)


def test_cli_accepts_the_new_models_and_gate_options(fixture_data, tmp_path):
    out = tmp_path / "cli"
    args = ["train", "--model", "mini_pointnet_gated_dual", "--epochs", "1", "--hidden-dim", "16", "--batch-size", "10",
            "--device", "cpu", "--gate-hidden", "8", "--gate-bias", "-1.5", "--data-dir", str(fixture_data),
            "--out", str(out)]
    assert training.main(args) == 0
    config = json.loads((out / "config.json").read_text())
    assert (config["model"], config["gate_hidden"], config["gate_bias"]) == ("mini_pointnet_gated_dual", 8, -1.5)
    _, checkpoint = load_checkpoint(out / "best.pt")
    assert checkpoint["model_config"]["gate_hidden"] == 8 and checkpoint["model_config"]["gate_bias"] == -1.5
    with pytest.raises(SystemExit):
        training.main(["train", "--model", "mini_pointnet_attention", "--out", str(out)])


def test_api_trains_and_serves_a_gated_dual_run(tmp_path, fixture_data):
    write_fixture_models(tmp_path / "models")
    write_fixture_samples(tmp_path / "samples")
    settings = ClassificationSettings(models_dir=tmp_path / "models", samples_dir=tmp_path / "samples",
                                      runs_dir=tmp_path / "runs", data_dir=fixture_data, allow_download=False)
    with TestClient(create_app(tmp_path / "db.sqlite3", settings)) as client:
        api = "/api/classification"
        assert "mini_pointnet_gated_dual" in client.get(f"{api}/models").json()["architectures"]
        run_id = client.post(f"{api}/runs", json={"model": "mini_pointnet_gated_dual", "epochs": 2, "hidden_dim": 16,
                                                   "batch_size": 10}).json()["id"]
        deadline = time.monotonic() + 120
        while (run := client.get(f"{api}/runs/{run_id}").json())["status"] in ("queued", "running"):
            assert time.monotonic() < deadline, run
            time.sleep(0.1)
        assert run["status"] == "completed", run
        sample = client.get(f"{api}/samples").json()["samples"][0]["id"]
        result = client.post(f"{api}/predict/sample", json={"sample_id": sample, "model_id": f"run-{run_id}"}).json()
        assert result["model"]["architecture"] == "mini_pointnet_gated_dual"
        assert result["model"]["config"]["gate_hidden"] == 16


# ----------------------------------------------------------------------------- bundled HW3 model (web classifier)

BACKEND = Path(__file__).resolve().parents[1]
BUNDLED = BACKEND / "models" / "modelnet10"
RELEASED_E2 = BACKEND.parent / "results" / "hw3" / "runs" / "e2_gated_dual_seed0"


@pytest.mark.skipif(not (BUNDLED / "mini_pointnet_gated_dual" / "best.pt").exists(), reason="HW3 model not bundled")
def test_bundled_hw3_model_is_the_released_e2_run_and_classifies_the_samples():
    bundled = BUNDLED / "mini_pointnet_gated_dual"
    for name in ("best.pt", "metrics.json", "config.json"):
        assert (bundled / name).read_bytes() == (RELEASED_E2 / name).read_bytes()
    registry = ModelRegistry(BUNDLED, BACKEND / "data" / "no-runs")
    assert [m["id"] for m in registry.available()][:3] == ["mini_pointnet", "shallow", "mini_pointnet_gated_dual"]
    model, checkpoint = registry.get("mini_pointnet_gated_dual")
    assert checkpoint["model"] == "mini_pointnet_gated_dual" and checkpoint["model_config"]["gate_hidden"] == 16
    assert checkpoint["test_accuracy"] == pytest.approx(json.loads((RELEASED_E2 / "metrics.json").read_text())["test_accuracy"])
    samples = json.loads((BACKEND / "samples" / "modelnet10" / "index.json").read_text())["samples"]
    hits = 0
    for entry in samples:
        points, _ = preprocess(parse_ply((BACKEND / "samples" / "modelnet10" / entry["file"]).read_bytes()))
        result = registry.predict("mini_pointnet_gated_dual", points)
        assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-5)
        hits += result["predicted_class"] == entry["class"]
    assert hits >= 0.8 * len(samples)
