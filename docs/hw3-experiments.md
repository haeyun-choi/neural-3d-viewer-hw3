# HW3 experiments

Design, hypothesis and decision rules were fixed beforehand in
[`hw3-design.md`](hw3-design.md) (commit `969fcfe`, before any HW3 training).
This file records how the runs were produced and what they showed.

## Environment

| Item | Value |
| --- | --- |
| Code | branch `feature/hw3-gated-pooling`; every training run executed commit **`14bc0d0`** from a detached Git worktree, so later documentation commits could not change the running code (`git_dirty: false` in every `metrics.json`) |
| Python / packages | Python 3.12.9 (conda-forge), torch 2.14.1+cu130 executed on CPU, numpy 2.5.4, matplotlib 3.11.2; `pip freeze` identical to the HW2 environment (pins in `backend/requirements-dev.txt`). The HW2 virtual environment could not be reused because its base interpreter was removed from the cluster, so it was rebuilt with the same pins |
| Hardware | UVA HPC, Slurm `interactive` partition restricted to `rivanna` nodes: **Intel Xeon Gold 6248 @ 2.50 GHz**, 8 CPU threads (`OMP_NUM_THREADS=8`) and 16 GB per run, three runs at a time on one node — the same CPU model and thread count as the HW2 runs. No GPU was used: the development host has none, a GPU job would have started hours later, and CPU execution on the HW2 CPU model makes the E0 rerun directly comparable with HW2 |
| Data | processed cache `modelnet10_1024pts_seed0_v1.npz` from HW2 (49,267,465 bytes, SHA-256 `a522b3f1e6137005ab4ccd554ff84549e045108e306bae3a5a9ed25214a8002c`), built from `ModelNet10.zip` SHA-256 `9d867943…07d51`; train 3,193 / val 798 / test 908 |
| Jobs | environment 22556128; verification 22556241 (backend tests, backend smoke, 1-epoch smoke of E0–E3 with `--skip-test`); matrix **22556284** (array 0–11, at most 3 concurrent) |

## Exact commands

From the repository root, with a prepared cache:

```bash
export MODELNET10_DATA_DIR=/path/to/modelnet10            # contains processed/modelnet10_1024pts_seed0_v1.npz
(cd backend && python -m app.classification.dataset prepare)   # no-op when the cache exists
OUT=backend/data/experiments/hw3 scripts/hw3/run_matrix.sh      # 12 runs; or TASK=i for run i only
cd backend
python -m app.classification.hw3 summarize --runs data/experiments/hw3 --out ../results/hw3
python -m app.classification.hw3 diagnose  --runs data/experiments/hw3 --out ../results/hw3
python -m app.classification.hw3 collect   --runs data/experiments/hw3 --dest ../results/hw3/runs
```

`scripts/hw3/run_matrix.sh` maps run *i* to condition *i* mod 4 and seed
⌊*i*/4⌋ and calls, for example (E2, seed 1):

```bash
python -m app.classification.training train --model mini_pointnet_gated_dual --hidden-dim 64 --seed 1 \
  --epochs 60 --learning-rate 0.001 --batch-size 32 --weight-decay 0.0001 --scheduler cosine \
  --dropout 0.3 --activation relu --augment true --device cpu --out data/experiments/hw3/e2_gated_dual_seed1
```

On the cluster each array task ran this script with `TASK=$SLURM_ARRAY_TASK_ID`.
`diagnose` also works on the released checkpoints:
`python -m app.classification.hw3 diagnose --runs ../results/hw3/runs --out /tmp/hw3-check`.

## Configurations

| ID | Run folders | `--model` | `--hidden-dim` | Extra | Parameters |
| --- | --- | --- | ---: | --- | ---: |
| E0 | `e0_max_seed{0,1,2}` | `mini_pointnet` | 64 | — | 76,362 |
| E1 | `e1_fixed_dual_seed{0,1,2}` | `mini_pointnet_fixed_dual` | 64 | — | 76,362 |
| E2 | `e2_gated_dual_seed{0,1,2}` | `mini_pointnet_gated_dual` | 64 | gate 512→16→256, `--gate-hidden 16 --gate-bias -2.0` (defaults) | 88,922 |
| E3 | `e3_max_h69_seed{0,1,2}` | `mini_pointnet` | 69 | — | 88,537 |

Shared by all runs: 60 epochs, batch 32, Adam lr 0.001, weight decay 1e-4 on all
parameters, cosine schedule to 1e-5, ReLU, dropout 0.3, training-only
augmentation (per-axis scale U(0.8, 1.25), shift U(−0.1, 0.1), jitter
σ = 0.01 clipped at 0.03, no rotation), checkpoint = best validation accuracy
(ties: lower validation loss), test split scored once with that checkpoint.

## Diagnostic definitions

- **Critical directional errors**: test-confusion-matrix entries
  (true → predicted) table→desk, desk→table, dresser→night_stand,
  night_stand→dresser, and their sum ("4-pair total").
- **Paired difference**: for each seed, condition minus E0 (same batches and
  augmentation; for E1/E2 also the same initial shared weights and dropout stream).
- **Exact McNemar test**: per seed, the test objects that only one of two runs
  classifies correctly (*b*, *c*); two-sided p = min(1, 2·P[Bin(b+c, ½) ≤ min(b, c)]).
- **Gate statistics (E2)**: `g` for every (object, channel) on the validation
  (798 × 256) and test (908 × 256) splits; mean, SD, min/max, quantiles,
  fractions below 0.1 / above 0.5 / above 0.9, per-channel means, the final
  output-layer bias `b₂` and the Frobenius norm of `W₂` (both 0 → −2 and 0 at
  initialisation).
- **Gradient participation (P1)**: eval mode, test split, `∂L/∂h` at the final
  per-point features (summed cross-entropy). Points with non-zero gradient;
  critical points (argmax of at least one channel); participation ratio
  `(Σᵢ wᵢ)² / (N Σᵢ wᵢ²)` with `wᵢ = ‖∂L/∂hᵢ‖₂` (1 = all points equal,
  1/N = one point); share of `|∂L/∂h|` that falls on non-argmax entries.
- **Point-count robustness (P1)**: each test object's cached 1,024 points
  subsampled to 512 or 256 with a fixed generator (seed 0, same subset for every
  model), then re-normalised with the same centroid/radius rule; no retraining.

## Run directories, seeds and selected epochs

All 12 runs: commit `14bc0d0`, `git_dirty: false`, Intel Xeon Gold 6248, 8 threads.
Released copies (checkpoint, config, history, metrics, curves, confusion matrix)
are in [`results/hw3/runs/`](../results/hw3/runs/).

| Run folder | Slurm task | Node | Selected epoch | Best val acc | Test acc | Train time |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `e0_max_seed0` | 22556284_0 | udc-aw32-10c1 | 45 | 97.74% | 93.61% | 458 s |
| `e1_fixed_dual_seed0` | 22556284_1 | udc-aw32-10c1 | 37 | 98.12% | 92.84% | 500 s |
| `e2_gated_dual_seed0` | 22556284_2 | udc-aw32-10c1 | 51 | 97.37% | 93.17% | 521 s |
| `e3_max_h69_seed0` | 22556284_3 | udc-aw32-10c1 | 42 | 97.49% | 91.63% | 499 s |
| `e0_max_seed1` | 22556284_4 | udc-aw32-10c1 | 40 | 97.74% | 92.29% | 422 s |
| `e1_fixed_dual_seed1` | 22556284_5 | udc-aw32-10c1 | 40 | 97.74% | 92.29% | 454 s |
| `e2_gated_dual_seed1` | 22556284_6 | udc-aw32-1c0 | 36 | 97.74% | 92.40% | 702 s |
| `e3_max_h69_seed1` | 22556284_7 | udc-aw32-11c0 | 33 | 97.24% | 91.52% | 549 s |
| `e0_max_seed2` | 22556284_8 | udc-aw32-10c1 | 49 | 97.62% | 92.40% | 320 s |
| `e1_fixed_dual_seed2` | 22556284_9 | udc-aw32-11c0 | 55 | 97.12% | 92.84% | 484 s |
| `e2_gated_dual_seed2` | 22556284_10 | udc-aw32-1c1 | 46 | 97.37% | 92.73% | 401 s |
| `e3_max_h69_seed2` | 22556284_11 | udc-aw32-1c0 | 49 | 97.49% | 92.29% | 431 s |

`e0_max_seed1` and `e1_fixed_dual_seed1` end with the same selected epoch, the
same validation accuracy (780/798) and the same test accuracy (838/908). This
was checked and is a coincidence of totals, not a shared run: their training
curves differ from epoch 1 (train loss 0.8009 vs 0.8318), their test losses
differ (0.211 vs 0.194) and their confusion matrices differ in 16 cells (e.g.
desk recall −5 objects, night_stand recall +4 for E1).

## Reproduction check: E0 = HW2

The E0 rerun reproduces the stored HW2 Mini-PointNet **exactly**: test accuracy
93.61% / 92.29% / 92.40%, selected epochs 45 / 40 / 49 and validation accuracy
97.74% / 97.74% / 97.62% for seeds 0 / 1 / 2, with identical per-epoch losses
(spot-checked at epochs 1, 10, 20, 23). Same CPU model, thread count and
package versions make the CPU computation deterministic, so every E1–E3 versus
E0 difference below is caused by the change, plus the seed-to-seed training
variability that the three seeds measure.

## Aggregate results (official test split, 908 objects)

Mean ± SD over seeds 0–2 (population SD, as in HW2). Critical-pair entries are
test objects (mean over seeds, per-seed values in brackets).

| Model | Params | Val acc | Test acc | table→desk | desk→table | dresser→night | night→dresser | 4-pair total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 max (HW2) | 76,362 | 97.70 ± 0.06 | **92.77 ± 0.60** | 16.0 [12, 17, 19] | 4.3 [4, 5, 4] | 11.0 [11, 10, 12] | 11.3 [11, 12, 11] | 42.7 |
| E1 fixed dual | 76,362 | 97.66 ± 0.41 | **92.66 ± 0.26** | 11.3 [7, 16, 11] | 4.7 [5, 5, 4] | 11.7 [13, 12, 10] | 12.3 [12, 11, 14] | 40.0 |
| E2 gated dual | 88,922 | 97.49 ± 0.18 | **92.77 ± 0.32** | 16.7 [17, 17, 16] | 4.3 [4, 5, 4] | 8.3 [8, 6, 11] | 15.3 [19, 15, 12] | 44.7 |
| E3 max, h = 69 | 88,537 | 97.41 ± 0.12 | **91.81 ± 0.34** | 19.7 [19, 21, 19] | 4.3 [4, 5, 4] | 15.7 [15, 17, 15] | 12.0 [12, 12, 12] | 51.7 |

Paired by seed (E1/E2 share initial weights, batches, augmentation and dropout
stream with E0; E3 shares batches and augmentation):

| Comparison | mean Δ test | per seed (pp) | wins/losses | Δ 4-pair errors | exact McNemar p (seeds 0, 1, 2) |
|---|---:|---|---:|---:|---|
| E1 − E0 | −0.11 pp | −0.77, 0.00, +0.44 | 1/1 | −2.7 | 0.31, 1.00, 0.52 |
| E2 − E0 | **+0.00 pp** | −0.44, +0.11, +0.33 | 2/1 | +2.0 | 0.57, 1.00, 0.61 |
| E3 − E0 | −0.95 pp | −1.98, −0.77, −0.11 | 0/3 | +9.0 | 0.001, 0.35, 1.00 |
| E2 − E1 | +0.11 pp | +0.33, +0.11, −0.11 | 2/1 | +4.7 | 0.76, 1.00, 1.00 |
| E2 − E3 | +0.95 pp | +1.54, +0.88, +0.44 | 3/0 | −7.0 | 0.03, 0.28, 0.52 |

Even with identical initial shared weights and identical batches, E2 and E0
disagree on 28, 39 and 15 test objects (seeds 0–2); the accuracy differences
are the small imbalance between two large sets of flips.

Per-class test recall (mean over seeds), the four classes of interest:
E0 desk 89.1%, table 84.0%, dresser 85.7%, night_stand 80.2%;
E2 desk 86.8%, table 83.3%, dresser 88.4%, night_stand 78.7%.

Training time: E0 400 s, E1 480 s, E2 541 s, E3 493 s per run (mean); 6.6 /
7.9 / 9.0 / 8.2 s per epoch. Three runs shared a node, and the three E0 runs —
identical work — took 320–458 s, so these differences are indicative only. The
gate itself is ~12.5k multiply-adds per object against ~42M in the point MLP;
the dual pools add one extra dense pass (and its gradient) over the
`(B, 256, N)` feature tensor.

Figures: [`results.png`](../results/hw3/results.png) (accuracy and critical
errors), [`curves_all.png`](../results/hw3/curves_all.png) (validation accuracy
and training loss of all 12 runs: the four conditions are indistinguishable
during training).

## E2 gate diagnostics

Initial gate 0.119 for every channel and object (`W₂ = 0`, `b₂ = −2`).

| Seed | split | mean | SD | median | g < 0.1 | g > 0.5 | g > 0.9 | channels with mean > 0.5 | channels with mean < 0.1 | across-object SD per channel | final mean `b₂` | ‖W₂‖ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | test | 0.256 | 0.267 | 0.133 | 42.8% | 19.8% | 2.4% | 52 / 256 | 110 | 0.017 | −0.83 | 2.36 |
| 1 | test | 0.283 | 0.350 | 0.078 | 54.4% | 26.9% | 11.0% | 69 / 256 | 140 | 0.011 | −0.74 | 2.36 |
| 2 | test | 0.313 | 0.416 | 0.023 | 62.2% | 32.4% | 23.9% | 83 / 256 | 160 | 0.006 | −0.50 | 2.14 |

Validation means agree with test within 0.002 and the fractions within 0.6 pp (see
[`diagnostics.md`](../results/hw3/diagnostics.md)). Readings:

1. **The gate did not stay at max pooling.** It became strongly bimodal:
   20–32% of channels moved toward the mean (some to g > 0.9) while 43–62% of
   (object, channel) values moved *below* the initial 0.119 toward pure max.
   L2 weight decay pulls `b₂` toward 0 (g → 0.5) uniformly, so it cannot produce
   this two-sided split; the channel specialisation is learned from the data.
2. **The gate is almost not data-dependent.** For a given channel its value
   varies across objects with SD 0.006–0.017, against 0.27–0.42 across channels,
   and the per-class mean gate differs by less than 0.01 between classes
   (seed 0: table 0.251, desk 0.254, dresser 0.259, night_stand 0.257). The
   SENet-style *per-object* recalibration was not used; E2 learned a static
   per-channel max/mean mixture — in effect a learned version of E1.

Figure: [`gates.png`](../results/hw3/gates.png).

## P1 diagnostics

### Gradient participation (test split, eval mode)

| Condition | points with non-zero gradient | critical (argmax) points | participation ratio | gradient share off the argmax | share of point-gradient norm on critical points |
|---|---:|---:|---:|---:|---:|
| E0 | 13.1% | 13.1% | 0.079 | 0% | 100% |
| E1 | 100% | 13.7% | 0.098 | 50.0% | 91% |
| E2 | 100% | 13.3% | 0.075 | 24.9% | 96% |
| E3 | 14.1% | 14.1% | 0.083 | 0% | 100% |

Max pooling sends gradient to ≈ 134 of 1,024 points per object. Dual pooling
reaches every point, but the mean spreads its share as 1/N per point, so the
critical points still carry 91–96% of the per-point gradient norm and the
participation ratio barely moves (0.079 → 0.098 / 0.075). The "non-maximal
points receive no learning signal" argument is true in support but weak in
magnitude.

### Fewer input points (no retraining; fixed subset, re-normalised)

| Condition | 1,024 | 512 | 256 | 4-pair errors at 1,024 / 512 / 256 |
|---|---:|---:|---:|---:|
| E0 | 92.77 ± 0.60 | 91.48 ± 0.63 | 89.61 ± 0.54 | 42.7 / 47.7 / 55.0 |
| E1 | 92.66 ± 0.26 | 91.81 ± 0.05 | 89.83 ± 0.05 | 40.0 / 42.3 / 49.0 |
| E2 | 92.77 ± 0.32 | 91.92 ± 0.32 | **90.75 ± 0.18** | 44.7 / 45.7 / 48.3 |
| E3 | 91.81 ± 0.34 | 91.19 ± 0.50 | 89.24 ± 0.81 | 51.7 / 50.0 / 61.3 |

Per seed at 256 points, E2 − E0 = +1.21, +1.98, +0.22 pp (3/3 seeds). This was
the pre-registered secondary prediction (design §8): a channel max over fewer
points can only shrink, while a mean is unbiased. E2 loses 2.0 pp from 1,024 to
256 points, E0 3.2 pp. One fixed subset per point count was evaluated.
Figure: [`robustness.png`](../results/hw3/robustness.png).

## Post-hoc probe of the alternative explanations (not pre-registered)

`python -m app.classification.hw3 probe` — no network: the best single
threshold on one feature, chosen on the official training pool and scored on
the official test split (balanced accuracy, 50% = chance). ModelNet10 axes:
x = depth, y = width, z = height. The per-axis scale augmentation adds noise
with SD 0.18 to every log aspect ratio; the "augmented" columns apply it to both
training and test features.

| Pair | Feature | Cohen's d (train) | test bal. acc. | augmented: d | augmented: bal. acc. |
|---|---|---:|---:|---:|---:|
| dresser vs night_stand | log depth/width (x/y) | −1.32 | 81.4% | −1.08 | 73.3% |
| dresser vs night_stand | log height/depth (z/x) | +1.05 | 72.7% | +0.91 | 68.2% |
| dresser vs night_stand | log absolute height (raw mesh units) | +0.78 | **85.5%** | removed by normalisation | |
| dresser vs night_stand | log absolute size (raw bbox diagonal) | +0.72 | 83.7% | removed by normalisation | |
| table vs desk | log height/depth (z/x) | −0.71 | 46.7% | −0.66 | 47.3% |
| table vs desk | log depth/width (x/y) | +0.46 | 45.6% | +0.41 | 41.9% |
| table vs desk | log absolute height (raw mesh units) | −0.51 | 55.5% | removed by normalisation | |

(All features: [`probe.md`](../results/hw3/probe.md).)

- **dresser vs night_stand:** a single number that the network never sees —
  the object's absolute height in file units (dressers ≈ 1.5× taller on average)
  — separates the pair with 85.5% balanced accuracy, close to the network's own
  within-pair balanced accuracy (E0 ≈ 87%: 11.0 and 11.3 of 86 objects confused).
  The best proportion cue (depth/width) loses 8 points of balanced accuracy
  under the training augmentation. Both cues are removed or blurred *before*
  pooling, so no aggregation change can restore them.
- **table vs desk:** neither size nor bounding-box proportions separate the
  pair on the test split (≈ chance), and the train-pool differences do not
  transfer to the test split. The distinction, where it exists, is local
  structure (drawers, side or modesty panels) — what max pooling already
  captures — or label ambiguity.

## Interpretation against the pre-registered rules

| Rule (design §9) | Observed | Verdict |
|---|---|---|
| E2 − E0 ≥ +0.5 pp and E2 − E3 ≥ +0.5 pp, ≥ 2/3 seeds, fewer critical errors | E2 − E0 = +0.00 pp; critical errors +2.0 | **not met** |
| E1 > E0 by ≥ 0.5 pp | −0.11 pp | mean information alone does not help |
| E2 > E1 by ≥ 0.5 pp | +0.11 pp | learned selection does not beat fixed mixing |
| E2 ≈ E3, both > E0 (capacity) | E3 < E0 by 0.95 pp (0/3 seeds) | extra width does not help; capacity explains nothing here |
| gate stays near 0.119 or moves to 0 | bimodal, static per channel | the network *uses* mean statistics on a subset of channels, but not per object |
| critical pairs change while accuracy does not | dresser→night ↓ 2.7, night→dresser ↑ 4.0 (E2) | the boundary between the pair *moved*; the pair is not better separated |
| all within ±0.5 pp | E0, E1, E2 within 0.11 pp | **null result for clean accuracy** |

**Conclusion.** The hypothesis — that adding learned, channel-wise mean
information to max pooling reduces the proportion-defined confusions — is not
supported. The module is not inert: the gate learned a stable max/mean channel
split and made the classifier more robust to sparser inputs (+1.1 pp at 256
points, all seeds). But the confused pairs are limited by information that is
already gone before pooling (absolute size; proportions blurred by the
augmentation) or that is local rather than distributional (table/desk).

**Alternative explanations still open.** (1) Seed variability: paired runs flip
15–39 test objects, so 3 seeds resolve ≈ 1 pp, not 0.1 pp. (2) E3's deficit
(−0.95 pp) shows that a harmless-looking change of initialisation and width
moves the result by about 1 pp — HW2's 93.6% seed 0 was a favourable draw.
(3) The gate may need a different input (e.g. per-object statistics that
actually vary, such as the normalisation radius) to become data-dependent.

**Next controlled experiment.** E4 = E0 plus one scalar, `log r` (the
normalisation radius, i.e. absolute size), concatenated to the pooled feature
before the head (+128 parameters); seeds 0–2; everything else fixed; primary
outcome: dresser↔night_stand errors. Then E5 = isotropic scale augmentation (one
factor for all three axes) to test the augmentation explanation separately.
Caveat for E4: ModelNet's units are per-model CAD units, and a size-aware
classifier depends on the units of uploaded objects.

## Environment notes

- `scripts/smoke_backend.py` passes all checks; in the HW3 environment Uvicorn
  prints a `TypeError` traceback while restoring signal handlers *after* the
  test server has shut down (not seen in the HW2 log; environment-related, does
  not affect any result).
- Final verification (job 22558304, commit `d8fb406`): `npm ci`, `tsc`, ESLint and
  the production build pass; the complete browser suite passes (139 tests,
  including the classification spec with the bundled HW3 model and the
  interactive-training case on the real ModelNet10 cache); sample integrity,
  backend suite (127 passed), backend smoke, whitespace check and static
  production smoke pass.
