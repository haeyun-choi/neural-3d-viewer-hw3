# HW3 experiments

Design, hypothesis and decision rules were fixed beforehand in
[`hw3-design.md`](hw3-design.md) (commit `40c78fb`, before any HW3 training;
see [commit SHAs](#commit-shas-recorded-in-the-artifacts)).
This file records how the runs were produced and what they showed.

## Environment

| Item | Value |
| --- | --- |
| Code | branch `feature/hw3-gated-pooling`; every training run executed commit **`14bc0d0`** (now `74ab502`, same tree; [see below](#commit-shas-recorded-in-the-artifacts)) from a detached Git worktree, so later documentation commits could not change the running code (`git_dirty: false` in every `metrics.json`) |
| Python / packages | Python 3.12.9 (conda-forge), torch 2.14.1+cu130 executed on CPU, numpy 2.5.4, matplotlib 3.11.2; `pip freeze` identical to the HW2 environment (pins in `backend/requirements-dev.txt`). The HW2 virtual environment could not be reused because its base interpreter was removed from the cluster, so it was rebuilt with the same pins |
| Hardware | UVA HPC, Slurm `interactive` partition restricted to `rivanna` nodes: **Intel Xeon Gold 6248 @ 2.50 GHz**, 8 CPU threads (`OMP_NUM_THREADS=8`) and 16 GB per run, three runs at a time on one node — the same CPU model and thread count as the HW2 runs. No GPU was used: the development host has none, a GPU job would have started hours later, and CPU execution on the HW2 CPU model makes the E0 rerun directly comparable with HW2 |
| Data | processed cache `modelnet10_1024pts_seed0_v1.npz` from HW2 (49,267,465 bytes, SHA-256 `a522b3f1e6137005ab4ccd554ff84549e045108e306bae3a5a9ed25214a8002c`), built from `ModelNet10.zip` SHA-256 `9d867943…07d51`; train 3,193 / val 798 / test 908 |
| Jobs | environment 22556128; verification 22556241 (backend tests, backend smoke, 1-epoch smoke of E0–E3 with `--skip-test`); matrix **22556284** (array 0–11, at most 3 concurrent) |

### Commit SHAs recorded in the artifacts

After all runs and analyses, the HW3 commit messages were rewritten once to
remove co-author trailers. Only the messages changed: every rewritten commit
keeps its tree (file contents), parent, author and dates, so each SHA recorded
at run time still identifies the exact code through its tree. The artifacts
were not edited and keep the SHAs recorded at run time.

| SHA recorded at run time | Where it is recorded | Commit in this history | Tree (identical) |
| --- | --- | --- | --- |
| `14bc0d0` | all 12 training runs: `git_sha` in every `metrics.json`, `summary.json`, `summary_runs.csv` and the bundled E2 checkpoint | `74ab502` | `b3de551463fe` |
| `a8b6874` | `generated_with` in `results/hw3/summary.json` and `diagnostics.json` | `9c7c2d8` | `85f12000c396` |
| `969fcfe` | the design-record commit, as cited here before the rewrite | `40c78fb` | `789afbe8bdf2` |

Check with `git rev-parse <commit>^{tree}`.

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
- **Exact McNemar test**: computed **separately for each seed**, on the paired
  predictions of two runs for the same 908 test objects (re-scored from the
  saved checkpoints). *b*, *c* = objects that only one of the two runs
  classifies correctly; two-sided p = min(1, 2·P[Bin(b+c, ½) ≤ min(b, c)]). This
  is the exact binomial form, so no continuity correction applies; the
  asymptotic χ² with continuity correction gives nearly the same values. The
  three seeds are never concatenated (the same 908 objects recur in every
  seed), and the p-values are not adjusted for multiple comparisons. An
  independent re-implementation reproduced every count and p-value.
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

All 12 runs: commit `14bc0d0` (tree-identical to `74ab502`), `git_dirty: false`, Intel Xeon Gold 6248, 8 threads.
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
package versions make the CPU computation deterministic, so the E1–E3 versus E0
differences below reflect the change itself or the seed-to-seed training
variability that the three seeds measure, not a drift in code, data or
environment.

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
| E3 − E0 | −0.95 pp | −1.98, −0.77, −0.11 | 0/3 | +9.0 | 0.0005, 0.35, 1.00 |
| E2 − E1 | +0.11 pp | +0.33, +0.11, −0.11 | 2/1 | +4.7 | 0.76, 1.00, 1.00 |
| E2 − E3 | +0.95 pp | +1.54, +0.88, +0.44 | 3/0 | −7.0 | 0.03, 0.28, 0.52 |

Even with identical initial shared weights and identical batches, E2 and E0
disagree on 28, 39 and 15 test objects (seeds 0–2); the accuracy differences
are the small imbalance between two large sets of flips. Error redistribution
within the confused pairs: E1 lowered table→desk (16.0 → 11.3) but none of the
other three directions; E2 lowered dresser→night_stand (11.0 → 8.3) and raised
night_stand→dresser (11.3 → 15.3). Four-direction total per seed: E0 38 / 44 /
46, E1 37 / 44 / 39, E2 48 / 43 / 43 — no consistent reduction.

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

The implementation and training worked as designed: the tests pass, every
parameter receives finite gradients, and the gate moved far from its
initialisation (‖W₂‖ 0 → 2.1–2.4). **The learned gate did not behave as
hypothesized**, however:

1. **Not an adaptive mixture but a bimodal channel split.** 52–83 of 256
   channels (20–32%) moved toward mean pooling (channel mean g > 0.5; 6–62 of
   them above 0.9), while 110–160 channels (43–63%) moved toward pure max
   pooling (channel mean g < 0.1, below the initial 0.119). L2 weight decay pulls
   `b₂` toward 0 (g → 0.5) uniformly, so it cannot produce this two-sided split.
2. **Almost no object dependence.** For a given channel the gate varies across
   objects with SD 0.006–0.017, against 0.27–0.42 across channels, and the
   per-class mean gate differs by less than 0.01 between classes (seed 0:
   table 0.251, desk 0.254, dresser 0.259, night_stand 0.257). The SENet-style
   *per-object* recalibration did not emerge; E2 learned a nearly static
   per-channel max/mean mixture — in effect a learned version of E1.

What failed is the adaptive-mixture hypothesis, not the implementation.

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

**Exploratory.** Per seed at 256 points, E2 − E0 = +1.21, +1.98, +0.22 pp
(higher in 3/3 seeds); E2 loses 2.0 pp from 1,024 to 256 points, E0 3.2 pp. The
direction was stated beforehand as a secondary prediction (design §8: a
channel max over fewer points can only shrink, a mean does not), but only one
fixed subset per point count was evaluated, with three seeds and no repeated
subsampling. Exploratory evidence therefore suggests — and does not establish —
that the proposed pooling may be more robust when fewer points are available.
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
  Absolute size never reaches the network, so no aggregation can use it; the
  dresser/night_stand confusion is consistent with this. The best proportion cue
  (depth/width) loses 8 points of balanced accuracy under the training
  augmentation; it does reach the network at test time, but the training
  augmentation may teach the network to discount it.
- **table vs desk:** neither size nor bounding-box proportions separate the
  pair on the test split (≈ chance), and the train-pool differences do not
  transfer to the test split. This suggests that the distinction, where it
  exists, is local structure (drawers, side or modesty panels) — what max
  pooling already targets — or label ambiguity.

These probes are post-hoc diagnostics without retraining; they are consistent
with, but do not prove, the alternative explanations.

## Interpretation against the pre-registered rules

| Rule (design §9) | Observed | Verdict |
|---|---|---|
| E2 − E0 ≥ +0.5 pp and E2 − E3 ≥ +0.5 pp, ≥ 2/3 seeds, fewer critical errors | E2 − E0 = +0.00 pp; critical errors +2.0 | **not met** |
| E1 > E0 by ≥ 0.5 pp | −0.11 pp | mean information alone does not help |
| E2 > E1 by ≥ 0.5 pp | +0.11 pp | learned selection does not beat fixed mixing |
| E2 ≈ E3, both > E0 (capacity) | E3 < E0 by 0.95 pp (0/3 seeds) | added parameters alone did not help |
| gate stays near 0.119 or moves to 0 | bimodal, nearly static per channel | the gate did not behave as hypothesized: mean statistics on 20–32% of channels, max on 43–63%, not adapted per object |
| critical pairs change while accuracy does not | E2: dresser→night ↓ 2.7, night→dresser ↑ 4.0; E1: table→desk ↓ 4.7, others ↑ | errors were redistributed within the pairs; no consistent reduction of the total |
| all within ±0.5 pp | E0, E1, E2 within 0.11 pp | **null result for clean accuracy** |

**Conclusion.** The hypothesis — that adding learned, channel-wise mean
information to max pooling reduces the proportion-defined confusions — is not
supported. Confirmatory: no clean-accuracy gain (E2 − E0 = +0.00 pp), no gain
from added parameters (E3 −0.95 pp), errors redistributed rather than reduced,
and a gate that settled into a nearly static channel split instead of the
hypothesized adaptive mixture. Exploratory: E2 may be more robust with fewer
points (+1.1 pp at 256 points). Post-hoc probes suggest that the confused pairs
may be limited by information the preprocessing removes (absolute size) or the
training augmentation perturbs (proportions), or by local rather than
distributional differences (table/desk).

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

- **Uvicorn shutdown traceback (fixed in the smoke harness).** Inside Slurm jobs
  on this cluster every process is started with
  `LD_PRELOAD=…/libxalt_init.so` (XALT job monitoring), which installs a native
  SIGTERM handler before Python starts: `/proc/self/status` shows SIGTERM caught,
  and `signal.getsignal(SIGTERM)` returns `None` (with the preload removed it is
  `SIG_DFL`). Uvicorn stores the previous handlers when it starts serving and
  re-installs them on shutdown; `signal.signal(SIGTERM, None)` raises
  `TypeError`, so the server exited with code 1 and a traceback after a
  successful shutdown, whether it was stopped with SIGINT or SIGTERM. The smoke
  harness now (1) starts the normal `uvicorn` entry point through a launcher
  that replaces only such an unrepresentable SIGTERM handler with `SIG_DFL`
  before Uvicorn starts, (2) stops the server with SIGINT (Ctrl+C), and
  (3) requires exit code 0 and no traceback in the server log. Checked in
  job 22559024: old launcher → exit 1 with traceback for SIGINT and SIGTERM;
  new launcher → exit 0 for SIGINT and 143 (re-raised SIGTERM, by Uvicorn's
  design) for SIGTERM, no traceback. Running `uvicorn app.main:app` by hand in
  such a job still prints the traceback on Ctrl+C; it is cosmetic.
- **Stale-prediction race in the classifier panel (fixed).** The final audit's
  first complete browser run failed once in the upload test: the panel enabled
  *Predict* while a newly chosen file was still being opened in the viewer, so a
  click could send the *previous* input, whose result then appeared under the
  new file name (a pre-existing HW2 race, exposed by a slower node). *Predict*
  is now disabled while an input is being opened, and a response that arrives
  after the input or model changed is discarded.
- Final verification of the submitted state: job 22559532 — focused HW3 tests
  (41 passed), full backend suite (127 passed), backend smoke with graceful
  shutdown, sample integrity, JSON/CSV consistency (56 JSON files; summary CSV =
  summary JSON = the 12 `metrics.json`, all from `14bc0d0`, clean); job
  22559915 after the panel fix — `tsc`, ESLint and build pass, the
  classification spec passes 25/25 (5 repetitions), the complete browser suite
  passes (139 tests), the static production smoke passes; `git diff --check` is
  clean.
