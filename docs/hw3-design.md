# HW3 design record: Residual Gated Dual Pooling (written before any HW3 result)

This document fixes the hypothesis, the design, the experiment matrix and the
decision rules **before** any HW3 model was trained. It was committed to Git
before the first HW3 training job started, so the hypothesis is separated from
the observations in the history. Post-experiment results and their
interpretation live in [`hw3-experiments.md`](hw3-experiments.md) and the
report; nothing below is edited after the results except for the clearly marked
addendum at the end.

## 1. Existing point representation (verified in code)

| Item | Value | Code |
| --- | --- | --- |
| Task | ModelNet10, 10-class object classification | `dataset.py` |
| Raw data | OFF triangle meshes (official split 3,991 train / 908 test) | `dataset.py` |
| Sampling | 1,024 points per mesh on the surface, triangle chosen with probability ∝ area, uniform barycentric point inside it, per-object seed = CRC of the path | `geometry.sample_surface` |
| Per-point input | `[x, y, z]` only — no RGB, no normal, no connectivity, no per-point label | `geometry.points_from_geometry` |
| Network input | `(B, 1024, 3)` = (objects in batch, points per object, coordinates); B = 32 in training | `training.py` |
| Label | one class index per object, stored separately | `dataset.PreparedData` |
| Split | official train pool split 80/20 stratified (seed 0): **3,193 train / 798 val / 908 test** | `dataset.stratified_split` |

### Normalisation (applied once, at caching and inference time)

For the sampled points `p_i`, `i = 1…N`:

```
μ  = (1/N) Σ_i p_i                         centroid
q_i = p_i − μ                              centred points
r  = max_i ‖q_i‖₂                          radius
p̂_i = q_i / r                              unit-sphere points
```

This removes global translation and overall (absolute) scale, and keeps relative
proportions. A night stand and a dresser of identical shape but different size
become the same input.

### Training-only augmentation (`training.augment`, per object in the batch)

- anisotropic scale: `s ∈ R³`, each axis `s_k ~ U(0.8, 1.25)` independently;
- translation: `t ∈ R³`, each axis `t_k ~ U(−0.1, 0.1)`;
- jitter: `ε_ik ~ N(0, 0.01²)` per coordinate, clipped to `[−0.03, 0.03]`;
- `x_i = p̂_i ⊙ s + t + ε_i`; **no rotation** (ModelNet10 is upright and aligned).

Trade-off: the per-axis scale changes aspect ratios by up to 1.25/0.8 = 1.56×
between two axes. It makes the network robust to proportions — and may
therefore blur exactly the proportion cues that separate `dresser` from
`night_stand` and `desk` from `table`.

## 2. Baseline architecture (HW2 Mini-PointNet, h = 64)

```
points (B, N, 3) ──transpose──▶ (B, 3, N)
  shared MLP, Conv1d(k=1, bias=False) → BatchNorm1d → ReLU, per point:
      3 → 64 → 128 → 256                         h_i ∈ R²⁵⁶ for every point i
  global max pool over the N points:  m_c = max_i h_{i,c}       (B, 256)
  head: Linear 256→128 (no bias) → BatchNorm1d → ReLU → Dropout 0.3 → Linear 128→10
logits (B, 10)
```

76,362 parameters. Cross-entropy, Adam (lr 0.001, weight decay 1e-4), cosine
schedule to 1% of the start rate, 60 epochs, batch 32, checkpoint = epoch with
the best validation accuracy (ties: lower validation loss), test split scored
once with that checkpoint.

## 3. Observed errors (HW2, stored metrics, official test split)

| HW2 Mini-PointNet seed | test acc | table→desk | desk→table | dresser→night_stand | night_stand→dresser |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (bundled) | 93.61% | 12 | 4 | 11 | 11 |
| 1 | 92.29% | 17 | 5 | 10 | 12 |
| 2 | 92.40% | 19 | 4 | 12 | 11 |

Mean ± SD (population, n = 3): 92.77% ± 0.60%. In seed 0, these four
directional errors are 38 of the 58 misclassified test objects (66%). Test
supports: table 100, desk 86, dresser 86, night_stand 86 objects.

## 4. The limitation we target: max pooling as the only aggregator

For the final per-point features `h_i ∈ R^C` (C = 256), the network sees only

```
m_c = max_i h_{i,c}.
```

Strengths: permutation invariant, keeps strong local responses, works for any N.
What it discards:

- **prevalence** — a feature present on 1 point or on 900 points gives the same `m_c`;
- **distribution** — how much of the surface looks like a drawer front, a thin leg
  or a flat top is lost; only the most extreme point per channel survives;
- **gradient participation** — `∂m_c/∂h_{i,c} = 1[i = argmax_j h_{j,c}]`, so through
  pooling each channel sends gradient to exactly one point; points that are never
  an argmax ("non-critical" points) receive no direct learning signal;
- **global proportions** — "how much table-top relative to legs" is a statistic over
  points, not an extreme value.

The confused pairs differ mostly in proportion or extent (a desk is a table with
drawers/side panels; a night stand is a short, near-cubic dresser). It is
therefore *plausible* that a prevalence-sensitive statistic helps. This is a
hypothesis to test; nothing so far shows that max pooling *causes* these errors.

## 5. Design principles from class

Only two course papers are used, and only for their principles; the HW3 module
is **not** a ResNet or an SENet.

- **ResNet** — K. He, X. Zhang, S. Ren, J. Sun, *Deep Residual Learning for Image
  Recognition*, CVPR 2016.
  <https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html>
  Principle used: keep a mapping that already works as the identity path and
  learn a **correction** around it (`y = x + F(x)`), so that "no change" is easy
  to represent.
- **SENet** — J. Hu, L. Shen, G. Sun, *Squeeze-and-Excitation Networks*, CVPR 2018.
  <https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html>
  Principle used: **squeeze** global information into a descriptor and
  **excite** — a small bottleneck MLP with a sigmoid produces a data-dependent,
  channel-wise recalibration instead of treating all channels identically.

Mixing max and mean statistics is not new in deep learning in general. The HW3
contribution is a deliberate, hypothesis-driven adaptation of these two
principles to this classifier's point-set aggregation, and its controlled test.

## 6. Proposed module: Residual Gated Dual Pooling (E2)

For `h ∈ R^{B×C×N}` (C = 256, N = 1024):

```
m = max_N h                 (B, C)   identity path: the HW2 descriptor
a = mean_N h                (B, C)   prevalence / distribution descriptor
g = σ( W₂ ReLU( W₁ [m; a] + b₁ ) + b₂ )      (B, C), g ∈ (0, 1)
     W₁: 512 → 16,  W₂: 16 → 256
z = m + g ⊙ (a − m)  =  (1 − g) ⊙ m + g ⊙ a
logits = head(z)            head unchanged
```

- `g ≈ 0` keeps max pooling, `g ≈ 1` uses mean pooling, intermediate values
  interpolate **per channel and per object**.
- ResNet reading: `m` is the identity path, `g ⊙ (a − m)` is the learned
  correction. Since `a ≤ m` element-wise, the correction only moves a channel
  from its extreme value toward its average, and `z_c ∈ [a_c, m_c]`.
- SENet reading: `[m; a]` is the squeeze (two global descriptors, because the
  decision is "which descriptor"), the 512→16→256 bottleneck with a sigmoid is
  the excitation; 16 = 256/16 is SENet's default reduction ratio applied to the
  256 output channels. The gate's hidden activation is ReLU, as in SENet.
- **Baseline-preserving initialisation (a deliberate choice, not a tuned
  hyperparameter):** `W₂ = 0`, `b₂ = −2.0`, so at initialisation
  `g = σ(−2) ≈ 0.119` for every channel and object, i.e. `z ≈ 0.881 m + 0.119 a`
  — close to the HW2 max pooling. `b₂ = −2` instead of a very negative value
  keeps the sigmoid out of saturation: `σ'(−2) ≈ 0.105`, 42% of the maximum
  slope, so the gate can still learn to open. With `W₂ = 0`, `W₁` receives no
  gradient on the first step; `W₂` does, and `W₁` starts learning from the
  second step (the same "start the correction at zero" logic as a residual
  branch). `W₁`, `b₁` use PyTorch's default `nn.Linear` initialisation.
- Gate parameters: 512·16 + 16 + 16·256 + 256 = **12,560** → E2 has
  76,362 + 12,560 = **88,922** parameters.
- Everything else is shared: the same optimizer, weight decay (also on the gate
  parameters), schedule and dropout. Note for the analysis: L2 weight decay pulls
  `b₂` toward 0, i.e. toward `g = 0.5`; the final `b₂` is recorded to check that
  any gate opening is not just this drift.

### Why this is a fair "near-nested" extension

E2 can represent (approximately) E0 by keeping `g → 0`; with the baseline-like
initialisation it starts there. If E2 ends up worse than E0, the reason is not
that the function class lost max pooling, but optimisation, regularisation or a
genuine preference of the data — which is itself informative.

## 7. Experiment matrix and controls

| ID | Model name (code) | Pooling | h | Params (expected) | Question |
| --- | --- | --- | ---: | ---: | --- |
| E0 | `mini_pointnet` | max | 64 | 76,362 | rerun of HW2 under the HW3 code/env |
| E1 | `mini_pointnet_fixed_dual` | `z = 0.5 m + 0.5 a`, no new parameters | 64 | 76,362 | does mean information itself help? |
| E2 | `mini_pointnet_gated_dual` | residual gated dual (above) | 64 | 88,922 | does learned channel-wise selection beat fixed mixing? |
| E3 | `mini_pointnet --hidden-dim 69` | max | 69 | 88,537 | is an E2 gain explained by +12.5k parameters? |

All three pooling conditions are instances of `z = m + g ⊙ (a − m)` with
`g ≡ 0` (E0), `g ≡ 0.5` (E1) and `g` learned (E2).

Seeds 0, 1, 2 for every condition (12 runs). Held fixed for all runs:
the processed cache `modelnet10_1024pts_seed0_v1.npz` (same 1,024 points per
object, same train/val/test indices), the training augmentation, 60 epochs,
batch 32, Adam lr 0.001, weight decay 1e-4, cosine schedule, ReLU, dropout 0.3,
validation-accuracy checkpoint selection, a single test evaluation of the
selected checkpoint, the same software environment and the same CPU type
(8 threads), as in HW2.

Additional pairing controls (by construction of the code):

- the mini-batch order and the augmentation draws come from a dedicated
  generator seeded with the run seed, so for a given seed **all four
  conditions see the same batches with the same augmentation**;
- E0, E1 and E2 construct the shared point MLP and head first, in the same
  order, so for a given seed they start from **identical shared weights**; E2's
  gate is created afterwards inside a forked RNG state, so the global random
  stream used by dropout is also identical to E0/E1. The only difference within
  a seed is the pooling. (E3 has different tensor shapes and cannot share
  weights; it shares batches and augmentation.)
- No per-model tuning: every condition uses the HW2 learning rate (chosen on
  validation in HW2). The test split is never used for any choice.

E0 is rerun under the HW3 commit and environment and compared with the stored
HW2 numbers as a reproducibility check.

## 8. Measurements

Per run: `config.json`, `history.json`, `metrics.json` (selected epoch, val
accuracy, test accuracy/loss, confusion matrix, per-class precision/recall,
wall-clock training time, git SHA, PyTorch version), `best.pt`, learning
curves, confusion matrix.

Aggregate: mean ± SD over seeds of validation and test accuracy; parameter
count; training time; the four directional errors table→desk, desk→table,
dresser→night_stand, night_stand→dresser; per-seed paired differences (same seed,
same batches); McNemar exact test on per-object test predictions of paired runs.

E2 gate diagnostics on validation and test: mean, SD, min, max, quantiles,
fraction `g < 0.1`, fraction `g > 0.9`, per-channel mean gate, histogram,
the learned `b₂` and `‖W₂‖`.

P1 diagnostics (only after the core results exist):

- **Gradient participation.** In eval mode, `∂L/∂h` at the final per-point
  features (all of it flows through pooling). Reported: fraction of points that
  receive non-zero gradient; for max pooling this is the "critical set" (points
  that are the argmax of at least one channel); the participation ratio
  `(Σ_i w_i)² / (N Σ_i w_i²)` with `w_i = ‖∂L/∂h_i‖₂`; and the share of gradient
  magnitude on non-argmax (point, channel) entries.
- **Point-count robustness.** Deterministic evaluation of the test split at
  N = 512 and 256: a fixed-seed subset of each object's cached 1,024 points,
  re-normalised with the same centroid/radius rule. Secondary prediction: a
  channel max over fewer points can only decrease, while a mean is unbiased, so
  conditions that use the mean should degrade less than E0.

## 9. Pre-registered expectations and decision rules

Noise reference: HW2 seed SD ≈ 0.6 percentage points (pp); one test object =
0.11 pp. With 3 seeds per condition, a difference of a mean below about 0.5 pp
is not distinguishable from seed noise.

| Outcome | Interpretation |
| --- | --- |
| E2 − E0 ≥ +0.5 pp **and** E2 − E3 ≥ +0.5 pp, E2 > E0 in ≥ 2 of 3 seeds, fewer critical-pair errors | **hypothesis supported**: gated mean information helps and is not just capacity |
| E1 > E0 by ≥ 0.5 pp | mean information alone helps |
| E2 > E1 by ≥ 0.5 pp | learned channel-wise selection matters beyond fixed mixing |
| E2 ≈ E3 (within 0.5 pp), both > E0 | extra capacity is a sufficient explanation |
| E2 gate stays near its initial 0.119 or moves toward 0 | the network prefers the max-pooling inductive bias |
| E1/E2 ≤ E0 − 0.5 pp | mean aggregation dilutes sparse discriminative features (**hypothesis contradicted**) |
| accuracy unchanged but critical-pair errors fall in all seeds | the design may target global-proportion distinctions specifically |
| critical-pair errors unchanged | other causes (below) are more likely than aggregation |
| everything within ±0.5 pp | **not supported at this sample size**; report as a null result |

### Expected failure modes and alternative explanations (stated in advance)

1. **Absolute size is removed by normalisation.** Night stands are smaller than
   dressers; after dividing by the radius, only shape and proportion remain. No
   pooling can recover information that is not in the input.
   *Next experiment:* give the head the normalisation radius `r` (log-scale) as
   one extra scalar, everything else fixed.
2. **Anisotropic scale augmentation erases aspect ratios.** A 0.8–1.25
   independent scale per axis can turn a dresser's proportions into a night
   stand's. *Next experiment:* isotropic scale augmentation (one factor for all
   axes), pooling fixed to the best condition.
3. **Max pooling is already the right inductive bias here.** The original
   PointNet design also chose max pooling over averaging. The mean of 1,024
   points is dominated by the most common surfaces (large flat panels shared by
   both classes of a pair), which may dilute the rare, sparse local cues that
   max pooling isolates.
4. **Label ambiguity.** Some ModelNet10 desks and tables are visually
   indistinguishable even to a person; part of the confusion may be irreducible.
5. **Test-set distribution shift and small counts.** The confused classes have
   86–100 test objects; differences of a few objects are within seed noise.

## 10. What is *not* changed

Data, preprocessing, augmentation, optimizer, learning rate, schedule, epochs,
batch size, dropout, head, point MLP, checkpoint rule and test protocol. No
hyperparameter search, no per-model tuning, no test-set feedback.

---

## Addendum (written after the experiments; the sections above are unchanged)

Deviations from this plan, all procedural:

- **Execution venue.** The 12 runs used the Slurm `interactive` partition
  restricted to the same CPU model as HW2 (Intel Xeon Gold 6248, 8 threads),
  because the `standard` partition queue would have delayed the start by days.
  The HW2 virtual environment could no longer start (its base interpreter was
  removed from the cluster); it was rebuilt with the identical package pins.
  E0 then reproduced the HW2 runs exactly, so neither change affected results.
- **One analysis was added after seeing the results**: a network-free probe of
  how well absolute size and bounding-box aspect ratios separate the two
  confused pairs (`python -m app.classification.hw3 probe`). It is reported as
  post-hoc in every document and does not change any decision rule.
- The point-count comparison of §8 (P1) is reported as **exploratory**: its
  direction was predicted here, but it uses one fixed subset per point count and
  is not a confirmatory test of robustness.
- Everything else ran as specified: 4 conditions × 3 seeds, no reruns, no
  tuning, the test split scored once per run, the P1 diagnostics as defined in
  §8. Results and their reading against §9: [`hw3-experiments.md`](hw3-experiments.md).
