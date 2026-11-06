# HW3 video script (≤ 3:00)

Target: about 430 spoken words at ~150 words per minute. Numbers below are the
actual HW3 results ([`docs/hw3-experiments.md`](hw3-experiments.md)).

| Time | On screen | Narration |
| --- | --- | --- |
| **0:00–0:35** | The web viewer: open a bundled ModelNet10 **dresser** sample, orbit it; then README §1–2 (input bullets and the baseline flowchart). | "This is my HW2 classifier inside the point-cloud viewer. Each ModelNet10 object is a CAD mesh; I sample 1,024 points on its surface, and each point is just x, y, z — so a batch is B by 1,024 by 3. Points are centred and scaled to the unit sphere. A shared MLP gives every point a 256-dimensional feature, a global max pool combines them, and a small head predicts ten classes: 92.8 percent test accuracy over three seeds." |
| **0:35–1:05** | HW2 confusion matrix, then README §3: highlight table→desk 12, dresser↔night stand 11 / 11. | "Most mistakes are look-alike pairs: tables called desks, and dressers and night stands confused both ways — 38 of 58 test errors in seed zero. These pairs differ in proportion. Max pooling keeps only the strongest point per channel, so a feature on one point counts the same as on nine hundred. My hypothesis: the network is missing *how much* of the object shows a feature." |
| **1:05–1:35** | README §4–5: the equation block, then the report's pooling diagram; point at the identity path and the gate. | "I took two principles from class. From ResNet: keep what already works as an identity path and learn a correction. From SENet: squeeze global information and learn a channel-wise sigmoid gate. Residual Gated Dual Pooling keeps the max-pooled vector m and adds g times the mean minus the max. A small 512-to-16-to-256 MLP computes g. I initialise it at g ≈ 0.12, so training starts close to the original max pooling." |
| **1:35–2:10** | README §6 experiment table (E0–E3), then the line "E0 reproduces HW2 exactly". | "To test it fairly I ran four conditions with three seeds each. E0 is the original model, rerun. E1 mixes max and mean fifty-fifty with no new parameters. E2 is my gated pooling. E3 is the original model made wider so that it has the same parameter count as E2. Everything else is identical — data, augmentation, optimizer, schedule, even the CPU. For a given seed, E0, E1 and E2 start from the same weights and see the same batches. The rerun E0 reproduced HW2 exactly." |
| **2:10–2:45** | `results/hw3/results.png`, then `gates.png`, then the probe table in README §8. | "It is a null result. E2 averages 92.77 percent — exactly E0 — and the confused pairs did not improve. The wider E3 was worse, so capacity explains nothing. The gate was not idle: it split channels into max-like and mean-like ones, and with only 256 points E2 wins in every seed, by 1.1 points on average. But the split is fixed per channel, not per object. A probe explains why: absolute height alone separates dressers from night stands 85 percent of the time, but normalisation deletes it; table versus desk has no global cue." |
| **2:45–3:00** | Report §5 "Next controlled experiment", then the repository front page. | "The lesson: the bottleneck is the input pipeline, not the aggregation. My next controlled experiment gives the network the normalisation radius as one extra number, and then tests isotropic scaling. Thank you." |

## Recording checklist

- Have open: the running app with a dresser sample, `README.md` (HW3 sections),
  `docs/hw3-report.pdf`, `results/hw3/results.png`, `results/hw3/gates.png`.
- Say "percentage points" only where written; round numbers as spoken above.
- Keep each segment within its time window; total ≤ 3:00.
