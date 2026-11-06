# HW3 post-hoc probe: size and proportion cues for the confused pairs

Not pre-registered; no network is involved. Best single-threshold classifier per feature, chosen on the official training pool, scored on the official test split (balanced accuracy; 50% = chance). The per-axis scale augmentation adds noise with SD 0.182 to every log aspect ratio.

| Pair | Feature | Cohen's d (train) | test bal. acc. | with augmentation: d | with augmentation: bal. acc. |
|---|---|---:|---:|---:|---:|
| dresser vs night_stand | log height/depth (z/x) | +1.05 | 72.7% | +0.91 | 68.2% |
| dresser vs night_stand | log height/width (z/y) | +0.04 | 51.7% | +0.03 | 50.1% |
| dresser vs night_stand | log depth/width (x/y) | -1.32 | 81.4% | -1.08 | 73.3% |
| dresser vs night_stand | log absolute size (raw bbox diagonal) | +0.72 | 83.7% | n/a (removed by normalisation) | n/a |
| dresser vs night_stand | log absolute height (raw z extent) | +0.78 | 85.5% | n/a (removed by normalisation) | n/a |
| table vs desk | log height/depth (z/x) | -0.71 | 46.7% | -0.66 | 47.3% |
| table vs desk | log height/width (z/y) | -0.27 | 50.9% | -0.25 | 51.6% |
| table vs desk | log depth/width (x/y) | +0.46 | 45.6% | +0.41 | 41.9% |
| table vs desk | log absolute size (raw bbox diagonal) | -0.31 | 51.8% | n/a (removed by normalisation) | n/a |
| table vs desk | log absolute height (raw z extent) | -0.51 | 55.5% | n/a (removed by normalisation) | n/a |
