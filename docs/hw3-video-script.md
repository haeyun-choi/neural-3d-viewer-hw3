# HW3 video script (target 2:35–2:45, hard limit 3:00)

Every number is from the completed runs ([`docs/hw3-experiments.md`](hw3-experiments.md)).
Timestamps assume about 145 spoken words per minute plus one second per screen change.
The model description is kept short; 1:26–2:45 (about half of the video) goes to results and failure analysis.

| Time | Segment and on-screen action | Narration |
| --- | --- | --- |
| **0:00–0:25** | Input and HW2 baseline. Web viewer: open the bundled ModelNet10 dresser sample and orbit it; then README §1 (input bullets: B × 1,024 × 3). | "This is my HW2 classifier in the point-cloud viewer. Each ModelNet10 mesh becomes 1,024 surface points, and each point is only x, y, z, so a batch is B by 1,024 by 3. Points are centred and scaled to the unit sphere. A shared per-point MLP, one global max pool and a small head reach 92.8 percent test accuracy." |
| **0:25–0:49** | Observed limitation. README §3: highlight the seed-0 confusions (table→desk 12, dresser↔night stand 11 / 11). | "Most errors are look-alike pairs: tables called desks, and dressers and night stands confused both ways. Max pooling keeps only the strongest point in each channel, so a feature on one point counts the same as one on nine hundred. My hypothesis: the network is missing how much of the object shows a feature." |
| **0:49–1:07** | ResNet / SENet motivation and the proposed pooling. README §5 equation block, then the pooling diagram in the report (point at the max path and the gate). | "From ResNet I took learning a correction around a working path; from SENet, a channel-wise sigmoid gate. My pooling keeps the max and adds a gated move toward the mean. The gate starts near 0.12, so training begins close to max pooling." |
| **1:07–1:26** | Controlled experiment E0–E3. README §6 experiment table (E0–E3), then the line "E0 reproduces HW2 exactly". | "I ran four conditions with three seeds each: E0, the original model; E1, a fixed fifty-fifty mix; E2, my gated pooling; and E3, a wider max-pooling model with E2's parameter count. Everything else was identical, and the E0 rerun reproduced HW2 exactly." |
| **1:26–1:59** | Confirmatory results: no clean gain, gate behaviour. results/hw3/results.png (accuracy dots, then the four error bars); then results/hw3/gates.png. | "The hypothesis was not supported. E2 averaged 92.77 percent, exactly like E0, and no seed showed a significant difference. E3 was worse, so adding parameters alone did not help. The confused pairs were redistributed, not reduced: fewer dressers called night stands, but more night stands called dressers. And the learned gate did not behave as hypothesized. Instead of adapting to each object, it settled into a nearly fixed split: some channels moved toward mean pooling, more toward pure max." |
| **1:59–2:14** | Exploratory 256-point result. results/hw3/robustness.png; point at the 256-point end. | "One exploratory result: with only 256 points per object, E2 was more accurate in every seed, by about one point on average. That suggests possible robustness to sparse inputs, not a proven effect." |
| **2:14–2:33** | Alternative explanation (post-hoc probe). README §8 probe table (absolute height 85.5%; table vs desk 42–56%). | "Why did pooling not help these pairs? A post-hoc probe suggests one reason. Absolute height alone separates dressers from night stands 85 percent of the time, but normalisation removes it before the network sees the points. Table versus desk has no such global cue." |
| **2:33–2:45** | Lesson and next experiment. Report §5 "Next controlled experiment", then the repository front page. | "The lesson: a module can train correctly and still miss its target. Next, I would give the network the object's scale as one extra input. Thank you." |

## Wording guardrails

- Say "not supported" for the hypothesis; do not call the gate or the code a failure.
- Call the 256-point result *exploratory*; call the probe *post-hoc*; use "suggests", never "proves".
- Mention both directions of the dresser/night-stand change, as the narration does.

## Recording checklist

- Open beforehand: the running app with the dresser sample, `README.md` (HW3
  sections §1–§8), `docs/hw3-report.pdf`, `results/hw3/results.png`,
  `results/hw3/gates.png`, `results/hw3/robustness.png`.
- Rehearse once with a timer; if a take runs past 2:50, drop the last sentence
  of the 1:26–1:59 segment ("some channels moved toward mean pooling, more toward pure max").

## Length

- Narration: **380 words**.
- Estimated duration: **2:37 of speech at 145 words/min, about 2:45 including 8 screen transitions** (≈ 2:40–2:57 for 150–135 words/min).
