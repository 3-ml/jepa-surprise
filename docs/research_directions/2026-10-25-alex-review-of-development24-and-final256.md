# Development24 contrasts and final256 planning analysis

Scientific attribution: Alex. These are dated proposals, not retrospective amendments of the executed final256 protocol. Private correspondence and transfer/task assignments are omitted.

Dated scientific analysis (2026-10-25) of the fresh-scene population, the within-world estimands, precision versus utility, and what threshold performance could justify the "when" claim. All numbers are from the stored `results/25-16-00-wmpai-vasi-development24-complete/scores/all-scores.json` (288 clips), recomputed independently; clean appearance unless stated. Intervals are 95% percentile bootstrap over the 24 configurations (10,000 draws). They are conditional on the development template and on one shared procedural generator.

## 0. What the development data say

Write the four world-and-trajectory cell means as S(solid,stop) 0.492895, S(solid,fall) 0.487254, S(gap,stop) 0.491487, S(gap,fall) 0.484740. Then, per configuration:

- trajectory effect `T` = mean over the two worlds of S(stop) - S(fall) = **+0.006194** [+0.005059, +0.007409];
- label effect `B` (the interaction, the "balanced margin") = **+0.000553** [-0.000669, +0.001756];
- `B/T` = 0.089 [-0.104, +0.300]; the upper limit of `B` is 28% of `T`.

The world margins are `M_solid = B - T` and `M_gap = B + T` (-0.005641 and +0.006747), so they carry the trajectory effect with opposite signs. Persistent hue: `B` = -0.000021 [-0.000904, +0.000851], `T` = +0.007514. Abrupt: `B` = +0.000010 [-0.000609, +0.000637], `T` = +0.008087.

Clip-level AUROC (positive class in brackets): stop versus fall **0.871** [0.814, 0.928] (0.938 pooled over appearances); inconsistent versus lawful **0.532** [0.465, 0.595] (0.512 pooled). The score separates the trajectories and not the physical labels.

## 1. Fresh-scene population

final256 draws radius (0.60 to 0.70), contact time (1.18 to 1.24 s), camera x (-8 to 8), initial x and colour from the development ranges, with the same template, renderer and score. It is a fresh sample of the template's population. It tightens estimates inside the template and cannot widen what the result covers.

- The estimates are already precise: the SD across configurations is 0.0031 for `B` and 0.0030 for `T`. A 95% half-width of 0.001 on `B` needs about 38 configurations; 256 gives a 98.75% half-width of 0.00049. More configurations will move `B` from about +-0.0012 to about +-0.0005 and are very unlikely to change the conclusion.
- Contact times span only 1.18 to 1.24 s, so there are effectively two drop heights. Statements about scenes outside these ranges need the independent generator (`docs/research_directions/2026-10-25-independent-final-generator-requirements-DRAFT.md`), which final256 does not provide.
- Proposed interpretation: final256 is a same-template final test; eligibility counts per factor level delimit its coverage. It does not test generality.

## 2. Within-world estimands

In a crossed 2x2 design only the cross-world contrast removes the trajectory. Each within-world margin is label plus trajectory: `M_solid = B - T` and `M_gap = B + T`.

- The planned reading, "consistent positive mean ordering requires all four lower bounds above 0", needs both `B - T > 0` and `B + T > 0`, that is `B > |T|`: the label effect must exceed the trajectory effect. In development, `B/T` is 0.09 with an upper limit of 0.30, so this reading is expected to fail by construction, and its failure would say nothing about physics beyond what we already see.
- Proposed reporting alternative, without changing frozen data or scoring: report `B` and `T` as the two primary quantities per appearance (the same four cells, two orthogonal contrasts), keep the four world means as derived secondary quantities, and state the reading in advance: `B` within the declared margin of 0 means no label-specific ordering at that resolution, whatever `T` does. Two primary contrasts also pay a smaller multiplicity price than four.

## 3. Precision versus utility

- **Precision is not the constraint.** The 0.001 target is reached at N about 38 for `B` and is well inside what N=256 gives. More scenes do not help the paper's question.
- **Utility depends on noise.** The spread of a score within a cell is `sigma` = 0.003747 (pooled across the four clean cells). Under an equal-variance normal model, AUROC = Phi(Delta / (sqrt 2 x sigma)), so the label effect `Delta` needed is:

| Target AUROC | `Delta` needed | As multiple of observed `B` | As multiple of 0.001 |
|---|---|---|---|
| 0.6 | 0.00134 | 2.4 | 1.3 |
| 0.7 | 0.00278 | 5.0 | 2.8 |
| 0.8 | 0.00446 | 8.1 | 4.5 |
| 0.9 | 0.00679 | 12.3 | 6.8 |

  Observed `B` implies an AUROC of 0.54 and observed `T` implies 0.88 (measured 0.871); the model fits the trajectory effect.
- **Consequence.** The upper limit of `B` (0.00176) is below the 0.7-AUROC requirement (0.00278). Within this template the label-specific signal is too small to support even a modest detector. This is a model-based bound, and the template caveat applies.
- **Proposal:** declare the smallest practically meaningful label effect before scoring, as `Delta` = 0.0028, the effect that would give AUROC 0.7 at the development noise level (a proposal; it comes from development data, not from any final score). The 0.001 yardstick stays a reporting resolution. Under the proposed criterion, a final-run interval for `B` inside +-0.0028 would support practical equivalence only if this criterion had been adopted in advance. Adoption is not established by this planning note.

## 4. What threshold performance could justify the "when" claim

The frozen threshold (0.4858863, fit on clean development clips) behaves as a trajectory detector. Applied unchanged to the clean development clips: it flags **100% of stop clips and 47.9% of fall clips**; false alarm 64.6% (31 of 48 lawful), miss 16.7% (8 of 48 inconsistent), balanced accuracy 59.4% (in-sample; the leave-one-configuration-out values are 64.6%, 20.8%, 57.3%). In the solid world every lawful (stop) clip is flagged and a third of the inconsistent clips are missed; in the gap world there are no misses because the inconsistent clips are the stops.

In a crossed design a trajectory-driven score has label AUROC near 0.5 by construction, and 0.512 is what we see. A threshold result can therefore justify the "when" claim only if it clears bars set in advance on the crossed design. Proposed (numbers to be agreed):

1. Label AUROC (inconsistent versus lawful) on held-out configurations with a configuration-cluster lower bound at or above 0.7.
2. At one frozen threshold, false-alarm and miss rates each at or below 30%, and this in each world separately, so a world-specific trajectory shortcut cannot pass.
3. The same performance under persistent and abrupt appearance (balanced accuracy within 5 points of clean).
4. Performance above chance within each trajectory class (stop clips: lawful in the solid world versus inconsistent in the gap world; fall clips likewise), where the motion is identical and only the context differs.
5. The threshold is fit on development scenes of the final generator, not on development24.

Given the development result, final256 is not expected to meet these bars. State that in advance, so the outcome reads as a pre-declared negative result and not as an exploratory finding. The honest statement the data support today: in this template the score tracks the trajectory strongly (AUROC 0.87 to 0.98) and carries at most about 30% of that amount of label-specific signal (upper limit), too little for a detector.

