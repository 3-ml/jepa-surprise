# Final256 — frozen native V-JEPA2-G crossed-support test

**Status: frozen-before-outcomes protocol, 2026-10-25.** The earlier joint statistics draft was not jointly signed. This protocol, manifest, development threshold, exclusion registry, code and tests were hashed before final rendering/scoring; no final outcomes informed the freeze. This is a public scientific adaptation, not a verbatim execution instruction.

## Question and claim boundary

For this bounded procedural population, does the unchanged native frozen V-JEPA2-G scalar assign higher error to inconsistent than lawful continuations in both visible-support contexts and under clean/persistent hue? How does one clean-development threshold perform unchanged across appearances?

Solid/stop and gap/fall are lawful under the stated support constraints; solid/fall and gap/stop are inconsistent. Cross-world stop trajectories match exactly, as do fall trajectories. Static supports are simulator-fixed; this is not a natural-scene population or an independent physical-knowledge benchmark. Target pixels across worlds need not match (support, shadows and occlusion differ). Mean native error is over the full future/image, not an event-selected region. No architecture/augmentation causality, absence-of-physical-knowledge or equivalence claim follows automatically.

## Sample and freshness

**256 fixed candidates**, IDs final256-0001 through final256-0256, scene seeds 2026102700 through 2026102955. For each seed, Python `random.Random(seed)` draws, in this exact order: radius uniform [0.60,0.70], nominal contact time uniform [1.18,1.24], camera-x from [-8,-4,4,8], initial x uniform [-0.035,0.035], and a colour independently from the four lexicographically sorted development RGB colours. Initial height = radius + 0.5*9.81*contact_time^2, initial y=-0.3. Other settings copy the tested development construction. The full parameter manifest, not this prose, is authoritative.

This samples new radius/time/position combinations rather than relabeling jittered copies of the 24 development configurations. Camera and colour are now independently sampled. Thus the target distribution is not the original fixed grid; the precision calculation is conditional planning, not a coverage guarantee.

No existing RGB, IntPhys2 video or historical Kubric scene is used to generate a final scene. Seed and parameter fingerprints exclude development; target trajectory fingerprints exclude locally retained development24 and first-contact traces. The registry records exact coverage; legacy Kubric8 seeds and archived16 seeds are also excluded. Exact trace comparison against inaccessible historical traces is not falsely claimed. Whole-set normalized-context fingerprints must be disjoint from the available inspected-input registry and other final base scenes before any final score. Any freshness/input-integrity collision stops the run rather than being repaired after scoring.

## Fixed cells, renderer and validity

Each candidate has solid/gap x lawful/inconsistent x clean/persistent +30-degree hue/abrupt +30-degree hue = **12 clips**. Total budget 3,072 clips, 147,456 raw frames. Same 48-frame/12-fps video, 12 context frames, 36 target frames, 240-Hz simulation, 256x256 input, CUDA Cycles 64 spp, no denoising, fixed lighting/floor/rails and render seed 1025. Appearance changes never modify simulated pose or dynamics. Abrupt hue begins at source frame 12 and remains a separate event diagnostic. Physical collision switch time and actual contact/divergence time are separately recorded.

Use the development state/contact/pose/replay/invalid-fixture tests unchanged. Target image area must be >=0.25% in every future frame; this is not an unoccluded-object fraction. All 12 cells must pass to make a candidate eligible. Preserve all attempts, failure logs and missing cells; no replacement or seed substitution. Eligibility is decided before scores, not from score signs.

The existing bounded context rule is retained: canonical clean/abrupt contexts use the raw lawful clean reference, persistent contexts use raw lawful persistent. A raw context discrepancy may be at most 1 uint8 level, normalized mean absolute difference <=1e-6 and changed-pixel fraction <=1e-4. Copies go into separate model-input directories; all raw renders remain and all 36 future PNGs are byte-identical to their own raw renders. All source hashes, normalized inputs, contexts and state traces are checked again across the entire eligible set before inference.

Attempt all 256 candidates. If fewer than **231** (more than 10% ineligible) remain, stop before any final score, report the incomplete attempt and do not top up. Otherwise analyse complete eligible candidates, report every exclusion and the achieved count/precision. Resource exhaustion, timeout or unrecorded artifact corruption stops scheduling; it is not a justification to silently drop an arbitrary running process or change the design. No automatic rerender/retry is enabled. A manual recovery may reuse verified completed artifacts under the identical freeze; failed/partial attempts remain recorded and source changes require an explicit amendment, not a silent resume.

## Frozen model and score

Checkpoint vitg.pt SHA-256 `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`; official source `204698b45b3712590f06245fbfba32d3be539812`. FP32 weights, bf16 autocast/subtraction, FP32 mean absolute predictor-versus-layer-normalized EMA target-token error. Fixed 48/12/36 frame mapping and 256 pixels; batch one. Matmul TF32 false, cuDNN TF32 true. Native scorer arithmetic is inherited unchanged from the audited development scorer.

Use disk-block loading for both encoders (not a full GPU-resident target), because the resident path intermittently failed host allocations. This affects residency, not weights or arithmetic. It has passed exact old-fixture reference and uncached-repeat checks. Cache predictor output only for identical normalized contexts, clear per world; never cache a target embedding from a different input. Save scalar, 18 tubelet scores and their exact original frame pairs, input/state/source hashes, timing and peak memory.

Before any final score, reproduce the inspected old development reference within 1e-6 for scalar and all tubelets, with an exact uncached repeat. On the first score-blind eligible scene, retain hold/shuffle and exact-repeat controls. These controls are diagnostics, not extra independent scenes. A control failure stops the run; no metric substitution follows. No scientific partial summaries, sequential peeking, resizing, dtype/window changes or new fitted score are permitted. Resume reads only complete atomic rows matching the frozen inputs; no sample extension after outcomes.

## Analysis frozen in executable code

**Primary estimands:** four means of M = S(inconsistent)-S(lawful): clean solid, clean gap, persistent solid, persistent gap. Resample complete base configurations together across all branches/worlds/appearances, 50,000 draws, NumPy seed 202610256. For each mean use a 98.75% percentile bootstrap interval (quantiles .00625/.99375), Bonferroni for the four-mean family at alpha .05. Bootstrap coverage remains approximate.

Report positive/negative/unresolved statistical direction per primary mean. Consistent positive *mean ordering* requires all four lower bounds >0. This is a narrow statistical statement, not a utility or physical-understanding gate. A negative interval in one context contradicts uniformly positive ordering in this population; an unresolved interval is not equivalence. Report maximum interval half-extent around each estimate against **0.001** as a reporting-resolution target, never as a meaningful-effect or equivalence band.

**Secondary/descriptive:** medians; positive counts/ties (ties not positive) with 95% Clopper-Pearson intervals; geometry-balanced margins; paired persistent-minus-clean/abrupt-minus-clean balanced-margin changes; separate abrupt-event results. Keep the two physical contexts visible—cancellation is not equivalence. No confirmatory temporal-window, layer, geometry-intervention or correction selection.

**Fixed threshold:** native score >= **0.4858863055706024** flags inconsistency. It was fit on all 24 clean development configurations by maximum balanced accuracy, largest-threshold tie break. Apply unchanged; no world-specific refit, reversal of score direction, oracle adjustment or test training. Report false alarms, misses, balanced accuracy and AUROC per appearance/world and pooled equally across worlds, with 10,000 scene-cluster bootstrap draws (the first draws of the same generator) and descriptive 95% intervals. The prevalence is artificial/balanced; no real-world PPV claim. No scientifically justified error-rate ceiling or equivalence margin has been established, so **utility/equivalence gates remain unassessed**, not automatically passed. This explicitly narrows the unratified earlier statistics draft.

The full eligible set is analysed once by the frozen script. Only after completion may outcomes inform a separately labelled follow-up; they do not change this run.

## Sample-size planning and measured hardware scope

Development four-contrast SDs: 0.004707, 0.003928, 0.003838, 0.002917. The 5,000-draw planning calculation gave a 90th-percentile maximum joint half-width of 0.000774 at N=256; a 25% SD-inflation sensitivity allowance gives 0.000968, below the proposed 0.001 reporting resolution. This allowance is not an empirical bound on new-scene variance. Final achieved precision is reported even if it misses the target; no top-up is allowed.

Rendering and inference used a local RTX 3080. At most two WSL render workers used four CPU threads each, followed by sequential native inference. Full disk-block scoring was estimated at 25–35 hours including rendering and input audits, compared with an earlier roughly 19-hour ideal resident-target estimate. These are engineering estimates, not scientific sample-size or stopping criteria. Failed/partial artifacts remain distinct from completed scientific records.
