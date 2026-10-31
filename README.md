# JEPA surprise: representations and native prediction error

Scientific code and numeric records for the study of appearance accessibility, representation geometry, physical-state readout and the interpretation of native latent prediction error.

## Current results

* Appearance remains accessible despite substantial layerwise changes in displacement and directional organization. The IntPhys2 readout is a post-hoc scene cross-fit, not an untouched new test.
* Action-conditioned models show genuine same-target action sensitivity. Appearance increments and absolute intervention scores answer different questions. In PushT, DINO-WM and JEPA-WM have negative absolute hue-minus-opposite contrasts, but larger hue increments over the respective lawful baselines at the highlighted match. This does not establish action dominance over appearance sensitivity.
* Crossed-support final256 contains 256 independent configurations and 3,072 clips. Inconsistent-minus-lawful native error reverses sign across solid and gap worlds. This limits consistent raw-error physical ordering under the tested protocol; it does not demonstrate absence of physical information.
* Physion++ Mass records retain static-content and persistent/target-only appearance diagnostics on 192 videos / 96 pairs. They are inspected synthetic-contact diagnostics; the full-clip encoder prevents a context-only causal interpretation. Geometry transfer is post-hoc and has no positive family-adjusted incremental lower bound.

## Release layout and verification

[Release index](release/README.md) describes scientific status, inputs, modifications and limitations. [Coverage](release/COVERAGE.md) accounts for all 105 branch result-directory identities. Alex's Oct24–25 analyses and Oct31 figure generators are under `scripts/wmpai2026/alex/` and their dated `results/` records. They remain exploratory where originally designated.

Run the independent offline checks with Python 3.11 or newer:

```sh
python tools/check_release.py
```

No dependencies, model loading, downloads, fitting or new bootstrap are required. Role-specific input and exact dependency wiring are described in [the dependency map](release/DEPENDENCIES.md) and tested without importing experiment modules. Checks verify current and historical byte manifests, final256 identities and all 28 figure-generator numeric acceptance checks, Physion row-to-summary means and hybrid identities, and geometry target lineage. Reconstructing or checking stored statistics is not a new experiment.

`historical/` is the unchanged curated anonymous scientific package, including its original manifest, audit, scripts, environments, compact representations and results. Run its original commands **from that directory**. Its README's historical scope and release status are preserved as evidence, not a claim that it includes the newer work.

The published camera-ready technical record remains unchanged at [`camera-ready-2026`](https://github.com/3-ml/jepa-surprise/tree/camera-ready-2026); `docs/reproducibility*` is preserved byte-for-byte. The expanded scientific package is pinned by the `reproducibility-2026-10-31` tag. It includes source snapshots and derived records, with the independent-execution limits documented in the release index.

## Rights and external inputs

No blanket license is asserted. Existing notices, including Kubric's license and notice, are retained. Third-party datasets, checkpoints, model source, raw media/render trees and large feature tensors are not redistributed. Their original terms apply. See the release index and `historical/environments/external_assets/` for acquisition/provenance and known missing hashes or licensing information.
