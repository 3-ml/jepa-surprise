# Scientific input and dependency contracts

The global-root substitution was removed. Each executable absolute path literal was restored from its pinned original, then mapped to the explicit role below. The [machine-readable contracts](path-contracts.json) identify each module, binding, original-identity hash, expression, suffix and file/directory kind. The original private path strings are not published. `python -B tools/check_paths.py` checks these expressions as data without importing experiment modules, models or numerical frameworks.

## External roles

Set only the roles needed by the selected module. A missing required environment variable raises `KeyError`; there is no private-cache fallback. File roles are files, not directories. The checked synthetic environment deliberately gives every role a different location. Original copies of the same G checkpoint and the same pinned source revision share one scientific role; unrelated inputs never alias.

| Environment | Kind at root | Role and preserved descendants |
|---|---|---|
| `JEPA_ALEX_SCRATCH` | directory | Alex extraction scratch directory |
| `JEPA_BENCHMARK_ROOT` | directory | Benchmark root; grid and stage-e remain separate subdirectories; descendants: `vjepa2g-grid`, `vjepa2g-grid-stage-e` |
| `JEPA_FINAL256_SCRATCH` | directory | Final256 renderer scratch |
| `JEPA_GEOMETRY_SCRATCH` | directory | Geometry batch scratch |
| `JEPA_INTPHYS2_ENDPOINT` | directory | Stored inspected endpoint score/metadata tree; descendants: `extracted/intphys2-appearance-completion/main/clean/videos/%s.json` |
| `JEPA_INTPHYS2_ZIP` | file | Official IntPhys2 dataset ZIP |
| `JEPA_LEWM_PUSHT_HDF5` | file | LeWM PushT expert training HDF5 |
| `JEPA_MULTIDOSE_CACHE` | directory | Packed multi-dose representation cache |
| `JEPA_MULTIFAMILY16_FRAMES` | directory | Archived16 rendered frame root |
| `JEPA_MULTIFAMILY16_RECORD` | directory | Archived16 numeric record root; descendants: `run/analysis/per_scene_responses.csv` |
| `JEPA_PILOT8_FRAMES` | directory | Pilot8 rendered frame root; descendants: `seed-230200` |
| `JEPA_VASI_WORKSPACE` | directory | Study workspace; original scientific subdirectory relationships retained; descendants: `contact-development-1`, `contact-development-1/lawful/clean/frames`, `contact-development-scores-1/scores.csv`, `final256-v1`, `pusht-spark-preparation/action-normalization`, `reviewer-physion-20261026-v1`, `reviewer-physion-canonical-20261026-v2`, `reviewer-physion-desktop-20261026-v2` |
| `JEPA_VJEPA2G_BLOCKS` | directory | Offloaded encoder weight shard directory |
| `JEPA_VJEPA2G_CHECKPOINT` | file | V-JEPA2-G checkpoint file |
| `JEPA_VJEPA2_SOURCE` | directory | V-JEPA2 model source at revision 204698b |

Project-root literals now resolve to the bundled repository using the module's own location; no external project checkout is selected implicitly. This applies to the regression and audit copies as well as the main modules. Suffixes retain relationships: the contact reference frames live under `JEPA_VASI_WORKSPACE/contact-development-1/lawful/clean/frames`, while the reference CSV is `JEPA_VASI_WORKSPACE/contact-development-scores-1/scores.csv`. The two grid benchmarks remain `JEPA_BENCHMARK_ROOT/vjepa2g-grid/benchmark_results.json` and `JEPA_BENCHMARK_ROOT/vjepa2g-grid-stage-e/benchmark_results.json`.

`JEPA_VASI_WORKSPACE/reviewer-physion-desktop-20261026-v2/model/` is the original qualified Physion model/fixture layout, not model source and not the scoring-core directory. `study_common.OLD` and `canonical_score.OLD` now resolve to that same descendant. `JEPA_VJEPA2G_CHECKPOINT` is the separate G checkpoint file, SHA-256 `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`; the Physion model safetensors hash remains `25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b`. `JEPA_VJEPA2_SOURCE` denotes model source at `204698b45b3712590f06245fbfba32d3be539812`, not a checkpoint. These external assets are not redistributed.

The bundled figure score input is an exception to required external roles: figures 1–3 default to `release/final256/original/scores/all-scores.json`, optionally overridden by `JEPA_FINAL256_SCORES`. Dose figures require `JEPA_MULTIDOSE_CACHE`; they do not fall back to a private cache. None of these figures was rendered during release repair.

## Exact scientific dependency wiring

* `release/{physion-static,physion-appearance,geometry-transfer}/frozen-source.zip/canonical_score.py` → `release/physion-core/source/run_vjepa2_section_surprise.py`; SHA-256 `72dddefb7e6b4c0dd87090249b9fd09ee4064b79d72a15fad3a346a60a3312ac`. Exact completion-runtime archived core; selected by frozen source hash, not filename.
* `scripts/wmpai2026/alex/gpu4gb_offload.py:SAMPLES` → `historical/manifests/intphys2/intphys2_confirmatory_samples.csv`; SHA-256 `1ddc3677781e605f6d298992584768e414801dc1479b03184150dd2af11fd32a`. Byte-identical to original main data/manifests/intphys2_confirmatory_samples.csv.
* `scripts/wmpai2026/vasi/{score_contact_development,score_development24,geometry_development24}.py` → `scripts/wmpai2026/vasi/disk_offload_vjepa2.py`; SHA-256 `365d1ea8862a72c48948b928fc3aa2fffe3bbff7d97d747b14467bfbafa3e143`. Pinned top-level implementation is an exact frozen final256 match; no cross-variant substitution.
* `scripts/wmpai2026/vasi/{score_contact_development,score_development24,geometry_development24}.py` → `scripts/wmpai2026/vasi/stream_vjepa2_checkpoint.py`; SHA-256 `b5fca7815ba38980f99251b72a50f21dc241b2a5e0f17b73edcaa05feb8c6b6a`. Pinned top-level implementation is an exact frozen final256 match; no cross-variant substitution.

The three canonical scorers use a separate bundled `CORE_SOURCE`, checked against its frozen hash before loading. Their mask/forward/normalization/reduction AST is statically compared with that exact core by the path tests. This does not import or execute PyTorch, validate a model's outputs, or repair historical Transformers provenance.

## Preserved variants

Filenames are not equivalence claims. These implementations remain separate, with their exact original/public hashes in `source-map.json` and `dependency-map.json`:

| Path | Original SHA-256 | Public SHA-256 |
|---|---|---|
| `historical/scripts/run_vjepa2_kubric_construct_validity.py` | `61282164a65ceb9d15353cd7225d11ca189cf10d95213547812c6035cb9d275b` | `61282164a65ceb9d15353cd7225d11ca189cf10d95213547812c6035cb9d275b` |
| `historical/scripts/run_vjepa2_section_surprise.py` | `24598ea36b059c2645b91e86a3b613da2bad435e8e34be5f92286432fde65e20` | `24598ea36b059c2645b91e86a3b613da2bad435e8e34be5f92286432fde65e20` |
| `release/final256/original/frozen-source/scripts/run_intphys2_dense.py` | `bd812402fefc7bc3c7b309b039448a5f444fb56e29d40ca7cd87c7b02afc05c6` | `bd812402fefc7bc3c7b309b039448a5f444fb56e29d40ca7cd87c7b02afc05c6` |
| `release/final256/original/frozen-source/scripts/run_vjepa2_kubric_construct_validity.py` | `27e17779b97b8abbbaf9bf7ce630a3778abf1fc49a0a6c030aeb869f18a15d1d` | `27e17779b97b8abbbaf9bf7ce630a3778abf1fc49a0a6c030aeb869f18a15d1d` |
| `scripts/run_intphys2_dense.py` | `36f03b5b3ea5ecca05b0e81182d2c7ad96c47e23570d76b3de2ac2e22cd5333c` | `36f03b5b3ea5ecca05b0e81182d2c7ad96c47e23570d76b3de2ac2e22cd5333c` |
| `scripts/run_vjepa2_kubric_construct_validity.py` | `d4a397fea3ff7c87b941b6948bff23534d5d426d462dde3ac746b50676bc649b` | `d4a397fea3ff7c87b941b6948bff23534d5d426d462dde3ac746b50676bc649b` |
| `scripts/run_vjepa2_section_surprise.py` | `fcd97657328c2c7caabac411926b487f03961e20340a39b37544fccf10d6d922` | `fcd97657328c2c7caabac411926b487f03961e20340a39b37544fccf10d6d922` |
| `release/physion-core/source/run_vjepa2_section_surprise.py` | `72dddefb7e6b4c0dd87090249b9fd09ee4064b79d72a15fad3a346a60a3312ac` | `72dddefb7e6b4c0dd87090249b9fd09ee4064b79d72a15fad3a346a60a3312ac` |

No frozen `run_intphys2_dense.py` or Kubric scorer is redirected to a differently hashed historical/current variant. Frozen final256 imports retain their local source tree; top-level code retains its own implementation. Historical curated code stays byte-identical.

## Archival execution boundaries

Path contracts repair structural input identity. A separate [local execution adapter](../docs/LOCAL_EXECUTION.md) now supports Physion, geometry extraction and scoring sealed final256 renders without the original controller. Its complete input bundle, runtime and device requirements are explicit; this does not make all historical launchers portable. Required historical execution-control functions deliberately raise, and original source-freeze checks still refer to original hashes and omitted operational members. Some older scripts expect additional relative configurations and external manifests in their original layout or via their existing CLI arguments. No complete executable closure is asserted for every dated attempt. Named alternatives in `source-coverage.json` remain alternatives without byte-equivalence claims.

Recorded commands and metadata use symbolic `${ROLE}` references where a known role exists. Other historical record-only local references are distinct `external-record/<identity-hash>/<basename>` references in `path-contracts.json`, not runnable paths or generic aliases. Their original bytes remain identified by source hashes. Do not execute archival command captures as a setup recipe. Independent release validation is `python -B tools/check_release.py`. The reviewed local adapter supplies controller-free dispatch for the three named scopes; actual inference still requires the documented external assets, original runtime/device gates and successful scientific parity checks.
