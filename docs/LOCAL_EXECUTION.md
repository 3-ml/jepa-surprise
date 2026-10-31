# Scoped local execution

`tools/run_local.py` has three explicit, full-population entrypoints:

| CLI study | Frozen implementation | Scope |
|---|---|---|
| `physion` | `release/physion-static/frozen-source.zip/run_physion.py` | B960 then A20708, with 337 qualified historical controls |
| `geometry` | `release/geometry-transfer/frozen-source.zip/extract_geometry.py` | Post-hoc extraction for the sealed 3,072 final inputs; no fitting/bootstrap |
| `final256` | `release/final256/original/frozen-source/scripts/wmpai2026/vasi/final256_input_gate.py` then `final256_score.py` | Recheck and score the published 256-scene sealed renders; no rendering |

```sh
python -B tools/run_local.py physion --inputs /data/bundle --output /data/new-run --check-inputs
python -B tools/run_local.py physion --inputs /data/bundle --output /data/new-run --timeout 3600 --execute
```

Without `--execute`, verification is read-only and standard-library-only. A pass
means **manifest-complete sealed inputs**, not GPU/runtime qualification. Missing
scientific assets are errors, including for geometry. Execution additionally
requires the frozen package versions, actual CUDA availability and the original
scientific parity gates. The Physion scorer retains its RTX 3080 and
PyTorch 2.8.0+cu128 requirements; other devices are not silently qualified.

## Bundle schema

Supply one directory with regular files/directories only, no symlinks anywhere.
Its root `local-inputs.json` must contain exactly these four fields:

```json
{
  "files": {"freeze.json": "64-lowercase-hex-sha256"},
  "study_files": ["freeze.json"],
  "workspace": {
    "reviewer-physion-desktop-20261026-v2": "qualified-model",
    "reviewer-physion-canonical-20261026-v2": "controls"
  },
  "roles": {}
}
```

This is a schema example, not a ready bundle. `files` must hash every bundle file
except the root inventory itself. A nested file named `local-inputs.json` is an
ordinary sealed file. Duplicate JSON keys, unsafe/noncanonical relative paths,
unknown fields, duplicate study files and nested/aliased asset roots are rejected.
The inventory detects local tampering; independently recorded identities below
are the scientific authority. Rehashing altered assets into this inventory does
not qualify them.

`workspace` keys must be exactly the selected study's keys below. Values are
arbitrary, disjoint bundle-relative directories. The launcher creates links with
the required suffixes only inside the new output's `workspace/`. Original
machine paths and parent directory trees are unnecessary.

### Physion

Workspace keys:

* `reviewer-physion-desktop-20261026-v2`: qualified L-model/runtime assets.
* `reviewer-physion-canonical-20261026-v2`: canonical controls.

`roles` must be empty. `study_files` must contain exactly `freeze.json`,
`input-seal.json`, every file named in the published seal's `files`, and every
tensor named in its A/B manifests. These retain bundle-relative names when
copied to the output. The supplied freeze and seal must match the corresponding
published `release/physion-static/` records. Seal file hashes independently bind
the population, input audit, both manifests and prior-input recheck; tensor
file hashes are checked before launch. Original decoding/tensor identity checks,
old-control tolerances, exact repeats and A/B agreement remain in the frozen code.

From the published freeze's external records, preflight requires the exact
control manifest and all 337 control assets, `model/{config.json,
video_preprocessor_config.json,model.safetensors}`, `canonical_codec.py`,
`canonical_score.py`, `model-source-manifest.json`, `prior-input-seal.json`,
`source/run_vjepa2_section_surprise.py`, and `qualification/complete.json` under
the qualified roots. At execution the installed Transformers V-JEPA2 modeling,
configuration and modeling-utils files, plus PyTorch `torch/nn/functional.py`,
`torch/nn/modules/module.py` and `torch/nn/modules/normalization.py`, must match
recorded source hashes resolved through installed distribution metadata.

### Geometry and final256

Both require workspace keys `contact-development-1`,
`contact-development-scores-1`, and `final256-v1`. Geometry additionally requires
`development24-v2`. The old contact CSV and its 48 lawful/clean reference frames
must match the published development record, rather than user-authored scores.

`roles` must contain exactly:

* `JEPA_VJEPA2_SOURCE`: bundle-relative directory at revision
  `204698b45b3712590f06245fbfba32d3be539812`, including Git revision metadata.
  Its recorded `src/` Python files must match hashes in the geometry freeze;
  additional/missing Python source is rejected. Root-level `src` module overrides
  (source, sourceless bytecode or extension binaries) are rejected too; import
  resolution must select only the recorded namespace directory. After verification,
  the fresh child explicitly binds `src.__path__` and its namespace `ModuleSpec`
  exclusively to that directory, before scientific imports. Already-loaded `src`
  modules cause failure. Native loaders can still insert their verified source
  root into `sys.path`; CWD/site-package modules cannot replace the bound namespace.
* `JEPA_VJEPA2G_CHECKPOINT`: bundle-relative checkpoint file, hash
  `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`.

The selected native loader streams blocks directly from this checkpoint ZIP.
It does not consume the separate `JEPA_VJEPA2G_BLOCKS` role used by other archived
implementations; that role is not accepted here.

The final render root must contain the published `freeze.json`, `configs.json`,
`threshold.json`, `inspected-registry.json` and `render-seal.json`, plus all 256
scene directories. Each world's pinned manifest must have the complete six
conditions, its matching `config.json`, state and canonicalization audits,
48 canonical and 48 raw frame paths per clip, and matching state traces.
Cold preflight checks their hashes, configuration/condition identities,
visibility and actual sorted scorer-directory layout. It does not pretend that
file hashes alone perform normalization or model parity.

Geometry's `study_files` are exactly `freeze.json` and
`geometry-coefficients.json`, matching `release/geometry-transfer/`. Its roots
also need the recorded final scores/status, development scores, frozen
feature matrix/index, first development clip's frames and exact reference
feature file. Their published external hashes are checked. The original
contact-reference and old-feature/repeat parity gates still run before final
feature extraction. The adapted final freeze is checked against the published
final freeze, not its obsolete pre-adaptation external hash.

Final256's `study_files` must be empty. Its metadata is copied from the verified
render root; scene trees are linked read-only by usage and new scores go only
under the fresh output. The adapter creates a new `run-lock.json` recording the
published freeze's unchanged original source-hash map and local execution mode.
It replaces only `final256_common.check_sources` with independent published-byte
verification plus lock-map equality. The frozen input gate is recomputed, and
its complete normalized manifest must equal the published gate manifest before
the frozen scorer runs. Source-frame/state hashes, freshness, paired context,
visibility, old-reference parity, finite-score and exact-repeat checks are not
removed. Old score files are never staged as resumable outputs.

## Execution boundary

Every published manifest entry is verified against `release-manifest.json`.
Original scientific source and freeze files are never rewritten. A dedicated
local scientific-freeze checker validates the selected assets above and recorded
package versions, and is also installed as `study_common.check_freeze`.
Original controllers, approvals, scheduling/recovery helpers, platform lifecycle
tests, distribution RECORD files and unrelated studies' external assets are
outside this execution closure, not silently treated as scientifically verified.

Only the known `study_common` runtime/deadline helpers and `desktop_runtime`
module are mapped in the child. `study_common.ROOT` and Physion
`canonical_score.ROOT` point to output; `BASE`, `OLD` and `INPUTS` resolve through
the staged workspace. Geometry retains its published top-level native dependency
variants; final256 retains its own frozen native source tree. No generic module
or numerical-function patch facility is exposed by the CLI.

Output must be fresh, outside the release, and neither an ancestor nor descendant
of the bundle. Input destinations cannot collide with assets or outputs. Optional
`--lock /data/study.lock` uses exclusive creation outside the release/bundle and
is removed on normal exit or child failure. A stale lock after supervisor death
needs manual inspection. The child independently validates before creating its
output. There is no public hidden-child bypass.

The timeout must be finite and in `(0, 28800]` seconds. A POSIX host with
`waitid`/`WNOWAIT` and process groups is required. The parent starts Python with `-I` to ignore CWD/PYTHONPATH import entries and
owns a fresh child session and kills its process group before reaping, including on timeout/error.
The selected entrypoints do not dispatch renderer workers. The child also has
monotonic deadline guards for Physion/geometry. CUDA free/total memory is
measured, without inventing a universal model-fit threshold or owner admission
policy. This is known-process isolation, not a security sandbox or read-only
filesystem enforcement. Keep the sealed bundle unchanged during a job.

Hugging Face offline flags are set and canonical loading remains local-only.
No acquisition, network/cache fallback, refit, bootstrap, clip-limit gate bypass
or claim of portability for every historical attempt is provided. No real-model
execution or clean inference environment qualification was performed here.

## CPU verification

```sh
python -B -m unittest discover -s tests -p 'test_local*.py' -v
uv run --offline --no-project --with numpy --with pillow python -B -m unittest discover -s tests -p 'test_local*.py' -v
```

The optional NumPy/Pillow test runs the actual frozen final256 input gate and
scorer reference-parity path on synthetic images with a fake CUDA/model backend;
it makes no inference claim. Other tests exercise actual Physion/geometry root
and runtime imports, recorded hash/version failures, malformed/missing inputs,
path/output isolation, cold CLI behavior and subprocess errors/timeouts. Tests
never download packages; the second command requires an existing local uv cache.
