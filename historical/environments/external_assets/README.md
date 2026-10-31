# External assets

Third-party datasets and checkpoints are not redistributed. Obtain them from the public sources below, then verify the recorded hashes before inference.

## IntPhys2 representation and geometry studies

| Asset | Public source | Pinned identity |
|---|---|---|
| IntPhys2 | `https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip` | archive SHA-256 `aae46b00767cc96f6eeb8bf967b30f3999b7543943d2cb56c71c9850e1558f9f`; repository `facebookresearch/IntPhys2` at `7f3f589b81168049c6330d077976ac4e76ef7d46` |
| V-JEPA2-G | `https://dl.fbaipublicfiles.com/vjepa2/vitg.pt` | SHA-256 `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`; source `facebookresearch/vjepa2` at `204698b45b3712590f06245fbfba32d3be539812` |
| VideoMAE2-G | `https://huggingface.co/OpenGVLab/VideoMAEv2-giant` | revision `d27568eb41ccb2d41bb191fc2e3fe5aad74942d4`; `model.safetensors` SHA-256 `a231169cd13307aeb29bc56259a494ef273bfd9b7232f881ac4e8d7e39713add` |

The exact split, sample inventory, transformations, and source hashes are under `configs/` and `manifests/intphys2/`. The current controlled geometry can be rerun directly from bundled representations. The multi-dose raw path downloads IntPhys2 and V-JEPA2-G, then runs the extraction, packing, analysis, and finalization commands in `results/intphys2/multidose_geometry/README.md`.

## Native-error controls

| Asset | Public source | Pinned identity |
|---|---|---|
| V-JEPA2-L | `https://huggingface.co/facebook/vjepa2-vitl-fpc64-256` | snapshot `b3c1679b7c34d3255ef3547f27c7b226aefab26f` |
| LeWorldModel TwoRoom | `https://huggingface.co/quentinll/lewm-tworooms` | revision `77adaae0bc31deab21c93740d1f8bb947cd0bdec`; weights SHA-256 `566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd` |
| LeWorldModel PushT | `https://huggingface.co/quentinll/lewm-pusht` | revision `22b330c28c27ead4bfd1888615af1340e3fe9052`; weights SHA-256 `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` |
| DINO-WM | `https://github.com/gaoyuezhou/dino_wm` and its linked public OSF checkpoint/data collection | source `0a9492fa12044b852ae9e001cc74604b79c8bb0c`; PushT checkpoint SHA-256 `e909f0cec958fc0b49a79f2e85730ae6b5f83dee61a224705f91883eb3bb74c5` |
| JEPA-WM | `https://github.com/facebookresearch/jepa-wms` and `https://huggingface.co/facebook/jepa-wms` | source `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`; model revision `9b9c41ef249466630dbf1a20e78391865d07b3b9`; checkpoint SHA-256 `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb` |
| DINOv2 | `https://github.com/facebookresearch/dinov2` | source `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; ViT-S/14 weight SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| stable-worldmodel | `https://github.com/galilai-group/stable-worldmodel` | source `0ef3856875e70a1283e637fcd2ab936eae6c4e6f` |
| Physion++ | `https://dingmyu.github.io/physion_v2/` | test archive `https://physion-v2.s3.amazonaws.com/test_data.zip`; selected source hashes are retained in the diagnostic record |

LeWorldModel code downloads the pinned model snapshots itself; task datasets remain explicit CLI inputs. DINO-WM and JEPA-WM commands, expected inputs, and hashes are retained in their result provenance files.

## Kubric rerendering

Use `google-research/kubric` at `61f2422c84bab75006df33c6989e0b483db3ccfe` with the container definition under `environments/kubric/`. `scripts/kubric_movia_base_worker.py` regenerates MOVi-A base scenes from the eight frozen seeds; the manifest preparation, intervention rendering, context canonicalization, validation, V-JEPA2-G scoring, and scene-bootstrap scripts then regenerate the analysis.

The original 9,369-file render tree is omitted because it is generated output. Its file count, byte count, checksum-manifest hash, and successful independent readback are retained. The KuBasic manifest itself was public but its content hash was not captured before execution, so exact byte-for-byte rerender identity is not guaranteed across future mutable asset releases; the frozen scene parameters, scientific construction, result tables, and validation tolerances remain reproducible.
