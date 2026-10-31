# WMPAI Alex: V-JEPA2-G encoder on a 4 GB GPU by block offload (feasibility, parity and batching)

**Status: OBSERVED, engineering check on one clip. Not a scientific result.** It shows that the frozen V-JEPA2-G encoder runs on an RTX 3050 Laptop GPU (4 GiB) and agrees with a stored A40 feature shard within bfloat16 hardware noise. It does not establish parity for other clips, for the predictor, or for the native score.

**Environment.** Windows 11, Python 3.10.11, PyTorch 2.5.1+cu121, NVIDIA driver 566.07, one RTX 3050 Laptop GPU (4 GiB), 15.4 GB RAM (about 1.9 GB free during the runs). Code revision `23da932` plus uncommitted `scripts/wmpai2026/alex/gpu4gb_offload.py`.

**Checkpoint.** `${JEPA_VJEPA2G_CHECKPOINT}`, 16,464,208,210 bytes, SHA-256 `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e` (matches the locked model config; verified earlier in this workstream, not re-hashed in these runs). Upstream code: `${JEPA_VJEPA2_SOURCE}` (revision `204698b`).

## Method

`torch.load(mmap=True)` crashes on this machine (Windows access violation on the first read of the 16.5 GB file; ordinary file reads at all offsets work). So the encoder tensors are read with ordinary reads and saved as one fp32 file per transformer block (`shard`), 41 files, 1,012,173,952 parameters, 3.9 GB. Hashes are in `shard_index.json`. The model is built on the meta device; before each block runs its weights are loaded onto the GPU and afterwards dropped. Arithmetic is unchanged: fp32 weights, bfloat16 autocast, TF32 off, the archived extractor's hooks and preprocessing.

```
python scripts/wmpai2026/alex/gpu4gb_offload.py shard
python scripts/wmpai2026/alex/gpu4gb_offload.py parity --sample-id Debug__0412d75b071ce599c9d8c5a1569bce4cd59dae473646ccb075fa8da9b39d47e7__clean --out results/24-18-53-wmpai-alex-gpu4gb-offload-parity
python scripts/wmpai2026/alex/gpu4gb_offload.py bench  --sample-id Debug__0412d75b071ce599c9d8c5a1569bce4cd59dae473646ccb075fa8da9b39d47e7__clean --out results/24-18-53-wmpai-alex-gpu4gb-offload-parity
```

## Parity against the stored A40 shard

Clip: IntPhys2 `Debug` video `0412d75b...47e7`, clean condition, 64 frames, 8,192 tokens, source video hash checked against the manifest. Reference: `features/intphys2-layerwise-confirmatory-v1/vjepa2/debug/shards/<sample>.npz` in R2 (fp16 pooled means at layers 0, 10, 20, 30, 40). Mine is rounded to fp16 for the comparison.

| Layer | Max abs difference | Relative L2 | Cosine |
|---|---|---|---|
| 0 | 0 | 0 | 1.000000 |
| 10 | 0.0078 | 6.0e-5 | 1.000000 |
| 20 | 0.0156 | 1.6e-4 | 1.000000 |
| 30 | 0.0391 | 5.2e-4 | 1.000000 |
| 40 | 0.0469 | 4.0e-3 | 0.999992 |

Layer 0 (patch embedding) is exact. Differences grow with depth, as expected for bfloat16 arithmetic on different GPUs; none was checked against a predeclared tolerance because none exists yet.

## Memory and speed

Peak allocated GPU memory for one 64-frame clip: 0.45 GiB (0.84 GiB reserved). One forward pass took 7.4 s after a 9.2 s first pass; model setup took 4.6 s.

| Batch (same clip repeated) | Seconds per clip | Peak allocated GiB |
|---|---|---|
| 1 | 7.24 | 0.45 |
| 2 | 5.11 | 0.79 |
| 4 | 4.17 | 1.48 |
| 8 | 12.00 | 2.84 |

Batching helps because the per-block weight load is shared by the clips in a batch. Batch 8 was slower, probably because the 4 GiB card begins spilling into shared system memory; that cause was not checked.

## Limitations

One clip, one run each. Timings come from a laptop with other applications open. Only the encoder was run; the predictor and the native prediction score were not. The throughput figure is for 64-frame clips; 48-frame clips are shorter. A different GPU would need its own parity check.
