"""Run the frozen V-JEPA2-G encoder on a 4 GB GPU by loading one transformer block at a time.

Step 1 (once): `shard` reads only the encoder tensors from vitg.pt with ordinary file reads and writes
one fp32 file per block (byte-identical weights). torch.load(mmap=True) is not used because mapping a
16.5 GB file crashes PyTorch on this Windows machine (access violation on the first read).
Step 2: the model is built on the meta device; before each block runs its weights are loaded onto the
GPU and afterwards dropped. GPU memory holds one block plus activations. The arithmetic is the original
one (fp32 weights, bfloat16 autocast, same hooks and pooling as scripts/extract_vjepa2_intphys2_layers.py).

Batching and bucketing do not reduce memory (a larger batch needs more activations); they only amortise
the per-block weight load. Use batch 1 unless measured otherwise.

  python gpu4gb_offload.py shard --out <dir>
  python gpu4gb_offload.py parity --sample-id <id> --out <dir>   compare with a stored A40 shard
"""
import argparse
import collections
import csv
import gc
import hashlib
import json
import pickle
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

VJEPA2_ROOT = Path(__import__("os").environ['JEPA_VJEPA2_SOURCE'])
CHECKPOINT = Path(__import__("os").environ['JEPA_VJEPA2G_CHECKPOINT'])
CHECKPOINT_SHA256 = "67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e"
SHARD_DIR = Path(__import__("os").environ['JEPA_VJEPA2G_BLOCKS'])
DATASET_ZIP = Path(__import__("os").environ['JEPA_INTPHYS2_ZIP'])
SAMPLES = ROOT / "historical/manifests/intphys2/intphys2_confirmatory_samples.csv"

STORAGE_DTYPES = {"FloatStorage": torch.float32, "HalfStorage": torch.float16, "BFloat16Storage": torch.bfloat16,
                  "DoubleStorage": torch.float64, "LongStorage": torch.int64, "IntStorage": torch.int32}


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def tensor_index(checkpoint):
    """Map each tensor in the checkpoint's 'encoder' dict to its raw storage, without reading data."""
    z = zipfile.ZipFile(checkpoint)
    prefix = z.namelist()[0].split("/")[0]

    def rebuild(storage, offset, size, stride, *_a, **_k):
        return {"storage": storage, "offset": int(offset), "size": tuple(size), "stride": tuple(stride)}

    class Reader(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild
            if module == "torch" and name.endswith("Storage"):
                return name
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _tag, stype, key, _loc, numel = pid
            return {"stype": stype if isinstance(stype, str) else stype.__name__, "key": key, "numel": int(numel)}

    payload = Reader(z.open(f"{prefix}/data.pkl")).load()
    return z, prefix, payload["encoder"]


def read_tensor(z, prefix, rec):
    st = rec["storage"]
    dtype = STORAGE_DTYPES[st["stype"]]
    raw = bytearray(z.read(f"{prefix}/data/{st['key']}"))
    flat = torch.frombuffer(raw, dtype=dtype)
    return torch.as_strided(flat, rec["size"], rec["stride"], rec["offset"]).clone()


def shard(out_dir=SHARD_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z, prefix, enc = tensor_index(CHECKPOINT)
    groups = collections.defaultdict(dict)
    for key, rec in enc.items():
        name = key.removeprefix("module.backbone.")
        m = re.match(r"blocks\.(\d+)\.(.*)", name)
        if m:
            groups[f"block_{int(m.group(1)):02d}"][m.group(2)] = rec
        else:
            groups["misc"][name] = rec
    manifest = {"source": str(CHECKPOINT), "source_sha256_expected": CHECKPOINT_SHA256, "files": {}}
    for gname in sorted(groups):
        tensors = {n: read_tensor(z, prefix, r) for n, r in groups[gname].items()}
        path = out_dir / f"{gname}.pt"
        torch.save(tensors, path)
        manifest["files"][path.name] = {"sha256": sha256_file(path), "tensors": len(tensors),
                                        "bytes": path.stat().st_size,
                                        "params": int(sum(t.numel() for t in tensors.values()))}
        print(gname, len(tensors), flush=True)
    manifest["total_params"] = sum(f["params"] for f in manifest["files"].values())
    (out_dir / "index.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


class BlockOffload:
    """Forward hooks that load one block's weights onto the GPU for its forward and then drop them."""

    def __init__(self, blocks, shard_dir, device="cuda"):
        self.dir, self.device = Path(shard_dir), device
        self.number = {id(b): i for i, b in enumerate(blocks)}
        # (dotted name, owning submodule, attribute, original meta Parameter) for every weight in the block
        self.slots = {}
        for b in blocks:
            slots = []
            for n, p in b.named_parameters():
                owner_path, _, attr = n.rpartition(".")
                slots.append((n, b.get_submodule(owner_path) if owner_path else b, attr, p))
            self.slots[id(b)] = slots
        self.handles = []
        for b in blocks:
            self.handles.append(b.register_forward_pre_hook(self._load))
            self.handles.append(b.register_forward_hook(self._drop))

    def _load(self, block, _args):
        state = torch.load(self.dir / f"block_{self.number[id(block)]:02d}.pt", map_location=self.device,
                           weights_only=True)
        for name, owner, attr, _orig in self.slots[id(block)]:
            setattr(owner, attr, torch.nn.Parameter(state[name], requires_grad=False))

    def _drop(self, block, _args, _output):
        for _name, owner, attr, orig in self.slots[id(block)]:
            setattr(owner, attr, orig)

    def close(self):
        for h in self.handles:
            h.remove()


def build_offloaded_encoder(num_frames=64, resolution=256, shard_dir=SHARD_DIR, root=VJEPA2_ROOT):
    sys.path.insert(0, str(root))
    from src.models import vision_transformer as vit

    # The constructor calls .item() on a linspace; on the meta device that fails, so the small
    # helper tensor is created on the CPU while every parameter is still created on meta.
    real_linspace = torch.linspace
    torch.linspace = lambda *a, **k: real_linspace(*a, **{**k, "device": "cpu"})
    try:
        with torch.device("meta"):
            encoder = vit.vit_giant_xformers(img_size=resolution, num_frames=num_frames, uniform_power=True,
                                             use_activation_checkpointing=False, use_rope=True)
    finally:
        torch.linspace = real_linspace
    misc = torch.load(Path(shard_dir) / "misc.pt", map_location="cpu", weights_only=True)
    result = encoder.load_state_dict(misc, strict=False, assign=True)
    block_keys = {k for k in result.missing_keys}
    if any(not k.startswith("blocks.") for k in block_keys) or result.unexpected_keys:
        raise RuntimeError(f"unexpected load result: {result}")
    left = [n for n, t in encoder.named_buffers() if t.is_meta]
    if left:
        raise RuntimeError(f"buffers still on meta: {left[:5]}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for name, child in encoder.named_children():
        if name != "blocks":
            child.to("cuda")
    offload = BlockOffload(list(encoder.blocks), shard_dir)
    return encoder, offload, f"misc loaded; {len(block_keys)} block tensors loaded on demand"


def parity(sample_id, out):
    import intphys2_confirmatory_common as common
    import extract_vjepa2_intphys2_layers as ex

    rows = {r["sample_id"]: r for r in csv.DictReader(open(SAMPLES, encoding="utf-8"))}
    row = rows[sample_id]
    assert row["appearance_condition"] == "clean", "parity uses the untransformed clip"
    indices = json.loads(row["source_frame_indices_0based"])
    video = Path(__import__("os").environ['JEPA_ALEX_SCRATCH']) / Path(row["source_relative_path"]).name
    with zipfile.ZipFile(DATASET_ZIP) as zf:
        video.write_bytes(zf.read(row["source_relative_path"]))
    assert hashlib.sha256(video.read_bytes()).hexdigest() == row["source_sha256"], "source video hash mismatch"
    frames = common.decode_selected_frames(video, indices)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    t0 = time.perf_counter()
    encoder, offload, strict = build_offloaded_encoder(num_frames=len(indices))
    load_s = time.perf_counter() - t0
    capture = ex.VJEPASummaryCapture(encoder, (4, 4, 4))
    x = ex.normalize_batch([frames], 256).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(2):
        torch.cuda.synchronize()
        t = time.perf_counter()
        pooled, _coarse = capture.infer(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30

    sys.path.insert(0, str(ROOT / "scripts" / "wmpai2026" / "alex"))
    import io
    raise RuntimeError("Authenticated remote parity adapter excluded; use bundled parity records")
    key = f"features/intphys2-layerwise-confirmatory-v1/vjepa2/{row['partition']}/shards/{sample_id}.npz"
    stored = np.load(io.BytesIO(r2.s3.get_object(Bucket=r2.BUCKET, Key=key)["Body"].read()), allow_pickle=True)
    ref = stored["pooled"].astype(np.float32)
    mine = pooled[0] if pooled.ndim == 3 else pooled
    ref = ref[0] if ref.ndim == 3 else ref
    mine16 = mine.astype(np.float16).astype(np.float32)
    per_layer = []
    for i, layer in enumerate(int(v) for v in stored["native_layers"]):
        a, b = mine16[i], ref[i]
        per_layer.append({"layer": layer, "max_abs_diff": float(np.abs(a - b).max()),
                          "relative_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                          "cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))})
    result = {"sample_id": sample_id, "checkpoint_sha256_expected": CHECKPOINT_SHA256, "frames": len(indices),
              "tokens": 8192, "load": strict, "gpu": torch.cuda.get_device_name(0),
              "peak_allocated_gib": peak, "peak_reserved_gib": reserved, "forward_seconds": times,
              "model_setup_seconds": load_s, "finite": bool(np.isfinite(pooled).all()), "per_layer": per_layer,
              "stored_shard_key": key, "reference": "A40 shard, fp16 pooled means"}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("peak_allocated_gib", "peak_reserved_gib", "forward_seconds", "finite")}))
    for r in per_layer:
        print(r)
    capture.close()
    offload.close()
    gc.collect()


def bench(sample_id, out, batches=(1, 2, 4, 8)):
    """Throughput and peak GPU memory for batches of clips (the same clip repeated; content does not matter)."""
    import intphys2_confirmatory_common as common
    import extract_vjepa2_intphys2_layers as ex

    row = {r["sample_id"]: r for r in csv.DictReader(open(SAMPLES, encoding="utf-8"))}[sample_id]
    indices = json.loads(row["source_frame_indices_0based"])
    video = Path(__import__("os").environ['JEPA_ALEX_SCRATCH']) / Path(row["source_relative_path"]).name
    with zipfile.ZipFile(DATASET_ZIP) as zf:
        video.write_bytes(zf.read(row["source_relative_path"]))
    frames = common.decode_selected_frames(video, indices)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder, offload, _ = build_offloaded_encoder(num_frames=len(indices))
    capture = ex.VJEPASummaryCapture(encoder, (4, 4, 4))
    rows = []
    for b in batches:
        try:
            x = ex.normalize_batch([frames] * b, 256).to("cuda")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            capture.infer(x)
            torch.cuda.synchronize()
            t = time.perf_counter()
            capture.infer(x)
            torch.cuda.synchronize()
            s = time.perf_counter() - t
            rows.append({"batch": b, "seconds_per_batch": s, "seconds_per_clip": s / b,
                         "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": b, "out_of_memory": True})
            torch.cuda.empty_cache()
        print(rows[-1], flush=True)
        del x
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "batch_benchmark.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--sample-id", required=True)
    b.add_argument("--out", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--out", default=str(SHARD_DIR))
    p = sub.add_parser("parity")
    p.add_argument("--sample-id", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "shard":
        m = shard(a.out)
        print(json.dumps({"files": len(m["files"]), "total_params": m["total_params"]}))
    elif a.cmd == "bench":
        bench(a.sample_id, a.out)
    else:
        parity(a.sample_id, a.out)
