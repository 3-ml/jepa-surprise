"""A3 feature extraction for Vasi's fresh scenes: same frozen V-JEPA2-G checkpoint, one feature row per
clip, aligned 1:1 with the scores table by clip_id and input hash.

The encoder runs through gpu4gb_offload.py (one block on the GPU at a time), so a 4 GB GPU is enough;
on a larger GPU the same weights give the same arithmetic. Run `gpu4gb_offload.py shard` once first.
Alignment logic is unit-tested (tests/test_wmpai_alex_a3_extract.py); the 4 GB path was compared with a stored
A40 shard in results/*-wmpai-alex-gpu4gb-offload-parity/.

Layers are an explicit, frozen list chosen on DEVELOPMENT scenes (--layers block_20 block_30 ...).
The script refuses to run without one, so layers can never be picked after seeing confirmatory outcomes.
Layer names: block_NN is the output of transformer block NN (0-based, as in the layerwise extractor),
`final` is the encoder output. Pooling: mean over all encoder tokens of the clip. A per-target-tubelet
pooling would need the token layout and is not implemented here.

Usage:
  python a3_extract_fresh_scenes.py --scores scores.csv --frames-root <dir> \
      --layers block_20 block_30 final --split development --out features_dev [--batch 4]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CKPT_SHA256 = "67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e"


def clip_input_sha256(frame_paths):
    """Hash of the exact input bytes: concatenated frame files in sorted order."""
    h = hashlib.sha256()
    for p in sorted(frame_paths):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def plan(score_rows, frames_root, split):
    """Pick the clips to extract and check each against the manifest's input hash.
    Returns (todo, problems). Any problem means: do not extract, fix the manifest first."""
    todo, problems = [], []
    for r in score_rows:
        if r["split"] != split:
            continue
        d = Path(frames_root) / r["clip_id"]
        files = sorted(d.glob("*.png"))
        if not files:
            problems.append(f"{r['clip_id']}: no frames under {d}")
            continue
        got = clip_input_sha256(files)
        want = r.get("input_sha256") or r.get("target_input_sha256")
        if want is None:
            problems.append(f"{r['clip_id']}: manifest row has no input_sha256")
        elif got != want:
            problems.append(f"{r['clip_id']}: input hash mismatch")
        else:
            todo.append({**r, "frames": files, "input_sha256": got})
    return todo, problems


def check_layers(requested, available):
    if not requested:
        raise ValueError("--layers is required: freeze layers on development scenes first")
    unknown = [l for l in requested if l not in available]
    if unknown:
        raise ValueError(f"unknown layers {unknown}")
    return list(requested)


def write_features(out, todo, feats, layers, dtype, checkpoint_sha256, extra=None):
    """feats: dict clip_id -> array (n_layers, D) or None for a failed clip (kept as NaN, never zero)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = next((v.shape[1] for v in feats.values() if v is not None), 0)
    arr = np.full((len(todo), len(layers), d), np.nan, np.float32)
    for i, r in enumerate(todo):
        if feats.get(r["clip_id"]) is not None:
            arr[i] = feats[r["clip_id"]]
    np.save(out / "features.npy", arr)
    with open(out / "feature_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "clip_id", "scene_id", "input_sha256", "missing"])
        for i, r in enumerate(todo):
            w.writerow([i, r["clip_id"], r["scene_id"], r["input_sha256"], int(np.isnan(arr[i]).any())])
    json.dump({"layers": layers, "dtype": dtype, "pooling": "mean over all encoder tokens, 48-frame window",
               "checkpoint_sha256": checkpoint_sha256, "expected_checkpoint_sha256": CKPT_SHA256,
               "n_clips": len(todo), "n_missing": int(sum(np.isnan(arr[i]).any() for i in range(len(todo)))),
               **(extra or {})},
              open(out / "feature_meta.json", "w"), indent=1)
    return arr


LAYER_NAMES = [f"block_{i:02d}" for i in range(40)] + ["final"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_clip(files):
    """Frames as uint8 (T, H, W, 3) RGB."""
    import cv2
    return np.stack([cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for f in files])


def normalize(clips, resolution=256):
    """Same preprocessing as extract_vjepa2_intphys2_layers.normalize_batch: bilinear resize, ImageNet stats."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.stack(clips)).float()
    b, steps, h, w, c = t.shape
    t = t.reshape(b * steps, h, w, c).permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(resolution, resolution), mode="bilinear", align_corners=False)
    t = t.reshape(b, steps, c, resolution, resolution).permute(0, 2, 1, 3, 4).contiguous() / 255.0
    mean = torch.tensor(MEAN).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1, 1)
    return (t - mean) / std


class PooledCapture:
    """Mean-over-tokens features at chosen layers, as float32 (batch, layers, dim)."""

    def __init__(self, encoder, layers):
        self.encoder, self.layers, self.store, self.handles = encoder, list(layers), {}, []
        for name in self.layers:
            if name != "final":
                self.handles.append(encoder.blocks[int(name.split("_")[1])].register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def capture(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self.store[name] = tensor.detach().float().mean(dim=1).cpu()
        return capture

    def infer(self, x):
        import torch
        self.store.clear()
        on_gpu = x.is_cuda
        with torch.inference_mode(), torch.autocast("cuda" if on_gpu else "cpu", dtype=torch.bfloat16, enabled=on_gpu):
            final = self.encoder(x)
        if "final" in self.layers:
            self.store["final"] = final.detach().float().mean(dim=1).cpu()
        return np.stack([self.store[n].numpy() for n in self.layers], axis=1)

    def close(self):
        for h in self.handles:
            h.remove()


def extract(todo, capture, batch=4, frames_per_clip=48, resolution=256, device="cuda", log=print):
    """clip_id -> (layers, dim) float32, or None when a clip could not be processed (never zero)."""
    import torch
    feats = {}
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        ok, clips = [], []
        for r in chunk:
            if len(r["frames"]) != frames_per_clip:
                feats[r["clip_id"]] = None
                log(f"skip {r['clip_id']}: {len(r['frames'])} frames, expected {frames_per_clip}")
                continue
            ok.append(r)
            clips.append(load_clip(r["frames"]))
        if not ok:
            continue
        try:
            out = capture.infer(normalize(clips, resolution).to(device))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out = np.concatenate([capture.infer(normalize([c], resolution).to(device)) for c in clips])
        for r, vec in zip(ok, out):
            feats[r["clip_id"]] = vec if np.isfinite(vec).all() else None
        log(f"{min(start + batch, len(todo))}/{len(todo)}")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--layers", nargs="+")
    ap.add_argument("--split", choices=("development", "confirmatory"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4, help="clips per forward pass (4 was fastest on a 4 GB GPU)")
    ap.add_argument("--frames", type=int, default=48, help="frames per clip")
    ap.add_argument("--shard-dir", type=Path, help="per-block weights from gpu4gb_offload.py shard")
    a = ap.parse_args()
    layers = check_layers(a.layers, LAYER_NAMES)

    rows = list(csv.DictReader(open(a.scores)))
    todo, problems = plan(rows, a.frames_root, a.split)
    if problems:
        raise SystemExit("refusing to extract:\n" + "\n".join(problems[:20]))

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gpu4gb_offload as off
    shard_dir = a.shard_dir or off.SHARD_DIR
    if not (Path(shard_dir) / "index.json").exists():
        raise SystemExit(f"no sharded weights in {shard_dir}; run: python gpu4gb_offload.py shard")
    encoder, offload, load_note = off.build_offloaded_encoder(num_frames=a.frames, shard_dir=shard_dir)
    capture = PooledCapture(encoder, layers)
    try:
        feats = extract(todo, capture, batch=a.batch, frames_per_clip=a.frames)
    finally:
        capture.close()
        offload.close()
    write_features(a.out, todo, feats, layers, "bfloat16 autocast, fp32 weights", CKPT_SHA256,
                   extra={"weights_source": str(shard_dir), "checkpoint_sha256_verified_in_this_run": False,
                          "batch": a.batch, "frames_per_clip": a.frames,
                          "load_note": load_note})


if __name__ == "__main__":
    main()
