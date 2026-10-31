import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ex", ROOT / "scripts/wmpai2026/alex/a3_extract_fresh_scenes.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


class AddOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


class TinyEncoder(torch.nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList([AddOne() for _ in range(depth)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def test_pooled_capture_means_tokens_at_requested_layers():
    enc = TinyEncoder()
    cap = ex.PooledCapture(enc, ["block_00", "block_02", "final"])
    out = cap.infer(torch.zeros(2, 5, 4))
    assert out.shape == (2, 3, 4)
    assert np.allclose(out[:, 0], 1) and np.allclose(out[:, 1], 3) and np.allclose(out[:, 2], 3)
    cap.close()
    assert not enc.blocks[0]._forward_hooks


def test_layer_names_cover_blocks_and_final():
    assert ex.LAYER_NAMES[0] == "block_00" and ex.LAYER_NAMES[39] == "block_39" and ex.LAYER_NAMES[-1] == "final"
    assert ex.check_layers(["block_20", "final"], ex.LAYER_NAMES) == ["block_20", "final"]
    with pytest.raises(ValueError):
        ex.check_layers(["block_40"], ex.LAYER_NAMES)


def test_normalize_matches_the_archived_extractor():
    sys.path.insert(0, str(ROOT / "scripts"))
    import extract_vjepa2_intphys2_layers as archived
    rng = np.random.default_rng(0)
    clips = [rng.integers(0, 256, (6, 40, 56, 3), dtype=np.uint8) for _ in range(2)]
    assert torch.equal(ex.normalize(clips, 32), archived.normalize_batch(clips, 32))


def test_extract_marks_wrong_frame_count_and_nonfinite_as_missing_not_zero():
    class Cap:
        def infer(self, x):
            out = np.ones((x.shape[0], 2, 4), np.float32)
            out[-1, 0, 0] = np.nan
            return out

    good = {"clip_id": "a", "frames": ["f"] * 6}
    short = {"clip_id": "b", "frames": ["f"] * 3}
    bad = {"clip_id": "c", "frames": ["f"] * 6}
    orig_load = ex.load_clip
    ex.load_clip = lambda files: np.zeros((len(files), 8, 8, 3), np.uint8)
    try:
        feats = ex.extract([good, short, bad], Cap(), batch=4, frames_per_clip=6, resolution=16,
                           device="cpu", log=lambda *_: None)
    finally:
        ex.load_clip = orig_load
    assert feats["b"] is None and feats["c"] is None
    assert feats["a"] is not None and np.isfinite(feats["a"]).all()
