"""Strict-load the pinned V-JEPA2-G checkpoint without Windows torch.load(mmap=True).

The official archive is trusted only after the caller verifies its SHA-256. Read
one tensor storage at a time through ZIP ordinary reads, then assign to meta
modules. Never mmap the 16.5 GB checkpoint or materialize its whole state dict.
"""
from __future__ import annotations

import gc
import pickle
import zipfile
from pathlib import Path

import torch

DTYPES = {"FloatStorage": torch.float32, "HalfStorage": torch.float16,
          "BFloat16Storage": torch.bfloat16, "DoubleStorage": torch.float64,
          "LongStorage": torch.int64, "IntStorage": torch.int32}


class Index(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return lambda storage, offset, size, stride, *_a, **_k: {
                "storage": storage, "offset": int(offset), "size": tuple(size), "stride": tuple(stride)}
        if module == "torch" and name.endswith("Storage"):
            return name
        return super().find_class(module, name)

    def persistent_load(self, pid):
        tag, storage_type, key, _location, count = pid
        if tag != "storage":
            raise ValueError(f"unexpected pickle storage tag: {tag}")
        return {"stype": storage_type if isinstance(storage_type, str) else storage_type.__name__,
                "key": key, "count": int(count)}


def read_tensor(archive, prefix, record):
    store = record["storage"]
    dtype = DTYPES[store["stype"]]
    # Ordinary ZIP read; bytearray owns the memory while frombuffer and clone run.
    raw = bytearray(archive.read(f"{prefix}/data/{store['key']}"))
    if len(raw) != store["count"] * dtype.itemsize:
        raise ValueError("storage byte count mismatch")
    flat = torch.frombuffer(raw, dtype=dtype)
    return torch.as_strided(flat, record["size"], record["stride"], record["offset"]).clone()


def meta_builder(builder):
    # RoPE constructors use linspace(...).item(); that small auxiliary scalar
    # cannot be computed on meta, while parameters must remain on meta.
    original = torch.linspace
    torch.linspace = lambda *args, **kw: original(*args, **{**kw, "device": "cpu"})
    try:
        with torch.device("meta"):
            return builder()
    finally:
        torch.linspace = original


def load_models_cpu(checkpoint: Path, root: Path, spec: dict):
    import sys
    sys.path.insert(0, str(root))
    from src.models import predictor as predictor_module
    from src.models import vision_transformer as vit

    common = {"img_size": spec["resolution"], "num_frames": spec["frames_per_clip"],
              "uniform_power": True, "use_activation_checkpointing": False,
              "use_rope": spec["use_rope"]}
    with zipfile.ZipFile(checkpoint) as archive:
        prefix = archive.namelist()[0].split("/")[0]
        with archive.open(f"{prefix}/data.pkl") as stream:
            payload = Index(stream).load()
        if not all(k in payload for k in ("encoder", "target_encoder", "predictor")):
            raise ValueError("checkpoint components missing")
        builders = (
            lambda: getattr(vit, spec["architecture"])(**common),
            lambda: getattr(vit, spec["architecture"])(**common),
            lambda: predictor_module.vit_predictor(
                img_size=spec["resolution"], num_frames=spec["frames_per_clip"],
                embed_dim=spec["embed_dim"], predictor_embed_dim=384,
                depth=12, num_heads=12, uniform_power=True,
                use_activation_checkpointing=False, use_mask_tokens=True,
                use_rope=spec["use_rope"], zero_init_mask_tokens=True,
                num_mask_tokens=spec["num_mask_tokens"]),
        )
        models, details = [], {}
        for component, builder in zip(("encoder", "target_encoder", "predictor"), builders):
            model = meta_builder(builder)
            state = {k.removeprefix("module.backbone."): v for k, v in payload[component].items()}
            keys = set(model.state_dict())
            if keys != set(state):
                raise ValueError(f"{component} strict key mismatch: missing {sorted(keys-set(state))[:6]}; "
                                 f"extra {sorted(set(state)-keys)[:6]}")
            model_state = model.state_dict()
            for name, record in state.items():
                dtype = DTYPES[record["storage"]["stype"]]
                expected = model_state[name]
                if tuple(expected.shape) != tuple(record["size"]) or expected.dtype != dtype:
                    raise ValueError(f"{component} tensor shape or dtype mismatch: {name}")
                tensor = read_tensor(archive, prefix, record)
                result = model.load_state_dict({name: tensor}, strict=False, assign=True)
                if result.unexpected_keys:
                    raise ValueError(f"{component} unexpected tensor: {name}")
                del tensor
            if any(t.is_meta for t in model.state_dict().values()):
                raise ValueError(f"{component} still contains meta tensors")
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            models.append(model)
            details[component] = {"strict_loaded_tensors": len(keys)}
            del state, model_state
            gc.collect()
            print("strict loaded", component, len(keys), flush=True)
    return (*models, details)


class CPUBlockOffload:
    """Keep frozen transformer blocks on CPU; stage exactly one block per forward.

    Hooks replace each CPU parameter with an equal-dtype/value GPU parameter
    before the block and restore the original CPU parameter afterwards. The
    non-block patch embedding and final norm remain on GPU. Inference only.
    """
    def __init__(self, encoder):
        self.slots = {}
        self.handles = []
        self.indices = {id(block): i for i, block in enumerate(encoder.blocks)}
        for block in encoder.blocks:
            slots = []
            for name, parameter in block.named_parameters():
                owner_path, _, attribute = name.rpartition(".")
                owner = block.get_submodule(owner_path) if owner_path else block
                slots.append((owner, attribute, parameter))
            self.slots[id(block)] = slots
            self.handles.append(block.register_forward_pre_hook(self._stage))
            self.handles.append(block.register_forward_hook(self._unstage))
        for name, child in encoder.named_children():
            if name != "blocks":
                child.to("cuda")

    def _stage(self, block, _args):
        if self.indices[id(block)] in (0, 1, 10, 20, 30, 39):
            free, _ = torch.cuda.mem_get_info()
            print("offload block", self.indices[id(block)], "gpu free", free,
                  "torch allocated", torch.cuda.memory_allocated(), flush=True)
        for owner, name, original in self.slots[id(block)]:
            setattr(owner, name, torch.nn.Parameter(original.to("cuda"), requires_grad=False))

    def _unstage(self, block, _args, _output):
        for owner, name, original in self.slots[id(block)]:
            setattr(owner, name, original)

    def close(self):
        for handle in self.handles:
            handle.remove()
