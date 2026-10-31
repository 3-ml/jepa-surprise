"""Development-only checkpoint streaming and disk-to-GPU block offload.

Use a SHA-256-verified official checkpoint. On Windows, torch.load(mmap=True)
access-violates; loading both giant encoders into RAM exhausts the host/GPU
budget. This reader indexes ZIP metadata, loads non-block weights strictly,
and reads each encoder block from the archive only during its forward pass.
"""
from __future__ import annotations

import sys
import zipfile

import torch
from stream_vjepa2_checkpoint import DTYPES, Index, meta_builder, read_tensor


class OffloadedEncoder:
    def __init__(self, model, component, archive, prefix, state):
        self.model = model
        self.component = component
        self.archive = archive
        self.prefix = prefix
        self.slots = {}
        self.handles = []
        for i, block in enumerate(model.blocks):
            slots = []
            for name, orig in block.named_parameters():
                owner_path, _, attr = name.rpartition(".")
                owner = block.get_submodule(owner_path) if owner_path else block
                full_name = f"blocks.{i}.{name}"
                slots.append((owner, attr, orig, state[full_name]))
            self.slots[id(block)] = (i, slots)
            self.handles.append(block.register_forward_pre_hook(self._stage))
            self.handles.append(block.register_forward_hook(self._release))
        for name, child in model.named_children():
            if name != "blocks":
                child.to("cuda")

    def _stage(self, block, _args):
        i, slots = self.slots[id(block)]
        if i in (0, 10, 20, 30, 39):
            free, _ = torch.cuda.mem_get_info()
            print(self.component, "block", i, "gpu_free_bytes", free, flush=True)
        for owner, name, orig, record in slots:
            tensor = read_tensor(self.archive, self.prefix, record).to("cuda")
            setattr(owner, name, torch.nn.Parameter(tensor, requires_grad=False))
            del tensor

    def _release(self, block, _args, _output):
        _i, slots = self.slots[id(block)]
        for owner, name, orig, _record in slots:
            setattr(owner, name, orig)

    def close(self):
        for h in self.handles:
            h.remove()


class DiskOffload:
    def __init__(self, checkpoint, root, spec):
        sys.path.insert(0, str(root))
        from src.models import predictor as predictor_module
        from src.models import vision_transformer as vit
        self.archive = zipfile.ZipFile(checkpoint)
        self.prefix = self.archive.namelist()[0].split("/")[0]
        with self.archive.open(self.prefix + "/data.pkl") as handle:
            payload = Index(handle).load()
        common = {"img_size": spec["resolution"], "num_frames": spec["frames_per_clip"],
                  "uniform_power": True, "use_activation_checkpointing": False,
                  "use_rope": spec["use_rope"]}
        self.models, self.offloads, self.detail = [], [], {}
        builders = (
            lambda: getattr(vit, spec["architecture"])(**common),
            lambda: getattr(vit, spec["architecture"])(**common),
            lambda: predictor_module.vit_predictor(
                img_size=spec["resolution"], num_frames=spec["frames_per_clip"],
                embed_dim=spec["embed_dim"], predictor_embed_dim=384, depth=12,
                num_heads=12, uniform_power=True, use_activation_checkpointing=False,
                use_mask_tokens=True, use_rope=spec["use_rope"],
                zero_init_mask_tokens=True, num_mask_tokens=spec["num_mask_tokens"]),
        )
        for component, builder in zip(("encoder", "target_encoder", "predictor"), builders):
            model = meta_builder(builder)
            state = {k.removeprefix("module.backbone."): v for k, v in payload[component].items()}
            expected = model.state_dict()
            if set(expected) != set(state):
                raise ValueError(f"{component} checkpoint/model key mismatch: "
                                 f"missing {sorted(set(expected)-set(state))[:5]}, "
                                 f"extra {sorted(set(state)-set(expected))[:5]}")
            for name, rec in state.items():
                if (tuple(expected[name].shape) != tuple(rec["size"])
                        or expected[name].dtype != DTYPES[rec["storage"]["stype"]]):
                    raise ValueError(f"{component} checkpoint tensor mismatch: {name}")
            for name, rec in state.items():
                if name.startswith("blocks.") and component != "predictor":
                    continue
                tensor = read_tensor(self.archive, self.prefix, rec)
                result = model.load_state_dict({name: tensor}, strict=False, assign=True)
                if result.unexpected_keys:
                    raise ValueError(f"{component} tensor rejected: {name}")
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            if component != "predictor":
                offload = OffloadedEncoder(model, component, self.archive, self.prefix, state)
                self.offloads.append(offload)
            else:
                if any(t.is_meta for t in model.state_dict().values()):
                    raise ValueError("predictor still has meta tensors")
                model.to("cuda")
            self.detail[component] = {"verified_names_shapes_dtypes": len(state),
                                      "blocks_loaded_on_demand": component != "predictor"}
            self.models.append(model)
            print("indexed model", component, self.detail[component], flush=True)
        self.encoder, self.target_encoder, self.predictor = self.models

    def close(self):
        for offload in self.offloads:
            offload.close()
        self.archive.close()
