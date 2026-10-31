"""Native archived score accepting canonical tensors at the preprocessing boundary."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,inspect,json,sys
import numpy as np
import torch
from transformers import VJEPA2Model
from canonical_codec import array_sha,sha
ROOT=Path(__file__).resolve().parent
OLD=Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'reviewer-physion-desktop-20261026-v2'))
CORE_SOURCE = ROOT.parents[1] / 'physion-core/source/run_vjepa2_section_surprise.py'
assert sha(CORE_SOURCE) == '72dddefb7e6b4c0dd87090249b9fd09ee4064b79d72a15fad3a346a60a3312ac', 'Frozen Physion core hash mismatch'
spec=importlib.util.spec_from_file_location('archived_physion_core',CORE_SOURCE)
core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
make_masks=core.make_masks;distances=core.distances;target_frame_ranges=core.target_frame_ranges

@torch.no_grad()
def score_tensor(model,inputs,indices,args):
    inputs = inputs.to(args.device)
    dtype = getattr(torch, args.dtype)
    if args.device.startswith("cuda"):
        inputs = inputs.to(dtype=dtype)
    context_mask, target_mask, target_time_ids, target_start, _ = make_masks(
        inputs, model, args.target_start_fraction
    )
    with torch.autocast("cuda", dtype=dtype, enabled=args.device.startswith("cuda")):
        output = model(
            inputs,
            context_mask=context_mask,
            target_mask=target_mask,
        ).predictor_output
    token_distances = distances(
        output.last_hidden_state,
        output.target_hidden_state,
        args.metric,
        args.normalization,
    )
    temporal = []
    for time_id in torch.unique(target_time_ids, sorted=True):
        temporal.append(token_distances[:, target_time_ids == time_id].mean(dim=1))
    temporal_tensor = torch.stack(temporal, dim=1).squeeze(0)
    temporal_values = temporal_tensor.detach().cpu().tolist()
    peak_index = int(np.argmax(np.asarray(temporal_values)))
    frame_ranges = target_frame_ranges(indices, int(model.config.tubelet_size), target_time_ids)
    return {
        "target_start_tubelet": target_start,
        "target_tubelet_source_frame_ranges_0based": frame_ranges,
        "scalar_surprise": float(token_distances.mean().item()),
        "temporal_tubelet_surprise": temporal_values,
        "peak_target_tubelet_index": peak_index,
        "peak_source_frame_range_0based": frame_ranges[peak_index],
    }

def verify_adapter():
 old=ast.parse(inspect.getsource(core.score_section)).body[0];new=ast.parse(inspect.getsource(score_tensor)).body[0]
 left=ast.dump(ast.Module(body=old.body[3:],type_ignores=[]),include_attributes=False);right=ast.dump(ast.Module(body=new.body[1:],type_ignores=[]),include_attributes=False)
 assert left==right,'native mask/forward/autocast/reduction body changed'
 return {'native_body_AST_exact':True,'native_body_AST_sha256':hashlib.sha256(left.encode()).hexdigest(),'original_source_sha256':sha(CORE_SOURCE),'adapter_sha256':sha(__file__),'changed_boundary':'replace video indexing/processor call with supplied exact FP32 tensor; no other executable-body change'}

def loading_info_json(info):
 def encode(value):
  if isinstance(value,(set,frozenset)):return sorted(value)
  raise TypeError(f'Unsupported loading metadata type: {type(value).__name__}')
 return json.dumps(info,indent=2,default=encode,allow_nan=False)

class Scorer:
 def __init__(self):
  self.proof=verify_adapter();torch.set_num_threads(4);torch.manual_seed(2026102600);torch.cuda.manual_seed_all(2026102600)
  assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'RTX 3080' in torch.cuda.get_device_name(0)
  assert torch.__version__=='2.8.0+cu128'
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
  assert sha(OLD/'model/model.safetensors')=='25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b'
  self.model,self.info=VJEPA2Model.from_pretrained(OLD/'model',local_files_only=True,torch_dtype=torch.float16,output_loading_info=True)
  (ROOT/'loading-info.json').write_text(loading_info_json(self.info));assert not any(self.info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
  assert sum(p.numel() for p in self.model.parameters())==325971328
  assert self.model.config._attn_implementation=='sdpa'
  self.model=self.model.to('cuda').eval()
  for p in self.model.parameters():p.requires_grad_(False)
  self.args=argparse.Namespace(device='cuda',dtype='float16',target_start_fraction=.75,metric='l1',normalization='both')
  self.pre=self.model.register_forward_pre_hook(self.pre_hook);self.post=self.model.register_forward_hook(self.post_hook)
  self.want_features=False;self.captured={};self.expected_fp16=None
 def pre_hook(self,module,args):
  x=args[0];assert x.dtype==torch.float16 and tuple(x.shape)==(1,16,3,256,256)
  assert array_sha(x.detach().cpu().numpy())==self.expected_fp16,'actual CUDA inputs differ from canonical Desktop FP16 bytes'
 def post_hook(self,module,args,out):
  p=out.predictor_output.last_hidden_state;t=out.predictor_output.target_hidden_state
  assert tuple(p.shape)==tuple(t.shape)==(1,512,1024) and torch.isfinite(p).all() and torch.isfinite(t).all()
  with torch.autocast('cuda',enabled=False):loss=distances(p,t,'l1','both')
  self.captured={'loss':loss[0].detach().cpu().numpy()}
  if self.want_features:self.captured.update(p=p[0].detach().cpu().numpy(),t=t[0].detach().cpu().numpy())
 def score(self,x,record,features=False):
  assert x.shape==(1,16,3,256,256) and x.dtype==np.float32 and np.isfinite(x).all() and array_sha(x)==record['tensor_sha256']
  self.want_features=features;self.expected_fp16=record['fp16_sha256'];self.captured={}
  result=score_tensor(self.model,torch.from_numpy(x),np.arange(16),self.args);loss=self.captured['loss']
  assert loss.shape==(512,) and np.isfinite(loss).all()
  assert abs(float(loss.astype(float).mean())-result['scalar_surprise'])<=1e-6
  assert np.max(np.abs(loss.reshape(2,256).astype(float).mean(axis=1)-result['temporal_tubelet_surprise']))<=1e-6
  result.update(tensor_sha256=record['tensor_sha256'],actual_CUDA_fp16_sha256=self.expected_fp16,loss_sha256=array_sha(loss))
  return result,self.captured
