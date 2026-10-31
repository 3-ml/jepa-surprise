"""Lossless FP32 tensor storage. No resizing, normalization, quantization or model code."""
from pathlib import Path
import hashlib
import numpy as np

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def encode(a):
 a=np.ascontiguousarray(a);assert a.dtype==np.float32 and a.ndim==5 and a.shape[0]==1 and a.shape[2]==3 and np.isfinite(a).all()
 frames=[];lookup={};mapping=[]
 for frame in a[0]:
  key=array_sha(frame)
  if key not in lookup:lookup[key]=len(frames);frames.append(frame.copy())
  else:assert frame.tobytes()==frames[lookup[key]].tobytes()
  mapping.append(lookup[key])
 unique=np.stack(frames);out={'frame_map':np.asarray(mapping,np.int32),'shape':np.asarray(a.shape,np.int32),'format':np.array(1,np.int32)}
 palettes=[];indices=[]
 for c in range(3):
  values,inv=np.unique(unique[:,c],return_inverse=True)
  if len(values)>256:break
  palettes.append(values);indices.append(inv.reshape(unique[:,c].shape).astype(np.uint8))
 if len(palettes)==3:
  out.update(mode=np.array(1,np.int32),indices=np.stack(indices,axis=1))
  for c in range(3):out[f'palette_{c}']=palettes[c]
  if array_sha(decode(out))==array_sha(a):return out
 # Exact float fallback also handles signed-zero distinctions; it is not approximation.
 return {**{k:v for k,v in out.items() if k in ['frame_map','shape','format']},'mode':np.array(0,np.int32),'frames':unique}
def decode(z):
 assert int(z['format'])==1
 if int(z['mode'])==1:
  idx=z['indices'];frames=np.stack([z[f'palette_{c}'][idx[:,c]] for c in range(3)],axis=1)
 else:assert int(z['mode'])==0;frames=z['frames']
 a=np.ascontiguousarray(frames[z['frame_map']][None]);assert a.dtype==np.float32 and list(a.shape)==list(z['shape']) and np.isfinite(a).all();return a
def save_tensor(path,a):
 p=Path(path);assert not p.exists();payload=encode(a);np.savez_compressed(p,**payload)
 with np.load(p,allow_pickle=False) as z:b=decode(z)
 assert a.tobytes()==b.tobytes()
 return {'tensor_sha256':array_sha(a),'fp16_sha256':array_sha(a.astype(np.float16)),'context_sha256':array_sha(a[:,:12]),'future_sha256':array_sha(a[:,12:]),'file_sha256':sha(p),'bytes':p.stat().st_size,'shape':list(a.shape),'dtype':'float32','unique_frames':len(set(payload['frame_map'].tolist())),'storage_mode':'exact_palette' if int(payload['mode'])==1 else 'exact_float32'}
def load_tensor(path,record):
 assert sha(path)==record['file_sha256']
 with np.load(path,allow_pickle=False) as z:a=decode(z)
 assert list(a.shape)==record['shape'] and array_sha(a)==record['tensor_sha256'] and array_sha(a.astype(np.float16))==record['fp16_sha256']
 assert array_sha(a[:,:12])==record['context_sha256'] and array_sha(a[:,12:])==record['future_sha256'];return a
