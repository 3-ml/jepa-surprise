"""Exact FP32 temporal-frame deduplication; no quantization or numerical transform."""
from pathlib import Path
import numpy as np
from canonical_codec import array_sha,sha

def tensor_info(x):
 assert x.dtype==np.float32 and x.shape==(1,16,3,256,256) and np.isfinite(x).all()
 return {'tensor_sha256':array_sha(x),'fp16_sha256':array_sha(x.astype(np.float16)),'context_sha256':array_sha(x[:,:12]),'future_sha256':array_sha(x[:,12:]),'shape':list(x.shape),'dtype':'float32'}
def pack(x):
 tensor_info(x);unique=[];indices=[];seen={}
 for frame in x[0]:
  h=array_sha(frame)
  if h not in seen:seen[h]=len(unique);unique.append(frame.copy())
  assert unique[seen[h]].tobytes()==frame.tobytes();indices.append(seen[h])
 return np.stack(unique),np.array(indices,dtype=np.uint8)
def unpack(frames,index):
 assert frames.dtype==np.float32 and frames.ndim==4 and frames.shape[1:]==(3,256,256) and 1<=len(frames)<=16
 assert index.dtype==np.uint8 and index.shape==(16,) and int(index.max())<len(frames)
 x=frames[index][None];tensor_info(x);return x

def save_tensor(path,x):
 path=Path(path);frames,index=pack(x)
 with path.open('xb') as f:np.savez(f,frames=frames,index=index)
 record={**tensor_info(x),'file_sha256':sha(path),'bytes':path.stat().st_size,'codec':'exact-fp32-temporal-dedup-v1'}
 assert load_tensor(path,record).tobytes()==x.tobytes();return record

def load_tensor(path,record):
 assert sha(path)==record['file_sha256'],'Canonical tensor file changed'
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)=={'frames','index'};x=unpack(z['frames'],z['index'])
 info=tensor_info(x)
 for key in info:assert info[key]==record[key],key
 return x
