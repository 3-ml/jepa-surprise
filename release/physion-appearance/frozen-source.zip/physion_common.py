"""Shared fixed inputs, transforms and durable output helpers; no model imports."""
from pathlib import Path
import csv,datetime,hashlib,json,os
import cv2
import numpy as np
from PIL import Image
ROOT=Path(os.environ.get('WMPAI_STUDY_ROOT',str(Path(__file__).resolve().parent)))
A_CONDITIONS=('original','static_first','static_middle','static_last')
B_CONDITIONS=('original','persistent_hue','target_only_hue','persistent_brightness','target_only_brightness')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def append(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def read_rows(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def video_frames(path):
 cap=cv2.VideoCapture(str(path));assert cap.isOpened(),str(path);frames=[];fps=float(cap.get(cv2.CAP_PROP_FPS));reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 try:
  while True:
   ok,frame=cap.read()
   if not ok:break
   frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
 finally:cap.release()
 assert frames and len(frames)==reported,(str(path),len(frames),reported)
 a=np.stack(frames);assert a.dtype==np.uint8 and a.ndim==4 and a.shape[-1]==3
 return a,fps

def static_variant(clip,name):
 assert clip.shape[0]==16
 if name=='original':return clip.copy()
 index={'static_first':0,'static_middle':8,'static_last':15}[name]
 return np.repeat(clip[index:index+1],16,axis=0)

def hue(clip,bins=21):
 if bins==0:return clip.copy()
 result=[]
 for frame in clip:
  hsv=np.asarray(Image.fromarray(frame).convert('HSV')).copy();hsv[...,0]=((hsv[...,0].astype(np.uint16)+bins)%256).astype(np.uint8)
  result.append(np.asarray(Image.fromarray(hsv,mode='HSV').convert('RGB')))
 return np.stack(result)

def appearance_variant(clip,name):
 assert clip.dtype==np.uint8 and clip.shape[0]==16
 if name=='original':return clip.copy()
 transformed=hue(clip) if name.endswith('hue') else ((clip.astype(np.uint16)*4)//5).astype(np.uint8)
 if name.startswith('persistent_'):return transformed
 assert name.startswith('target_only_');out=clip.copy();out[12:]=transformed[12:];return out

def motion(clip):
 d=np.abs(np.diff(clip.astype(np.int16),axis=0))
 return {'mean_consecutive_rgb':float(d.mean()),'changed_fraction_gt5':float((d.max(axis=-1)>5).mean()),'first_last_mean_rgb':float(np.abs(clip[-1].astype(np.int16)-clip[0].astype(np.int16)).mean())}

def historical_path(value):return ROOT/'data/testdata_v1'/value.replace('\\','/').split('/testdata_v1/',1)[1]

def validate_population(pop):
 assert len(pop)==192 and len({r['relative_path'] for r in pop})==192
 pairs={}
 for r in pop:pairs.setdefault(r['pair_id'],[]).append(r)
 assert len(pairs)==96 and all(len(v)==2 for v in pairs.values())
 assert all({r['label'] for r in v}=={'0','1'} for v in pairs.values())
 counts={}
 for v in pairs.values():
  assert len({r['scenario'] for r in v})==1;counts[v[0]['scenario']]=counts.get(v[0]['scenario'],0)+1
 assert sorted(counts.values())==[32,32,32]
 return pairs
