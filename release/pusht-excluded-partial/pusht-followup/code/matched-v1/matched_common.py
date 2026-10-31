from pathlib import Path
import hashlib,json,datetime
import numpy as np
ROOT=Path.home()/'jepa-wmpai/pusht-followup';OUT=ROOT/'development-matched256-v1'
MODELS=('lewm','dino','jepa')
OLD=('official_clean','lawful_scale_0p0','appearance_hue_030','appearance_hue_060','opposite_scale_0p25','opposite_scale_0p75')
ROLES=('block_blue','block_orange_matched','agent_gray','agent_purple_matched')
NEW=('persistent_hue_030','persistent_hue_060','rerender_default')+tuple(r+'_'+s for r in ROLES for s in ('target','persistent'))
CONDITIONS=OLD+NEW
PARENTS={**{k:'lawful_scale_0p0' for k in CONDITIONS if 'hue_' in k},'rerender_default':'official_clean',**{r+'_'+s:'rerender_default' for r in ROLES for s in ('target','persistent')}}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,data):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');t.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify():
 lock=json.loads((OUT/'freeze.json').read_text())
 for p,h in lock['files'].items():
  if sha(p)!=h:raise ValueError('Frozen file changed: '+p)
 return lock

def decompose(p,t,p0,t0):
 p,t,p0,t0=[x.astype(np.float64) for x in (p,t,p0,t0)]
 dp,dt,e=p-p0,t-t0,p0-t0
 terms={'predictor_shift_sq':float(np.mean(dp*dp)),'target_shift_sq':float(np.mean(dt*dt)),'shift_cross':float(-2*np.mean(dp*dt)),'residual_predictor':float(2*np.mean(e*dp)),'residual_target':float(-2*np.mean(e*dt))}
 reconstructed=sum(terms.values());direct=float(np.mean((p-t)**2)-np.mean(e**2))
 if not np.isclose(reconstructed,direct,rtol=1e-10,atol=1e-10):raise ValueError('Decomposition identity failed')
 return {**terms,'delta_mse_float64':direct,'reconstruction_abs_error':abs(reconstructed-direct)}

def contrasts():
 result=[]
 for dose,scale in [(30,'0p25'),(60,'0p75')]:
  a=f'appearance_hue_{dose:03d}';p=f'persistent_hue_{dose:03d}';o='opposite_scale_'+scale
  result.extend([(f'hue{dose}_persistent_minus_target',{p:1,a:-1}),(f'hue{dose}_target_minus_opposite',{a:1,o:-1}),(f'hue{dose}_persistent_minus_opposite',{p:1,o:-1})])
 for role,control in [('block_blue','block_orange_matched'),('agent_gray','agent_purple_matched')]:
  for scope in ['target','persistent']:result.append((role+'_'+scope+'_minus_matched',{role+'_'+scope:1,control+'_'+scope:-1}))
  result.append((role+'_persistent_gap_minus_target_gap',{role+'_persistent':1,control+'_persistent':-1,role+'_target':-1,control+'_target':1}))
 assert len(result)==12
 return result
