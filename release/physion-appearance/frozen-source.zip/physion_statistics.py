"""Pair-preserving summaries and exact L1 hybrid bookkeeping, not mediation."""
import numpy as np

def rank_correlation(x,y):
 def ranks(a):
  _,inverse,counts=np.unique(a,return_inverse=True,return_counts=True)
  return (np.cumsum(counts)-counts+(counts-1)/2)[inverse].astype(float)
 a=ranks(np.asarray(x));b=ranks(np.asarray(y));a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
 return None if den==0 else float(np.dot(a,b)/den)

def hybrid_l1(p0,t0,p1,t1):
 arrays=[np.asarray(x,dtype=np.float64) for x in [p0,t0,p1,t1]]
 assert len({x.shape for x in arrays})==1 and all(np.isfinite(x).all() for x in arrays)
 p0,t0,p1,t1=arrays
 e00=float(np.abs(p0-t0).mean());e01=float(np.abs(p0-t1).mean());e10=float(np.abs(p1-t0).mean());e11=float(np.abs(p1-t1).mean())
 target=((e01-e00)+(e11-e10))/2;predictor=((e10-e00)+(e11-e01))/2;delta=e11-e00
 assert np.isclose(target+predictor,delta,atol=1e-10,rtol=1e-10)
 dp=p1-p0;dt=t1-t0;den=np.linalg.norm(dp)*np.linalg.norm(dt)
 return {'E00':e00,'E01':e01,'E10':e10,'E11':e11,'native_delta_fp64':delta,'target_contribution':target,'predictor_contribution':predictor,'reconstruction_abs_error':abs(target+predictor-delta),'predictor_displacement_rms':float(np.sqrt(np.square(dp).mean())),'target_displacement_rms':float(np.sqrt(np.square(dt).mean())),'displacement_cosine':None if den==0 else float(np.sum(dp*dt)/den)}

def bootstrap_pairs(pair_rows,columns,seed,draws=50000,family=1,layout=False):
 rng=np.random.default_rng(seed);scenarios=sorted({r['scenario'] for r in pair_rows});boot=np.zeros((draws,len(columns)),np.float64);point=np.zeros(len(columns));groups={}
 for scenario in scenarios:
  rows=[r for r in pair_rows if r['scenario']==scenario];x=np.array([[r[c] for c in columns] for r in rows],float);assert np.isfinite(x).all()
  point+=x.mean(axis=0)/len(scenarios)
  if layout:
   keys=sorted({r['layout_group'] for r in rows});sums=np.array([x[[r['layout_group']==k for r in rows]].sum(axis=0) for k in keys]);counts=np.array([sum(r['layout_group']==k for r in rows) for k in keys]);idx=rng.integers(0,len(keys),size=(draws,len(keys)));b=sums[idx].sum(axis=1)/counts[idx].sum(axis=1)[:,None];groups[scenario]=len(keys)
  else:
   idx=rng.integers(0,len(rows),size=(draws,len(rows)));b=x[idx].mean(axis=1);groups[scenario]=len(rows)
  boot+=b/len(scenarios)
 alpha=.05/family
 return {'seed':seed,'draws':draws,'resampling_unit':'layout cluster within scenario, whole paired trials retained' if layout else 'matched pair within scenario, both videos and all conditions retained','stratum_groups':groups,'metrics':{c:{'mean':float(point[i]),'descriptive95':np.quantile(boot[:,i],[.025,.975]).tolist(),'family_adjusted_interval':np.quantile(boot[:,i],[alpha/2,1-alpha/2]).tolist(),'family_size':family} for i,c in enumerate(columns)}}
