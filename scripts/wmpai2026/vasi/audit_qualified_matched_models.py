"""CPU-only reconstruction audit of two complete models; no contrast estimates."""
from pathlib import Path
import json,sys,traceback
import numpy as np
ROOT=Path.home()/'jepa-wmpai/pusht-followup';sys.path.insert(0,str(ROOT/'code/matched-v1'))
import matched_common as c

def dotmean(a,b):return float(np.dot(a,b)/a.size)
def main():
 dest=c.OUT/'qualified-model-audit.json';failure=c.OUT/'qualified-model-audit-failure.json'
 if dest.exists() or failure.exists():raise ValueError('No automatic retry/overwrite')
 try:
  c.verify();seal=json.loads((c.OUT/'input-seal.json').read_text());bases=[m['base_id'] for m in seal['manifest']]
  if len(bases)!=256 or len(set(bases))!=256 or len({m['episode'] for m in seal['manifest']})!=256:raise ValueError('Invalid episode units')
  for i,m in enumerate(seal['manifest']):
   if c.sha(c.OUT/'inputs'/f'{i:04d}.npz')!=m['sha256']:raise ValueError('Input changed')
  results=[];count=0
  for model in ['lewm','dino']:
   records={};completions={};max_scalar=0.;max_identity=0.;max_native_delta=0.;shape=None
   for phase,names in [('controls',c.OLD),('new',c.NEW)]:
    root=c.OUT/model/phase;done=json.loads((root/'complete.json').read_text());completions[phase]=done
    if done['status']!='pass' or done['rows']!=256*len(names) or c.sha(root/'rows.jsonl')!=done['rows_sha256']:raise ValueError('Score completion changed')
    if json.loads((root/'launch.json').read_text())['input_seal_sha256']!=c.sha(c.OUT/'input-seal.json'):raise ValueError('Seal changed')
    rows=[json.loads(x) for x in (root/'rows.jsonl').read_text().splitlines()]
    if len(rows)!=256*len(names) or {(r['base_id'],r['condition']) for r in rows}!={(b,n) for b in bases for n in names}:raise ValueError('Invalid row keyset')
    for r in rows:
     if phase=='controls' and (not r['reference_pass'] or r['reference_max_temporal_abs_difference']>1e-5):raise ValueError('Invalid archival control')
     if len(r['temporal_prediction_mse'])!=3 or not np.isfinite(r['temporal_prediction_mse']).all() or r['prediction_mse']!=r['temporal_prediction_mse'][-1]:raise ValueError('Invalid native temporal scores')
     records[r['base_id'],r['condition']]=r
   for i,bid in enumerate(bases):
    features={};meta=seal['manifest'][i]
    for phase,names in [('controls',c.OLD),('new',c.NEW)]:
     path=c.OUT/model/phase/f'{i:04d}.npz'
     if c.sha(path)!=completions[phase]['features_sha256'][path.name]:raise ValueError('Feature hash changed')
     with np.load(path,allow_pickle=False) as z:
      if set(z.files)!={n+s for n in names for s in ['__p','__t']}:raise ValueError('Invalid feature keys')
      for n in names:
       p,t=z[n+'__p'],z[n+'__t']
       if p.dtype!=np.float32 or t.dtype!=np.float32 or p.ndim!=1 or p.shape!=t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():raise ValueError('Invalid vectors')
       if shape is None:shape=p.shape
       if p.shape!=shape:raise ValueError('Changing native feature shape')
       features[n]=(p,t);err=p.astype(np.float64)-t.astype(np.float64);d=abs(dotmean(err,err)-records[bid,n]['prediction_mse']);max_scalar=max(max_scalar,d)
       if d>1e-5:raise ValueError('Native score reconstruction failed')
       count+=1
    for n in c.CONDITIONS:
     r=records[bid,n]
     if r['input_npz_sha256']!=meta['sha256'] or r['target_sha256']!=meta['rows'][c.CONDITIONS.index(n)]['target_sha256']:raise ValueError('Input provenance changed')
     if n in c.PARENTS:
      parent=c.PARENTS[n];p,t,p0,t0=[x.astype(np.float64) for x in (*features[n],*features[parent])];dp=p-p0;dt=t-t0;e=p0-t0
      terms={'predictor_shift_sq':dotmean(dp,dp),'target_shift_sq':dotmean(dt,dt),'shift_cross':-2*dotmean(dp,dt),'residual_predictor':2*dotmean(e,dp),'residual_target':-2*dotmean(e,dt)}
      direct=dotmean(p-t,p-t)-dotmean(e,e);identity=abs(sum(terms.values())-direct);max_identity=max(max_identity,identity)
      if not np.isclose(sum(terms.values()),direct,rtol=1e-10,atol=1e-10) or any(abs(r[k]-v)>1e-12 for k,v in terms.items()) or abs(r['delta_mse_float64']-direct)>1e-12:raise ValueError('Decomposition audit failed')
      native=r['prediction_mse']-records[bid,parent]['prediction_mse'];d=abs(native-direct);max_native_delta=max(max_native_delta,d)
      if r['native_delta']!=native or d>1e-5:raise ValueError('Native delta reconstruction failed')
      if (n.endswith('_target') or n.startswith('appearance_hue_')) and not np.array_equal(features[n][0],features[parent][0]):raise ValueError('Predictor invariant failed')
     if n.startswith('persistent_hue_') or n.endswith('_persistent'):
      pair=n.replace('persistent_hue_','appearance_hue_') if n.startswith('persistent_hue_') else n.replace('_persistent','_target')
      if not np.array_equal(features[n][1],features[pair][1]):raise ValueError('Target invariant failed')
    c.save(c.OUT/'qualified-model-audit-progress.json',{'model':model,'completed_bases':i+1,'at':c.now()})
   results.append({'model':model,'cells':4352,'feature_dimensions':list(shape),'max_scalar_reconstruction_error':max_scalar,'max_decomposition_identity_error':max_identity,'max_native_delta_reconstruction_error':max_native_delta,'status':'pass'})
  if count!=8704:raise ValueError('Partial matrix denominator mismatch')
  c.save(dest,{'status':'pass_for_two_qualified_models','cells':count,'models':results,'all_input_files_rehashed':256,'blocked_model':'jepa','combined_experiment_complete':False,'contrasts_or_intervals_computed':False,'freeze_sha256':c.sha(c.OUT/'freeze.json'),'execution_amendment_sha256':c.sha(c.OUT/'execution-amendment-1.json'),'auditor_sha256':c.sha(__file__),'at':c.now()})
 except BaseException as e:c.save(failure,{'error':repr(e),'traceback':traceback.format_exc(),'at':c.now()});raise
if __name__=='__main__':main()
