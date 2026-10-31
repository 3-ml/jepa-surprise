"""Fixed episode-paired descriptive analysis and feature/score integrity audit."""
import csv,json,traceback
import numpy as np
from matched_common import *

def bootstrap(matrix,draws=50000,seed=202610258):
 rng=np.random.default_rng(seed);out=np.empty((draws,matrix.shape[1]),dtype=np.float64)
 for start in range(0,draws,500):
  count=min(500,draws-start);idx=rng.integers(0,len(matrix),size=(count,len(matrix)));out[start:start+count]=matrix[idx].mean(axis=1)
 return out

def main():
 verify();dest=OUT/'analysis';dest.mkdir(exist_ok=False)
 try:
  seal=json.loads((OUT/'input-seal.json').read_text());bases=[x['base_id'] for x in seal['manifest']]
  if len(bases)!=256 or len(set(bases))!=256 or len({x['episode'] for x in seal['manifest']})!=256:raise ValueError('Episode unit invalid')
  records={};audit_count=0;max_diff=0.;components=[]
  for model in MODELS:
   for phase,names in [('controls',OLD),('new',NEW)]:
    root=OUT/model/phase;done=json.loads((root/'complete.json').read_text())
    if done['status']!='pass' or done['rows']!=256*len(names) or sha(root/'rows.jsonl')!=done['rows_sha256']:raise ValueError('Incomplete or changed scores')
    if json.loads((root/'launch.json').read_text())['input_seal_sha256']!=sha(OUT/'input-seal.json'):raise ValueError('Input seal changed across phases')
    rows=[json.loads(s) for s in (root/'rows.jsonl').read_text().splitlines()]
    if {(r['base_id'],r['condition']) for r in rows}!={(b,n) for b in bases for n in names}:raise ValueError('Wrong/duplicate matrix keys')
    for row in rows:
     key=(model,row['base_id'],row['condition'])
     if key in records:raise ValueError('Duplicate row')
     records[key]=row
    for i,bid in enumerate(bases):
     path=root/f'{i:04d}.npz'
     if sha(path)!=done['features_sha256'][path.name]:raise ValueError('Features changed')
     with np.load(path,allow_pickle=False) as z:
      for name in names:
       p,t=z[name+'__p'],z[name+'__t']
       if p.dtype!=np.float32 or t.dtype!=np.float32 or not np.isfinite(p).all() or not np.isfinite(t).all():raise ValueError('Invalid features')
       value=float(np.mean((p.astype(np.float64)-t.astype(np.float64))**2));difference=abs(value-records[model,bid,name]['prediction_mse']);max_diff=max(max_diff,difference)
       if difference>1e-5:raise ValueError('Feature/native-score reconstruction failed')
       audit_count+=1
   for i,bid in enumerate(bases):
    features={}
    for phase,names in [('controls',OLD),('new',NEW)]:
     with np.load(OUT/model/phase/f'{i:04d}.npz',allow_pickle=False) as z:
      for name in names:features[name]=(z[name+'__p'],z[name+'__t'])
    for name in CONDITIONS:
     row=records[model,bid,name];meta=seal['manifest'][i]
     if row['input_npz_sha256']!=meta['sha256'] or row['target_sha256']!=meta['rows'][CONDITIONS.index(name)]['target_sha256']:raise ValueError('Score/input provenance mismatch')
     if name in PARENTS:
      parent=PARENTS[name];terms=decompose(*features[name],*features[parent])
      if any(abs(row[k]-v)>1e-12 for k,v in terms.items()):raise ValueError('Decomposition re-audit mismatch')
      if name.endswith('_target') or name.startswith('appearance_hue_'):
       if not np.array_equal(features[name][0],features[parent][0]):raise ValueError('Predictor invariant failed in audit')
     if name.startswith('persistent_hue_') or name.endswith('_persistent'):
      pair=name.replace('persistent_hue_','appearance_hue_') if name.startswith('persistent_hue_') else name.replace('_persistent','_target')
      if not np.array_equal(features[name][1],features[pair][1]):raise ValueError('Target invariant failed in audit')
   for name in PARENTS:
    rs=[records[model,b,name] for b in bases];fields=['predictor_shift_sq','target_shift_sq','shift_cross','residual_predictor','residual_target','native_delta']
    components.append({'model':model,'condition':name,'n':256,**{k:float(np.mean([r[k] for r in rs])) for k in fields}})
  if audit_count!=13056:raise ValueError('Wrong complete scientific denominator')
  labels=[];columns=[]
  for model in MODELS:
   for label,weights in contrasts():
    labels.append((model,label));columns.append([sum(weight*records[model,bid,cond]['prediction_mse'] for cond,weight in weights.items()) for bid in bases])
  matrix=np.asarray(columns,dtype=np.float64).T;draws=bootstrap(matrix);family=36;tail=.05/(2*family);summaries=[]
  for k,(model,label) in enumerate(labels):
   values=matrix[:,k];summaries.append({'model':model,'contrast':label,'n_episodes':256,'mean':float(values.mean()),'sd':float(values.std(ddof=1)),'ci95_descriptive':np.quantile(draws[:,k],[.025,.975]).tolist(),'bonferroni_family36_interval':np.quantile(draws[:,k],[tail,1-tail]).tolist(),'positive':int((values>0).sum()),'negative':int((values<0).sum()),'ties_exact':int((values==0).sum())})
  with (dest/'episode-contrasts.csv').open('x',newline='') as f:
   writer=csv.writer(f);writer.writerow(['base_id']+[m+':'+c for m,c in labels]);writer.writerows([[bid,*matrix[i].tolist()] for i,bid in enumerate(bases)])
  save(dest/'decomposition-means.json',components)
  save(dest/'summary.json',{'status':'complete','scope':'Inspected archival-cohort development; not fresh confirmation or training-disjoint validation','episodes':256,'models':MODELS,'conditions_per_model':17,'native_scores':audit_count,'audited_features':audit_count,'max_feature_native_score_difference':max_diff,'bootstrap_draws':50000,'bootstrap_seed':202610258,'bootstrap_unit':'episode; all conditions and all models use identical resampled indices','family_size':36,'comparisons':summaries,'interpretation':['No raw-loss comparison across models.','DINO-WM and JEPA-WM share a backbone; not independent replications.','Palette may carry task-role meaning; not automatically nuisance.','Decomposition is algebra, not causal mediation.','Intervals spanning zero are not equivalence; no utility ceiling established.','Role-panel proprioception follows actual replayed state, which may differ slightly from stored states; all paired colors share it.'],'freeze_sha256':sha(OUT/'freeze.json'),'input_seal_sha256':sha(OUT/'input-seal.json'),'at':now()})
 except BaseException as e:save(dest/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'at':now()});raise
if __name__=='__main__':main()
