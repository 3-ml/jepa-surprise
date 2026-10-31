"""All archival controls for every model must pass before new cells are scored."""
import argparse,csv,json,traceback
import numpy as np
from matched_common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--phase',choices=['controls','new'],required=True);args=ap.parse_args()
 verify();seal=json.loads((OUT/'input-seal.json').read_text())
 if seal['status']!='pass' or seal['cells']!=4352 or seal['conditions']!=list(CONDITIONS):raise ValueError('Incomplete input seal')
 if args.phase=='new':
  for m in MODELS:
   if json.loads((OUT/m/'controls/complete.json').read_text())['status']!='pass':raise ValueError('All models must pass controls first')
 dest=OUT/args.model/args.phase;dest.mkdir(parents=True,exist_ok=False);rows=[];files={}
 save(dest/'launch.json',{'at':now(),'model':args.model,'phase':args.phase,'input_seal_sha256':sha(OUT/'input-seal.json'),'freeze_sha256':sha(OUT/'freeze.json')})
 try:
  from matched_adapters import load
  score=load(args.model);names=OLD if args.phase=='controls' else NEW
  refs={(r['base_id'],r['condition']):r for r in csv.DictReader((ROOT/f'data/{args.model}-reference-scores.csv').open())}
  previous={}
  if args.phase=='new':
   control_rows=OUT/args.model/'controls/rows.jsonl'
   if sha(control_rows)!=json.loads((control_rows.parent/'complete.json').read_text())['rows_sha256']:raise ValueError('Control rows changed')
   previous={(r['base_id'],r['condition']):r for r in map(json.loads,control_rows.read_text().splitlines())}
  with (dest/'rows.jsonl').open('x') as log,(dest/'observations.jsonl').open('x') as raw_log:
   for i,base in enumerate(seal['manifest']):
    if sha(base['path'])!=base['sha256']:raise ValueError('Sealed input file changed')
    with np.load(base['path'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
    features={};scores={};new_features={}
    if args.phase=='new':
     control_path=OUT/args.model/'controls'/f'{i:04d}.npz';done=json.loads((OUT/args.model/'controls/complete.json').read_text())
     if sha(control_path)!=done['features_sha256'][control_path.name]:raise ValueError('Control features changed')
     with np.load(control_path,allow_pickle=False) as z:
      for name in OLD:features[name]=(z[name+'__p'],z[name+'__t']);scores[name]=previous[base['base_id'],name]['prediction_mse']
    for name in names:
     j=CONDITIONS.index(name);p,t,temporal=score(data['pixels'][j],data['actions'][j],data['states'][j]);s=temporal[-1]
     row={'model':args.model,'base_id':base['base_id'],'episode':base['episode'],'condition':name,'prediction_mse':s,'temporal_prediction_mse':temporal,'target_sha256':base['rows'][j]['target_sha256'],'input_npz_sha256':base['sha256']}
     raw_log.write(json.dumps(row,allow_nan=False)+'\n');raw_log.flush()
     if i==0 and name==names[0]:
      pp,tt,repeat=score(data['pixels'][j],data['actions'][j],data['states'][j])
      if not np.array_equal(p,pp) or not np.array_equal(t,tt) or temporal!=repeat:raise ValueError('Uncached repeat not exact')
     valid=True
     if args.phase=='controls':
      ref=refs[base['base_id'],name];old=json.loads(ref['temporal_prediction_mse']);difference=max(abs(a-b) for a,b in zip(temporal,old))
      valid=len(old)==3 and difference<=1e-5 and abs(s-float(ref['prediction_mse']))<=1e-5
      row.update(reference_max_temporal_abs_difference=difference,reference_pass=valid)
     parent=PARENTS.get(name)
     if parent:
      p0,t0=features[parent];terms=decompose(p,t,p0,t0)
      if abs(terms['delta_mse_float64']-(s-scores[parent]))>1e-5:raise ValueError('Native/decomposition discrepancy')
      row.update(parent=parent,native_delta=s-scores[parent],**terms)
      if name.endswith('_target') or name.startswith('appearance_hue_'):
       if not np.array_equal(p,p0):raise ValueError('Target-only predictor changed')
     if name.startswith('persistent_hue_') or name.endswith('_persistent'):
      pair=name.replace('persistent_hue_','appearance_hue_') if name.startswith('persistent_hue_') else name.replace('_persistent','_target')
      if not np.array_equal(t,features[pair][1]):raise ValueError('Shared target embedding changed between scopes')
     log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();rows.append(row)
     if not valid:raise ValueError('Old-control parity failed; no new cells allowed')
     features[name]=(p,t);scores[name]=s;new_features[name+'__p']=p;new_features[name+'__t']=t
    path=dest/f'{i:04d}.npz';np.savez(path,**new_features);files[path.name]=sha(path)
    save(dest/'progress.json',{'completed_bases':i+1,'completed_cells':len(rows),'planned_bases':256,'at':now()});print(args.model,args.phase,i+1,flush=True)
  if len(rows)!=256*len(names):raise ValueError('Wrong denominator')
  save(dest/'complete.json',{'status':'pass','rows':len(rows),'base_configurations':256,'features_sha256':files,'rows_sha256':sha(dest/'rows.jsonl'),'at':now()})
 except BaseException as e:
  if 'new_features' in locals():
   partial=dict(new_features)
   if 'p' in locals() and 't' in locals():partial.update(failed_or_last_p=p,failed_or_last_t=t)
   np.savez(dest/'partial-base-features.npz',**partial)
  save(dest/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'completed_rows':len(rows),'last_base':base['base_id'] if 'base' in locals() else None,'last_condition':name if 'name' in locals() else None,'at':now()});raise
if __name__=='__main__':main()
