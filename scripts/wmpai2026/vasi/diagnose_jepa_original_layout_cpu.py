"""One known failed archival fixture, unchanged original batch-8 scorer."""
from pathlib import Path
import csv,hashlib,json,os,subprocess,sys,traceback,datetime
import numpy as np
ROOT=Path.home()/'jepa-wmpai/pusht-followup';OUT=ROOT/'logs/jepa-original-layout-failed-fixture-cpu-v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def main():
 OUT.mkdir(exist_ok=False)
 try:
  bid='pusht:12008:63'
  with np.load(ROOT/'data/pusht-controlled256-bundle.npz',allow_pickle=False) as z:
   indices=np.flatnonzero(z['base_id']==bid)
   if len(indices)!=1:raise ValueError('Wrong fixture identity')
   np.savez_compressed(OUT/'fixture.npz',**{k:z[k][indices] for k in z.files})
  provenance=json.loads((ROOT/'data/pusht-controlled256-bundle.json').read_text());provenance.update(bundle_sha256=sha(OUT/'fixture.npz'),samples=1,engineering_subset_of='359dae44cdf32f5dcd6917e6624c3be4dcc0b268e9b51b0e6978828605f7fdf7');save('fixture.json',provenance)
  checkpoint=json.loads((ROOT/'logs/jepa-wm-staged.json').read_text())['path']
  command=[sys.executable,'-u',str(ROOT/'code/scripts/run_jepa_wm_pusht_controlled_displacement.py'),'--jepa-wm-root',str(ROOT/'code/jepa-wms'),'--stable-worldmodel-root',str(ROOT/'code/stable-worldmodel'),'--checkpoint',checkpoint,'--input-bundle',str(OUT/'fixture.npz'),'--bundle-provenance',str(OUT/'fixture.json'),'--reference-scores',str(ROOT/'data/dino-reference-scores.csv'),'--output-dir',str(OUT/'run'),'--samples','1','--batch-size','8','--device','cpu','--phase','smoke']
  save('lock.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base_id':bid,'selection':'Known failed archived control, engineering diagnosis only; not scientific sample selection','tolerance':1e-5,'recipe':'Unchanged original program, original batch=8 and original target-group batching, explicitly CPU-only; same 1e-5 tolerance','command':command,'wrapper_sha256':sha(__file__),'native_script_sha256':sha(ROOT/'code/scripts/run_jepa_wm_pusht_controlled_displacement.py')})
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','TORCH_HOME':str(ROOT/'torch-cache'),'HF_HOME':str(ROOT/'hf-cache'),'XFORMERS_DISABLED':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MPLBACKEND':'Agg','SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy','PYTHONPATH':str(ROOT/'code')}
  with (OUT/'run.log').open('x') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
  refs={(r['base_id'],r['condition']):r for r in csv.DictReader((ROOT/'data/jepa-reference-scores.csv').open())};rows=[]
  for row in csv.DictReader((OUT/'run/scores.csv').open()):
   ref=refs[row['base_id'],row['condition']];a=json.loads(row['temporal_prediction_mse']);b=json.loads(ref['temporal_prediction_mse']);d=max(abs(x-y) for x,y in zip(a,b))
   rows.append({'condition':row['condition'],'temporal':a,'reference':b,'max_abs_difference':d,'target_exact':row['target_sha256']==ref['target_sha256'],'pass':d<=1e-5 and row['target_sha256']==ref['target_sha256']})
  save('result.json',{'status':'pass' if len(rows)==18 and all(r['pass'] for r in rows) else 'parity_failed','rows':rows,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_missing_cells':0})
 except BaseException as e:save('failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
