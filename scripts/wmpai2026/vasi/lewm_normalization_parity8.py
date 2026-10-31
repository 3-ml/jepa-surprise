"""Original full-column action normalization and eight archived LeWM GPU fixtures."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,os,sys,time,traceback
ROOT=Path.home()/'jepa-wmpai/pusht-followup'
os.environ.update(HF_HOME=str(ROOT/'hf-cache'),TORCH_HOME=str(ROOT/'torch-cache'),XFORMERS_DISABLED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MPLBACKEND='Agg',SDL_VIDEODRIVER='dummy')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,data):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
 import torch,numpy as np
 torch.set_num_threads(4)
 p=ROOT/'data/full-dataset-actions.npy';prov=json.loads((ROOT/'data/full-dataset-actions-provenance.json').read_text())
 if sha(p)!=prov['actions_sha256'] or prov['dataset_sha256']!='b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb':raise ValueError('Full action data mismatch')
 raw=torch.from_numpy(np.load(p,allow_pickle=False)).float();raw=raw[~torch.isnan(raw).any(dim=1)]
 mean,std=raw.mean(dim=0),raw.std(dim=0)
 if not torch.isfinite(raw).all() or not (std>0).all():raise ValueError('Invalid normalization data')
 record={'status':'complete','action_mean':mean.tolist(),'action_std':std.tolist(),'formula':'torch.float32 full-column mean(dim=0), std(dim=0) with correction=1, after removing any-NaN rows; original make_dataset formula','valid_rows':len(raw),'actions_sha256':prov['actions_sha256'],'dataset_sha256':prov['dataset_sha256'],'torch':torch.__version__,'threads':4,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 dest=ROOT/'data/action-normalization.json'
 if dest.exists():
  old=json.loads(dest.read_text())
  if old['action_mean']!=record['action_mean'] or old['action_std']!=record['action_std']:raise ValueError('Normalization changed')
 else:save(dest,record)
 del raw
 if a.prepare_only:print(json.dumps(record),flush=True);return
 out=ROOT/'logs/lewm-gpu-parity8-v1';out.mkdir(exist_ok=False);rows=[];started=time.time()
 save(out/'lock.json',{'episodes':'first eight unchanged bundle rows','condition':'official_clean','absolute_tolerance':1e-5,'repeat_exact_required':True,'normalization':record,'source_sha256':sha(__file__),'reference_sha256':sha(ROOT/'data/lewm-reference-scores.csv')})
 try:
  if not torch.cuda.is_available():raise ValueError('No authorized GPU')
  torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  sys.path.insert(0,str(ROOT/'code'))
  from scripts.run_lewm_prediction_error_audit import load_model,normalize_action_blocks
  import stable_pretraining as spt
  model,snapshot,cfg,keys=load_model('pusht',torch.device('cuda'))
  preprocess=spt.data.transforms.Compose(spt.data.transforms.ToImage(**spt.data.dataset_stats.ImageNet,source='pixels',target='pixels'),spt.data.transforms.Resize(224,source='pixels',target='pixels'))
  refs={r['base_id']:r for r in csv.DictReader((ROOT/'data/lewm-reference-scores.csv').open()) if r['condition']=='official_clean'}
  if sha(ROOT/'data/pusht-controlled256-bundle.npz')!='359dae44cdf32f5dcd6917e6624c3be4dcc0b268e9b51b0e6978828605f7fdf7':raise ValueError('Bundle mismatch')
  def score(pixels,actions):
   with torch.inference_mode():
    encoded=model.encode({'pixels':pixels,'action':actions});pred=model.predict(encoded['emb'][:,:3],encoded['act_emb'][:,:3])
    return (pred-encoded['emb'][:,1:]).pow(2).mean(dim=-1)[0].tolist()
  with np.load(ROOT/'data/pusht-controlled256-bundle.npz',allow_pickle=False) as b:
   for i in range(8):
    bid=str(b['base_id'][i]);pix=preprocess({'pixels':torch.from_numpy(b['pixels'][i])})['pixels'].unsqueeze(0).cuda();act=normalize_action_blocks(torch.from_numpy(b['action'][i]),mean,std).unsqueeze(0).cuda()
    first,repeat=score(pix,act),score(pix,act);ref=json.loads(refs[bid]['temporal_prediction_mse']);diff=max(abs(x-y) for x,y in zip(first,ref))
    rows.append({'base_id':bid,'temporal_mse':first,'reference':ref,'repeat':repeat,'max_abs_difference':diff,'pass':len(first)==len(ref) and diff<=1e-5 and first==repeat});save(out/'progress.json',rows)
  result={'status':'pass' if all(r['pass'] for r in rows) else 'parity_failed','rows':rows,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.time()-started,'strict_load_keys':keys,'normalization_sha256':sha(dest),'gpu':torch.cuda.get_device_name(),'scope':'Old archival fixtures only; no new missing cells'}
  save(out/'result.json',result)
  if result['status']!='pass':raise ValueError('LeWM native parity failed; no new cells')
 except BaseException as e:save(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'rows':rows});raise
if __name__=='__main__':main()
