"""Qualified Desktop-only acquisition: B960 then A20708, no scientific retries."""
import time,traceback
from study_common import *
def main():
 wait_permit('physion');started=time.monotonic();seal=load(ROOT/'input-seal.json');assert seal['status']=='pass' and seal['A_cells']==20708 and seal['B_cells']==960
 for n,h in seal['files'].items():assert sha(ROOT/n)==h
 import numpy as np,torch
 from canonical_codec import load_tensor as old_load,array_sha
 from tensor_store import load_tensor
 from canonical_score import Scorer
 from desktop_runtime import resources,sufficient
 # Audit every stored input before constructing a model. No partially sealed acquisition.
 for study,total in [('B',960),('A',20708)]:
  records=lines(ROOT/(study+'-input-manifest.jsonl'));assert len(records)==total
  for i,r in enumerate(records):assert sha(ROOT/r['file'])==r['file_sha256'];deadline()
  save(ROOT/(study+'-pre-model-files.json'),{'at':now(),'status':'pass','files':total,'manifest_sha256':sha(ROOT/(study+'-input-manifest.jsonl'))})
 assert sufficient(resources());scorer=Scorer();save(ROOT/'model-runtime.json',{'at':now(),'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'adapter':scorer.proof,'parameters':sum(p.numel() for p in scorer.model.parameters()),'dtype':str(next(scorer.model.parameters()).dtype),'attention':scorer.model.config._attn_implementation,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32})
 out=ROOT/'qualification';out.mkdir(exist_ok=False);old=lines(INPUTS/'control-manifest.jsonl');maximum=0.;original_maps={}
 for i,r in enumerate(old):
  if i%16==0:inference_guard()
  x=old_load(INPUTS/r['file'],r);res,arrays=scorer.score(x,r,features=i==0);np.save(out/f'{i:04d}-loss.npy',arrays['loss'],allow_pickle=False)
  if i==0:np.savez_compressed(out/'first-vectors.npz',**arrays)
  error=abs(res['scalar_surprise']-r['expected_scalar'])
  if 'expected_temporal' in r:error=max(error,float(np.max(np.abs(np.array(res['temporal_tubelet_surprise'])-r['expected_temporal']))))
  maximum=max(maximum,error);append(out/'observations.jsonl',{'index':i,'reference':r['file'],'error':error,'pass':error<=1e-4,'result':res});assert error<=1e-4,'Historical model parity failed; no new cells'
  key=(r['relative_video'],r['source_start'])
  if i<273:original_maps[key]=arrays['loss'].copy()
  elif r['condition']=='original':assert arrays['loss'].tobytes()==original_maps[key].tobytes()
  if i==0:
   res2,again=scorer.score(x,r,features=True);np.savez_compressed(out/'repeat-vectors.npz',**again);assert res2==res and all(arrays[k].tobytes()==again[k].tobytes() for k in arrays)
  save(ROOT/'physion-progress.json',{'at':now(),'phase':'old_controls','done':i+1,'maximum_error':maximum})
 assert len(old)==337;save(out/'complete.json',{'at':now(),'status':'pass','controls':337,'maximum_error':maximum,'tolerance':1e-4,'repeat_exact':True})
 (ROOT/'acquisition').mkdir(exist_ok=False)
 for study,total in [('B',960),('A',20708)]:
  records=lines(ROOT/(study+'-input-manifest.jsonl'));first=True
  for i,r in enumerate(records):
   deadline()
   if i%16==0:inference_guard()
   x=load_tensor(ROOT/r['file'],r);res,arrays=scorer.score(x,r,features=study=='B');folder=ROOT/'acquisition'/f"{r['ordinal']:03d}";folder.mkdir(exist_ok=True)
   name=f"B-{r['condition']}-vectors.npz" if study=='B' else f"A-{r['segment']:03d}-{r['condition']}-loss.npy";path=folder/name
   if study=='B':np.savez_compressed(path,**arrays)
   else:np.save(path,arrays['loss'],allow_pickle=False)
   if first:
    res2,again=scorer.score(x,r,features=study=='B');np.savez_compressed(ROOT/(study+'-first-repeat.npz'),**again);assert res2==res and all(arrays[k].tobytes()==again[k].tobytes() for k in arrays);first=False
   if study=='A' and r['condition']=='original':
    audit=lines(ROOT/'input-audit.jsonl')[r['ordinal']]
    if r['segment']==audit['middle_segment']:
     with np.load(folder/'B-original-vectors.npz',allow_pickle=False) as z:assert arrays['loss'].tobytes()==z['loss'].tobytes(),'A/B original repeat mismatch'
   row={k:r[k] for k in ['ordinal','video_id','pair_id','scenario','layout_group','segment','start','condition','source_frame_mapping','raw_rgb_sha256','tensor_sha256','context_sha256','future_sha256']};row.update(res);row['context_sha256']=r['context_sha256'];row['future_sha256']=r['future_sha256'];row['artifact']=str(path.relative_to(ROOT));row['artifact_sha256']=sha(path);append(ROOT/(study+'-rows.jsonl'),row)
   if i%8==0 or i+1==total:save(ROOT/'physion-progress.json',{'at':now(),'phase':study,'done':i+1,'total':total,'elapsed_seconds':time.monotonic()-started})
  save(ROOT/(study+'-complete.json'),{'at':now(),'status':'pass','cells':total,'videos':192,'pairs':96,'rows_sha256':sha(ROOT/(study+'-rows.jsonl')),'input_manifest_sha256':sha(ROOT/(study+'-input-manifest.jsonl')),'first_repeat_exact':True,'elapsed_seconds':time.monotonic()-started,'freeze_sha256':sha(ROOT/'freeze.json')})
 scorer.pre.remove();scorer.post.remove();del scorer;torch.cuda.empty_cache();save(ROOT/'physion-complete.json',{'at':now(),'status':'pass','B_cells':960,'A_cells':20708,'old_checks':337,'elapsed_seconds':time.monotonic()-started})
if __name__=='__main__':
 try:main()
 except BaseException as e:save(ROOT/'physion-failure.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
