"""Post-hoc final256 target features only; no new final256 native scores."""
import sys,time,traceback
from study_common import *
def main():
 wait_permit('geometry');started=time.monotonic();sys.path.insert(0,str(REPO/'scripts/wmpai2026/vasi'))
 import score_development24 as native
 import geometry_development24 as original
 import numpy as np,torch
 import torch.nn.functional as F
 from geometry_analysis import LAYERS,tables
 from desktop_runtime import resources,sufficient
 assert sufficient(resources());assert native.sha(native.CHECKPOINT)==native.CHECKPOINT_SHA and native.core.revision(native.MODEL_ROOT)==native.MODEL_REV
 out=ROOT/'geometry';out.mkdir(exist_ok=False);root=BASE/'final256-v1';dev=BASE/'development24-v2';rows=load(root/'scores/all-scores.json');assert len(rows)==3072 and len({r['scene_id'] for r in rows})==256;assert load(root/'status.json')['phase']=='complete'
 assert sha(root/'scores/all-scores.json')=='055700d438bf765d381c9a142bdddaf2624f519bc2850428f9109e38ca309bde'
 scorer=native.Scorer.__new__(native.Scorer);scorer.spec=native.core.MODEL_SPECS['vjepa2-g'];scorer.models=native.disk.DiskOffload(native.CHECKPOINT,native.MODEL_ROOT,scorer.spec);scorer.encoder,scorer.target,scorer.predictor=scorer.models.models;scorer.cache={};scorer.context_predictors={}
 from src.masks.utils import apply_masks
 scorer.apply_masks=apply_masks;handles=[]
 try:
  import csv
  reference=next(r for r in csv.DictReader(open(BASE/'contact-development-scores-1/scores.csv')) if r['physical_condition']=='lawful' and r['appearance_condition']=='clean')
  clip=native.pilot.load_clip(BASE/'contact-development-1/lawful/clean/frames');parity=scorer.score(clip);scorer.cache.clear();scorer.context_predictors.clear();repeat=scorer.score(clip)
  error=max(abs(parity['score']-float(reference['score'])),float(np.max(np.abs(np.array(parity['temporal_tubelet_l1'])-json.loads(reference['tubelet_scores_json'])))))
  exact=parity['score']==repeat['score'] and parity['temporal_tubelet_l1']==repeat['temporal_tubelet_l1'] and parity['predictor_sha256']==repeat['predictor_sha256'];save(out/'native-reference-parity.json',{'at':now(),'status':'pass' if error<=1e-6 and exact else 'fail','max_difference':error,'uncached_repeat_exact':exact,'observed':parity,'repeat':repeat,'reference':reference});assert error<=1e-6 and exact;scorer.cache.clear();scorer.context_predictors.clear()
  enc,pred,full=native.core.token_masks(12,scorer.spec,1);indices=pred[0];captured={}
  def hook(name):
   def capture(_m,_inputs,output):captured[name]=original.pool_target(output,indices)
   return capture
  for layer in LAYERS[:3]:handles.append(scorer.target.blocks[int(layer.split('_')[1])].register_forward_hook(hook(layer)))
  def infer(norm):
   captured.clear();x=norm.permute(1,0,2,3).unsqueeze(0).contiguous().to('cuda')
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    final=scorer.target(x,full);captured['encoder_final']=original.pool_target(final,indices);raw=scorer.apply_masks(final,[pred],concat=False)[0];captured['native_target']=F.layer_norm(raw,(raw.size(-1),)).detach().float().mean(dim=1).cpu()
   result=np.stack([captured[layer].numpy()[0] for layer in LAYERS]);assert result.shape==(5,1408) and np.isfinite(result).all();return result
  def normalized(folder,row):
   paths=[folder/p for p in row['source_frame_paths']];assert [native.sha(p) for p in paths]==row['source_frame_sha256'];norm=native.core.normalize_video(torch.from_numpy(native.pilot.load_clip(paths[0].parent)),256);assert native.tsha(norm[:12])==row['context_sha'] and native.tsha(norm[12:])==row['target_input_sha256'];return norm
  firstdev=load(dev/'scores/all-scores.json')[0];norm=normalized(dev,firstdev);feature=infer(norm);again=infer(norm);expected=np.load(dev/'geometry-development-v2/clips'/(firstdev['clip_id']+'.npy'),allow_pickle=False);np.savez(out/'development-reference.npz',observed=feature,repeat=again,expected=expected);assert feature.tobytes()==expected.tobytes()==again.tobytes(),'Exact old-feature parity failed'
  save(out/'feature-parity.json',{'at':now(),'status':'pass','old_feature_exact':True,'repeat_exact':True,'layers':list(LAYERS)});(out/'clips').mkdir();features=[];index=[]
  for i,row in enumerate(rows):
   deadline()
   if i%8==0:inference_guard()
   feature=infer(normalized(root,row))
   if i==0:
    again=infer(normalized(root,row));np.savez(out/'first-final-repeat.npz',observed=feature,repeat=again);assert feature.tobytes()==again.tobytes(),'First final feature repeat failed'
   p=out/'clips'/(row['clip_id']+'.npy');np.save(p,feature,allow_pickle=False);features.append(feature);index.append({'row':i,'clip_id':row['clip_id'],'scene_id':row['scene_id'],'context_sha':row['context_sha'],'target_input_sha256':row['target_input_sha256'],'feature_sha256':sha(p)})
   save(ROOT/'geometry-progress.json',{'at':now(),'done':i+1,'total':3072,'elapsed_seconds':time.monotonic()-started})
  matrix=np.stack(features);np.save(out/'features.npy',matrix,allow_pickle=False);save(out/'feature-index.json',index);table=tables(rows,matrix)
  for name,t in table.items():save(out/(name+'-table.json'),t)
  save(ROOT/'geometry-complete.json',{'at':now(),'status':'pass','clips':3072,'configurations':256,'features_sha256':sha(out/'features.npy'),'feature_index_sha256':sha(out/'feature-index.json'),'tables':{name:sha(out/(name+'-table.json')) for name in table},'coefficient_sha256':sha(ROOT/'geometry-coefficients.json'),'elapsed_seconds':time.monotonic()-started,'new_final_native_scores':0,'scope':'post-hoc representation transfer, not original final256 analysis'})
 finally:
  for h in handles:h.remove()
  scorer.models.close()
if __name__=='__main__':
 try:main()
 except BaseException as e:save(ROOT/'geometry-failure.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
