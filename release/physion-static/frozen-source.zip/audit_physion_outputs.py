"""CPU-only native token/vector reconstruction and fixed pair-level summaries."""
import os,sys,traceback
from study_common import *
def main(study):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';wait_permit('audit'+study);complete=load(ROOT/(study+'-complete.json'));assert complete['status']=='pass' and sha(ROOT/(study+'-rows.jsonl'))==complete['rows_sha256']
 import numpy as np,torch
 import torch.nn.functional as F
 from canonical_codec import array_sha
 from physion_statistics import bootstrap_pairs,hybrid_l1,rank_correlation
 torch.set_num_threads(4);rows=lines(ROOT/(study+'-rows.jsonl'));audit=lines(ROOT/'input-audit.jsonl');pop=load(ROOT/'population.json');assert len(pop)==len(audit)==192
 out=ROOT/('analysis-'+study);out.mkdir(exist_ok=False);video_rows=[];hybrids=[];maxerr=0.;indexed={}
 manifest=lines(ROOT/(study+'-input-manifest.jsonl'));assert len(rows)==len(manifest)
 for r,m in zip(rows,manifest):
  for k in ['ordinal','video_id','pair_id','scenario','layout_group','segment','start','condition','source_frame_mapping','raw_rgb_sha256','tensor_sha256','context_sha256','future_sha256']:assert r[k]==m[k],k
  key=(r['ordinal'],r['segment'],r['condition']);assert key not in indexed;indexed[key]=r
 for i,(entry,record) in enumerate(zip(pop,audit)):
  deadline();v={'ordinal':i,'video_id':entry['video_id'],'pair_id':entry['pair_id'],'scenario':entry['scenario'],'layout_group':entry['layout_group']};operands={};scores={}
  conditions=['original','persistent_hue','target_only_hue','persistent_brightness','target_only_brightness'] if study=='B' else ['original','static_first','static_middle','static_last'];segments=[record['middle_segment']] if study=='B' else range(record['segments'])
  for j in segments:
   for c in conditions:
    r=indexed[i,j,c];path=ROOT/r['artifact'];assert sha(path)==r['artifact_sha256']
    if study=='B':
     with np.load(path,allow_pickle=False) as z:
      p=z['p'];t=z['t'];loss=z['loss'];assert p.shape==t.shape==(512,1024) and np.isfinite(p).all() and np.isfinite(t).all();pn=F.layer_norm(torch.from_numpy(p).float(),(1024,)).numpy().astype(np.float64);tn=F.layer_norm(torch.from_numpy(t).float(),(1024,)).numpy().astype(np.float64)
     token=np.abs(pn-tn).mean(axis=-1);assert np.max(np.abs(token-loss.astype(float)))<=1e-6;operands[c]=(pn,tn);maxerr=max(maxerr,abs(float(token.mean())-r['scalar_surprise']));assert maxerr<=1e-6
    else:loss=np.load(path,allow_pickle=False)
    assert loss.shape==(512,) and loss.dtype==np.float32 and np.isfinite(loss).all() and array_sha(loss)==r['loss_sha256'];err=abs(float(loss.astype(float).mean())-r['scalar_surprise']);assert err<=1e-6;maxerr=max(maxerr,err);assert np.max(np.abs(loss.astype(float).reshape(2,256).mean(axis=1)-r['temporal_tubelet_surprise']))<=1e-6;scores[j,c]=r['scalar_surprise']
  if study=='A':
   x=np.array([[scores[j,c] for c in conditions] for j in segments]);delta=x[:,0]-x[:,1:].mean(axis=1);motions=np.array([w['mean_consecutive_rgb'] for w in record['windows']]);low=motions<.01
   v.update(mean_original_minus_mean_static=float(delta.mean()),fraction_nonpositive_residual=float((delta<=0).mean()),raw_motion_rank_correlation=rank_correlation(x[:,0],motions),near_static_segments=int(low.sum()),near_static_mean_residual=None if not low.any() else float(delta[low].mean()),segments=record['segments'],raw_mean=float(x[:,0].mean()),static_first_mean=float(x[:,1].mean()),static_middle_mean=float(x[:,2].mean()),static_last_mean=float(x[:,3].mean()))
  else:
   j=record['middle_segment'];p0,t0=operands['original']
   for c in conditions[1:]:
    delta=scores[j,c]-scores[j,'original'];v[c+'_minus_original']=delta;h=hybrid_l1(p0,t0,*operands[c]);assert abs(h['native_delta_fp64']-delta)<=2e-6;hybrids.append({'ordinal':i,'pair_id':entry['pair_id'],'condition':c,**h})
   for kind in ['hue','brightness']:
    a=indexed[i,j,'persistent_'+kind];b=indexed[i,j,'target_only_'+kind];assert a['future_sha256']==b['future_sha256'];assert b['context_sha256']==indexed[i,j,'original']['context_sha256'];v['persistent_minus_target_only_'+kind]=scores[j,'persistent_'+kind]-scores[j,'target_only_'+kind];v['same_future_target_context_rms_'+kind]=float(np.sqrt(np.square(operands['persistent_'+kind][1]-operands['target_only_'+kind][1]).mean()))
  video_rows.append(v);save(out/'progress.json',{'at':now(),'videos_audited':i+1,'total':192})
 expected=960 if study=='B' else 20708;assert len(rows)==expected
 columns=['mean_original_minus_mean_static','fraction_nonpositive_residual'] if study=='A' else [c+'_minus_original' for c in conditions[1:]]+['persistent_minus_target_only_hue','persistent_minus_target_only_brightness'];pairs=[]
 for pid in sorted({r['pair_id'] for r in video_rows}):
  m=[r for r in video_rows if r['pair_id']==pid];assert len(m)==2 and len({r['layout_group'] for r in m})==1;pairs.append({'pair_id':pid,'scenario':m[0]['scenario'],'layout_group':m[0]['layout_group'],**{c:float(np.mean([r[c] for r in m])) for c in columns}})
 assert len(pairs)==96
 seed=2026102601 if study=='A' else 2026102602;family=1 if study=='A' else 6;save(out/'video-metrics.json',video_rows);save(out/'pair-metrics.json',pairs)
 if study=='B':save(out/'hybrid-decomposition.json',hybrids)
 result={'at':now(),'study':study,'videos':192,'pairs':96,'cells':expected,'columns':columns,'seed':seed,'family':family,'bootstrap':bootstrap_pairs(pairs,columns,seed,family=family),'layout_sensitivity':bootstrap_pairs(pairs,columns,seed+100,layout=True,family=family),'native_reconstruction_max_abs_error':maxerr,'scope':'inspected synthetic Mass diagnostic; not violation classification or mediation','scenario_means':{s:{c:float(np.mean([r[c] for r in pairs if r['scenario']==s])) for c in columns} for s in sorted({r['scenario'] for r in pairs})},'cuda_initialized':torch.cuda.is_initialized()};assert not result['cuda_initialized'];save(out/'summary.json',result);save(out/'complete.json',{'at':now(),'status':'pass','summary_sha256':sha(out/'summary.json'),'pair_metrics_sha256':sha(out/'pair-metrics.json'),'rows_sha256':complete['rows_sha256']})
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException as e:save(ROOT/('audit'+sys.argv[1]+'-failure.json'),{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
