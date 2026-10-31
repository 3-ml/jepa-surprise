"""Post-completion, read-only CPU audit; output ONLY in separate audit root.
No imports of model/scorer/renderer code, no forward, refit, or source mutation.
"""
from pathlib import Path
import datetime,hashlib,json,os,sys,time,traceback,importlib.metadata
import psutil
ROOT=Path(__file__).resolve().parent
FINAL=Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'final256-v1'))
REPO=Path(str(__import__("pathlib").Path(__file__).resolve().parents[3]))
QUEUE=Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'reviewer-physion-desktop-20261026-v2'))
def now():return datetime.datetime.now().astimezone().isoformat()
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,d):
 p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def digest_array(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def trace_hash(p):return hashlib.sha256(json.dumps([[r[k] for k in ['position','quaternion_xyzw','linear_velocity','angular_velocity']] for r in load(p)['trace']],sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 start=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not sys.flags.optimize
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 with (ROOT/'audit-lock.json').open('x') as f:json.dump({'at':now(),'pid':os.getpid(),'create_time':psutil.Process().create_time(),'source_sha256':sha(__file__)},f)
 import numpy as np
 import torch
 from PIL import Image
 from completion_gate import validate_completion
 from independent_statistics import calculate,compare
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 save('environment.json',{'at':now(),'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','Pillow','psutil']},'CUDA_VISIBLE_DEVICES':'','torch_threads':1,'code':{p.name:sha(p) for p in ROOT.glob('*.py')},'base_revision':'44e07fa94a504329646dcfdd8d974a13fe9a13b9'})
 proof=validate_completion(FINAL,REPO,load(QUEUE/'protocol.json')['approved_at']);save('completion-proof.json',proof)
 before=proof['files'];rows=load(FINAL/'scores/all-scores.json');summary=load(FINAL/'analysis/summary.json');gate=load(FINAL/'input-gate.json');inputs={r['clip_id']:r for r in load(FINAL/'input-gate-manifest.json')};scores={r['clip_id']:r for r in rows};assert len(scores)==3072
 # Reconstruct all reported statistics with frequency-weighted bootstrap and tie-group AUC.
 result=calculate(rows);comparison=compare({k:summary[k] for k in result},result)
 save('reconstructed-statistics.json',result);save('statistics-audit.json',{'at':now(),'status':'pass',**comparison,'seed':202610256,'draws':50000,'rate_auc_draws':10000,'resampling_unit':'configuration; all conditions kept together','no_original_analysis_import':True})
 registry=load(FINAL/'inspected-registry.json');contexts=set(registry['normalized_context_sha256']);seen_traces=set(registry['trajectory_sha256']);owner={};paired={};predictors={};states={};unique_files=set();files=0;states_count=0;cells=0;changed=0;maximum=0;area=1.;max_reconstruction=0.;control_proof=[]
 seal=load(FINAL/'render-seal.json');control=load(FINAL/'scores/controls.json')
 def image(path,mode):
  with Image.open(path) as im:return np.array(im.convert(mode))
 def normalized(rgb):
  x=torch.from_numpy(rgb.copy()).to(torch.float32).permute(0,3,1,2)
  assert x.shape==(48,3,256,256)
  x=torch.nn.functional.interpolate(x,size=(256,256),mode='bilinear',align_corners=False)
  return ((x/255.0-torch.tensor([.485,.456,.406]).reshape(1,3,1,1))/torch.tensor([.229,.224,.225]).reshape(1,3,1,1)).contiguous().numpy()
 for candidate in seal['candidates']:
  assert time.monotonic()-start<7200 and psutil.virtual_memory().available>=1024**3
  sid=candidate['scene_id'];assert candidate['status']=='eligible'
  hashes={(w,b):trace_hash(FINAL/sid/w/(b+'-states.json')) for w in ['solid','gap'] for b in ['lawful','inconsistent']}
  assert hashes['solid','lawful']==hashes['gap','inconsistent'] and hashes['solid','inconsistent']==hashes['gap','lawful']
  assert not ({hashes['solid','lawful'],hashes['solid','inconsistent']} & seen_traces)
  seen_traces.update([hashes['solid','lawful'],hashes['solid','inconsistent']]);states_count+=4
  for world in ['solid','gap']:
   data=FINAL/sid/world;mp=data/'manifest.json';mh=sha(mp);assert mh==candidate['worlds'][world]['manifest_sha256'];manifest=load(mp)
   assert len(manifest['clips'])==6
   refs={c['appearance_condition']:c for c in manifest['clips'] if c['physical_condition']=='lawful'}
   ref_pixels={a:np.stack([image(data/p,'RGBA') for p in c['raw_source_frame_paths'][:12]]) for a,c in refs.items()}
   for c in manifest['clips']:
    branch=c['physical_condition'];app=c['appearance_condition'];cid=f'{sid}__{world}__{branch}__{app}';r=scores[cid];assert r==load(FINAL/'scores'/(cid+'.json'))
    assert r['source_manifest_sha256']==mh==inputs[cid]['source_manifest_sha256'] and c['split']==r['split']=='confirmatory'
    pp=[data/p for p in c['source_frame_paths']];rr=[data/p for p in c['raw_source_frame_paths']]
    assert [str(p.relative_to(FINAL)) for p in pp]==r['source_frame_paths'] and c['source_frame_sha256']==r['source_frame_sha256']
    for paths,expected in [(pp,c['source_frame_sha256']),(rr,c['raw_source_frame_sha256'])]:
     assert len(paths)==len(expected)==48
     for p,h in zip(paths,expected):assert sha(p)==h;unique_files.add(str(p));files+=1
    assert c['source_frame_sha256'][12:]==c['raw_source_frame_sha256'][12:]
    assert sha(rr[0].parent.parent/'passes.npz')==c['passes_sha256']
    rawctx=np.stack([image(p,'RGBA') for p in rr[:12]]);ref=ref_pixels['persistent_hue' if app=='persistent_hue' else 'clean'];delta=np.abs(rawctx.astype(np.int16)-ref.astype(np.int16));mx=int(delta.max());assert mx<=1 and delta.mean()/255<=1e-6 and np.any(delta,axis=-1).mean()<=1e-4
    maximum=max(maximum,mx);changed+=int(mx>0)
    rgb=np.stack([image(p,'RGB') for p in pp]);x=normalized(rgb);ctx=digest_array(x[:12]);target=digest_array(x[12:]);assert ctx==inputs[cid]['context_sha']==r['context_sha'] and target==inputs[cid]['target_input_sha256']==r['target_input_sha256']
    assert ctx not in contexts and owner.setdefault(ctx,sid)==sid and paired.setdefault((sid,world,app),ctx)==ctx and predictors.setdefault(ctx,r['predictor_sha256'])==r['predictor_sha256']
    state=sha(data/c['state_trace_path']);assert state==c['state_trace_sha256']==r['state_sha']==inputs[cid]['state_sha'];assert states.setdefault((sid,world,branch),state)==state
    assert r['trajectory']==('stop' if (branch=='lawful')==(world=='solid') else 'fall')
    assert r['target_tubelet_source_frames']==[[12+2*j,13+2*j] for j in range(18)]
    area=min(area,float(c['target_area_min_target_fraction']));assert c['target_area_min_target_fraction']==r['min_target_area_fraction'] and area>=.0025
    err=abs(float(np.mean(r['temporal_tubelet_l1']))-r['score']);max_reconstruction=max(max_reconstruction,err);assert err<=1e-6
    if sid==control['scene_id'] and world=='solid' and branch=='lawful' and app=='clean':
     assert sorted(control['target_tubelet_order'])==list(range(18))
     assert control['original_shuffle_frame_map']==list(range(12))+[12+2*k+j for k in control['target_tubelet_order'] for j in [0,1]]
     assert control['hold_original_frame_map']==list(range(12))+[11]*36
     for cr in control['rows']:
      mapping=control['hold_original_frame_map'] if cr['name']=='hold' else control['original_shuffle_frame_map'] if cr['name']=='shuffle' else list(range(48))
      nx=normalized(rgb[mapping]);assert digest_array(nx[:12])==cr['context_sha']==ctx and digest_array(nx[12:])==cr['target_input_sha256'] and cr['predictor_sha256']==r['predictor_sha256']
      if cr['name']=='repeat':assert cr['exact'] and cr['score']==r['score'] and cr['temporal_tubelet_l1']==r['temporal_tubelet_l1']
      assert abs(np.mean(cr['temporal_tubelet_l1'])-cr['score'])<=1e-6
      control_proof.append({'name':cr['name'],'score':cr['score'],'delta_from_original':cr['score']-r['score'],'input_maps_verified':True})
    cells+=1
   del x,rgb
  save('input-audit-progress.json',{'at':now(),'cells':cells,'PNG_path_checks':files,'last_scene':sid,'elapsed_seconds':time.monotonic()-start})
 assert cells==3072 and files==294912 and changed==gate['nonzero_context_roundoff_cells'] and maximum==gate['max_context_roundoff_uint8']
 assert load(FINAL/'scores/missing-scenes.json')==summary['missing_scenes']==[]
 checkpoint=Path(__import__("os").environ['JEPA_VJEPA2G_CHECKPOINT']);assert checkpoint.stat().st_size==16464208210 and sha(checkpoint)=='67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e'
 for name,h in before.items():assert sha(FINAL/name)==h,'Source mutated during audit: '+name
 assert not torch.cuda.is_initialized()
 save('audit-complete.json',{'at':now(),'status':'pass','eligible_configurations':256,'scored_cells':cells,'missing_scenes':0,'PNG_path_checks':files,'unique_PNG_paths':len(unique_files),'state_traces':states_count,'crossed_fresh_trajectories':512,'context_roundoff_nonzero_cells':changed,'context_roundoff_max_uint8':maximum,'min_target_area_fraction':area,'maximum_scalar_from_tubelets_difference':max_reconstruction,'controls':control_proof,'checkpoint_rehashed':True,'statistics':comparison,'original_sources_and_results_unchanged':True,'CUDA_initialized':False,'new_model_forwards':0,'elapsed_seconds':time.monotonic()-start})
if __name__=='__main__':
 try:main()
 except BaseException as e:save('audit-failure.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc(),'no_model_or_source_changes':True});raise
