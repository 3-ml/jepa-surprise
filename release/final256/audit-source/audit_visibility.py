"""Read-only reconstruction of recorded visibility and renderer-pose checks.
No new eligibility rule, simulation or inference. Same1e-6 pose/.0025 area bounds.
"""
from pathlib import Path
import numpy as np
import os,json,datetime,time,traceback,psutil
ROOT=Path(__file__).resolve().parent;F=Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'final256-v1'))
def load(p):return json.loads(Path(p).read_text())
def save(n,d):(ROOT/n).write_text(json.dumps(d,indent=2,allow_nan=False))
def now():return datetime.datetime.now().astimezone().isoformat()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);start=time.monotonic();count=0;minimum=65536;maxpose=0.
 with (ROOT/'visibility-audit-lock.json').open('x') as f:json.dump({'at':now(),'pid':os.getpid(),'create_time':psutil.Process().create_time()},f)
 for c in load(F/'render-seal.json')['candidates']:
  for world in ['solid','gap']:
   d=F/c['scene_id']/world;m=load(d/'manifest.json');states={b:load(d/(b+'-states.json'))['trace'] for b in ['lawful','inconsistent']}
   for clip in m['clips']:
    raw=d/clip['raw_source_frame_paths'][0];path=raw.parent.parent/'passes.npz'
    with np.load(path,allow_pickle=False) as z:
     seg=z['segmentation'];depth=z['depth'];assert seg.shape==depth.shape==(48,256,256,1) and np.isfinite(seg).all() and np.isfinite(depth).all();area=(seg[...,0]==1).sum(axis=(1,2));assert area.tolist()==clip['target_area_pixels_per_frame'];assert float(area[12:].min())/65536==clip['target_area_min_target_fraction']>=.0025;minimum=min(minimum,int(area[12:].min()))
    expected=np.array([states[clip['physical_condition']][frame*20]['position'] for frame in range(48)]);observed=np.array(clip['rendered_target_positions']);assert expected.shape==observed.shape==(48,3);delta=float(np.abs(expected-observed).max());assert delta<=1e-6;maxpose=max(maxpose,delta);count+=1
  save('visibility-audit-progress.json',{'at':now(),'clips':count,'last_scene':c['scene_id'],'elapsed_seconds':time.monotonic()-start});assert time.monotonic()-start<7200
 assert count==3072
 save('visibility-audit-complete.json',{'at':now(),'status':'pass','passes_arrays':count,'segmentation_and_depth_finite':True,'target_area_counts_reconstructed':True,'minimum_future_target_pixels':minimum,'minimum_future_target_fraction':minimum/65536,'maximum_renderer_pose_discrepancy':maxpose,'pose_tolerance':1e-6,'new_renderer_or_model_calls':0,'elapsed_seconds':time.monotonic()-start})
if __name__=='__main__':
 try:main()
 except BaseException as e:save('visibility-audit-failure.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
