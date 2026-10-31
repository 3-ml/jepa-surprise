"""One-shot Desktop CPU export of337 archived-control tensors; no model loading."""
from pathlib import Path
import ctypes,csv,datetime,hashlib,json,os,shutil,time,traceback,zipfile
from canonical_codec import array_sha,sha,save_tensor
ROOT=Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'reviewer-physion-canonical-20261026-v2'))
OLD=Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'reviewer-physion-20261026-v1'))
REPO=Path(str(__import__("pathlib").Path(__file__).resolve().parents[3]))
def now():return datetime.datetime.now().astimezone().isoformat()
def save(n,x):
 p=ROOT/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
class Memory(ctypes.Structure):_fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual']]
def guard(deadline):
 end=min(deadline,time.monotonic()+1800)
 while True:
  m=Memory();m.dwLength=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));disk=shutil.disk_usage(ROOT.anchor).free
  save('resources.json',{'at':now(),'commit_free':m.ullAvailPageFile,'ram_free':m.ullAvailPhys,'D_free':disk,'minimum_GiB':[5,4,50]})
  if time.monotonic()>deadline:raise TimeoutError('two-hour export budget reached')
  if m.ullAvailPageFile>=5*1024**3 and m.ullAvailPhys>=4*1024**3 and disk>=50*1024**3:return
  if time.monotonic()>end:raise TimeoutError('30-minute resource wait; no renderer intervention')
  time.sleep(15)
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';started=time.monotonic();deadline=started+7200
 with (ROOT/'export-lock.json').open('x') as f:json.dump({'at':now(),'pid':os.getpid(),'code_sha256':sha(__file__),'amendment_sha256':sha(ROOT/'amendment.md')},f)
 import psutil
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);guard(deadline)
 import cv2,numpy as np,torch,PIL,transformers
 from transformers import VJEPA2VideoProcessor
 cv2.setNumThreads(1);torch.set_num_threads(1)
 assert cv2.__version__=='4.13.0' and np.__version__=='1.26.4' and PIL.__version__=='12.1.1' and transformers.__version__=='5.13.1'
 processor=VJEPA2VideoProcessor.from_pretrained(OLD/'model-metadata',local_files_only=True)
 save('environment.json',{'at':now(),'python':os.sys.version,'torch':torch.__version__,'opencv':cv2.__version__,'pillow':PIL.__version__,'numpy':np.__version__,'transformers':transformers.__version__,'cuda_initialized':torch.cuda.is_initialized(),'config_sha256':sha(OLD/'model-metadata/config.json'),'processor_config_sha256':sha(OLD/'model-metadata/video_preprocessor_config.json')})
 old=REPO/'results/21-17-30-ten-video-static-surprise-audit'
 def rows(n):
  with (old/n).open(newline='') as f:return list(csv.DictReader(f))
 original=rows('segment_surprise_pixel_summary.csv');selected=rows('static_baseline_candidates.csv');assert len(original)==273 and len(selected)==16
 controls=[{'row':r,'condition':'original','expected_scalar':float(r['scalar_surprise']),'expected_temporal':json.loads(r['temporal_tubelet_surprise'])} for r in original]
 for r in selected:
  for name in ['rescore_original','static_first','static_middle','static_last']:controls.append({'row':r,'condition':'original' if name=='rescore_original' else name,'expected_scalar':float(r[name])})
 with zipfile.ZipFile(OLD/'physion-inputs.zip') as z:source_manifest=json.loads(z.read('bundle-manifest.json'))['entries']
 fixtures=json.loads((OLD/'preprocessing-complete.json').read_text())['records'];expected={r['relative_path']:r for r in fixtures};verified=set();cache=None;frames=None;records=[]
 (ROOT/'controls').mkdir(exist_ok=False)
 for i,control in enumerate(controls):
  guard(deadline);row=control['row'];path=Path(row['path']);relative='data/testdata_v1/'+str(path).replace('\\','/').split('/testdata_v1/',1)[1]
  if cache!=path:
   assert sha(path)==source_manifest[relative]['sha256'];cap=cv2.VideoCapture(str(path));assert cap.isOpened();reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));raw=[]
   try:
    while True:
     ok,bgr=cap.read()
     if not ok:break
     raw.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
   finally:cap.release()
   assert len(raw)==reported;frames=np.stack(raw);del raw;cache=path
  start=int(row['section_source_start_frame_0based']);clip=frames[start:start+16].copy();assert len(clip)==16
  name=control['condition'];mapping=list(range(start,start+16))
  if name!='original':
   k={'static_first':0,'static_middle':8,'static_last':15}[name];clip=np.repeat(clip[k:k+1],16,axis=0);mapping=[start+k]*16
  x=processor([clip],return_tensors='pt')['pixel_values_videos'].numpy();assert x.shape==(1,16,3,256,256) and x.dtype==np.float32 and np.isfinite(x).all()
  if name=='original' and start==0 and relative in expected:
   assert array_sha(clip)==expected[relative]['rgb_sha256'] and array_sha(x)==expected[relative]['input_sha256'];verified.add(relative)
  assert array_sha(torch.from_numpy(x).half().numpy())==array_sha(x.astype(np.float16))
  info=save_tensor(ROOT/f'controls/{i:04d}.npz',x)
  record={**{k:v for k,v in control.items() if k!='row'},'ordinal':i,'relative_video':relative,'source_sha256':source_manifest[relative]['sha256'],'source_start':start,'source_frame_mapping':mapping,'raw_rgb_sha256':array_sha(clip),'file':f'controls/{i:04d}.npz',**info};records.append(record)
  with (ROOT/'control-manifest.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
  save('status.json',{'at':now(),'phase':'exporting_canonical_old_controls','completed':i+1,'planned':337,'exported_bytes':sum(r['bytes'] for r in records),'verified_prior_fixtures':len(verified),'model_scores':0,'cuda_initialized':torch.cuda.is_initialized()})
 assert len(records)==337 and len(verified)==10 and not torch.cuda.is_initialized()
 save('export-complete.json',{'at':now(),'controls':337,'prior_fixture_hashes_exact':10,'manifest_sha256':sha(ROOT/'control-manifest.jsonl'),'elapsed_seconds':time.monotonic()-started,'tensor_file_bytes':sum(r['bytes'] for r in records),'model_scores':0,'cuda_initialized':False})
 members={}
 names=[p.relative_to(ROOT).as_posix() for p in ROOT.glob('*') if p.is_file() and p.suffix in ['.py','.json','.jsonl','.md','.log']]+[r['file'] for r in records]
 with zipfile.ZipFile(ROOT/'canonical-controls.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for n in sorted(names):
   p=ROOT/n;members[n]={'bytes':p.stat().st_size,'sha256':sha(p)};z.write(p,n,compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED)
  z.writestr('archive-manifest.json',json.dumps(members,indent=2))
 with zipfile.ZipFile(ROOT/'canonical-controls.zip') as z:
  for n,v in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
 save('bundle-ready.json',{'at':now(),'bytes':(ROOT/'canonical-controls.zip').stat().st_size,'sha256':sha(ROOT/'canonical-controls.zip'),'members':len(members)+1,'all_member_readbacks_verified':True});save('status.json',{'at':now(),'phase':'canonical_old_controls_exported','controls':337,'model_scores':0})
if __name__=='__main__':
 try:main()
 except BaseException as e:save('export-failure.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});save('status.json',{'at':now(),'phase':'export_failed','error':repr(e)});raise
