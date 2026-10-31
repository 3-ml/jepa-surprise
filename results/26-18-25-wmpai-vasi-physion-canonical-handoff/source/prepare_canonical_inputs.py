"""CPU-only transfer/decoding gate for canonical tensor controls; no model construction."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,os,traceback,zipfile
from canonical_codec import sha,load_tensor
ROOT=Path.home()/'jepa-wmpai/reviewer-physion-canonical-20261026-v2'
OLD=Path.home()/'jepa-wmpai/reviewer-physion-20261026-v1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,x):(ROOT/n).write_text(json.dumps(x,indent=2)+'\n')
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 with (ROOT/'input-gate-lock.json').open('x') as f:json.dump({'at':now(),'pid':os.getpid()},f)
 expected=json.loads((ROOT/'desktop-bundle-ready.json').read_text());p=ROOT/'canonical-controls.zip';assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256']
 out=ROOT/'desktop-export';out.mkdir(exist_ok=False)
 with zipfile.ZipFile(p) as z:
  m=json.loads(z.read('archive-manifest.json'));assert set(z.namelist())==set(m)|{'archive-manifest.json'}
  for n,r in m.items():
   target=out/n;assert target.resolve().is_relative_to(out.resolve());b=z.read(n);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
  (out/'archive-manifest.json').write_bytes(z.read('archive-manifest.json'))
 assert sha(ROOT/'canonical_codec.py')==json.loads((out/'source-lock.json').read_text())['files']['canonical_codec.py']
 rows=[json.loads(s) for s in (out/'control-manifest.jsonl').read_text().splitlines()];assert len(rows)==337
 def csvrows(name):
  with (OLD/'reference'/name).open(newline='') as f:return list(csv.DictReader(f))
 old=csvrows('segment_surprise_pixel_summary.csv');selected=csvrows('static_baseline_candidates.csv');assert len(old)==273 and len(selected)==16
 controls=[{'row':r,'condition':'original','scalar':float(r['scalar_surprise']),'temporal':json.loads(r['temporal_tubelet_surprise'])} for r in old]
 for r in selected:
  for c in ['rescore_original','static_first','static_middle','static_last']:controls.append({'row':r,'condition':'original' if c=='rescore_original' else c,'scalar':float(r[c])})
 fixtures={r['relative_path']:r for r in json.loads((OLD/'preprocessing-complete.json').read_text())['records']};verified=set()
 for i,(r,c) in enumerate(zip(rows,controls)):
  original=c['row'];relative='data/testdata_v1/'+original['path'].replace('\\','/').split('/testdata_v1/',1)[1]
  assert r['ordinal']==i and r['file']==f'controls/{i:04d}.npz' and r['relative_video']==relative and r['condition']==c['condition'] and r['source_start']==int(original['section_source_start_frame_0based']) and r['expected_scalar']==c['scalar']
  if 'temporal' in c:assert r['expected_temporal']==c['temporal']
  a=load_tensor(out/r['file'],r);assert a.shape==(1,16,3,256,256)
  start=r['source_start'];mapping=list(range(start,start+16)) if r['condition']=='original' else [start+{'static_first':0,'static_middle':8,'static_last':15}[r['condition']]]*16;assert r['source_frame_mapping']==mapping
  if r['condition']=='original' and start==0 and relative in fixtures:
   f=fixtures[relative];assert r['tensor_sha256']==f['input_sha256'] and r['raw_rgb_sha256']==f['rgb_sha256'] and r['source_sha256']==f['source_sha256'];verified.add(relative)
 assert len(verified)==10
 from canonical_score import verify_adapter
 proof=verify_adapter();save('adapter-proof.json',proof)
 import torch
 assert not torch.cuda.is_initialized()
 save('input-seal.json',{'at':now(),'status':'pass','controls':337,'ten_prior_Desktop_fixtures_exact':True,'all_FP32_FP16_context_future_hashes_exact':True,'manifest_sha256':sha(out/'control-manifest.jsonl'),'archive_sha256':sha(p),'adapter_proof':proof,'model_constructed':False,'cuda_initialized':False})
 files=['canonical_codec.py','canonical_score.py','run_canonical_qualification.py','supervise_qualification.py','amendment.md','input-seal.json','desktop-export/control-manifest.jsonl','adapter-proof.json']
 save('qualification-freeze.json',{'at':now(),'files':{n:sha(ROOT/n) for n in files},'old_source_sha256':sha(OLD/'source/run_vjepa2_section_surprise.py'),'old_model_config_sha256':sha(OLD/'model/config.json'),'old_checkpoint_sha256':'25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b','tolerance':1e-4,'stop_first_failure':True,'new_cells_permitted_in_this_job':False})
 print('All337 canonical tensors and ten prior fixtures verified; adapter body AST exact; no CUDA/model construction.',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:save('input-gate-failure.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
