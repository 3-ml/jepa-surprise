"""Read-only supplemental configuration/source/attempt audit; no rendering."""
from pathlib import Path
import hashlib,json,datetime,random,sys
ROOT=Path(__file__).resolve().parent;F=Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'final256-v1'));R=Path(str(__import__("pathlib").Path(__file__).resolve().parents[3]))
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 cfgs=load(F/'configs.json')['scenes'];freeze=load(F/'freeze.json');seal=load(F/'render-seal.json');assert len(cfgs)==256
 palette=sorted({tuple(c['base_color_rgb']) for c in cfgs});assert len(palette)==4
 # This checks internal adherence to the declared draw order/palette, not an
 # independent recovery of unavailable historical random state.
 sources={}
 for n,h in freeze['source_sha256'].items():assert sha(R/n)==h;sources[n]=h
 for p in (F/'renderer-source').glob('*.py'):
  candidates=[h for n,h in freeze['source_sha256'].items() if Path(n).name==p.name]
  assert len(candidates)==1 and sha(p)==candidates[0]
 checks=[]
 for i,c in enumerate(cfgs):
  seed=2026102700+i;assert c['scene_seed']==seed and c['scene_id']==f'final256-{i+1:04d}' and c['split']=='confirmatory'
  rng=random.Random(seed);radius=rng.uniform(.60,.70);contact=rng.uniform(1.18,1.24);camera=rng.choice([-8,-4,4,8]);x=rng.uniform(-.035,.035);colour=list(rng.choice(palette))
  assert c['sphere_radius']==radius and c['nominal_contact_time']==contact and c['camera_position'][0]==camera and c['sphere_initial_position']==[x,-.3,radius+.5*9.81*contact**2] and c['base_color_rgb']==colour
  sid=c['scene_id'];candidate=next(v for v in seal['candidates'] if v['scene_id']==sid)
  for w in ['solid','gap']:
   expected=dict(c,world=w);d=F/sid/w;manifest=load(d/'manifest.json');assert manifest['config']==load(d/'config.json')==load(F/sid/(w+'-config.json'))==expected
   assert manifest['status']=='pass' and not manifest['issues'] and manifest['clip_count']==6
   assert manifest['script_sha256']==freeze['source_sha256']['scripts/wmpai2026/vasi/final256_render_worker.py']
   attempt=load(F/sid/(w+'-attempt.json'));assert attempt['status']=='pass' and attempt.get('exit_code') in [None,0] and attempt['manifest_sha256']==sha(d/'manifest.json')==candidate['worlds'][w]['manifest_sha256']
   audit=load(d/'state-audit.json');assert audit['lawful_repeat_exact'] and audit['missing_stopping_contact_rejected'] and audit['different_context_state_rejected']
   checks.append({'scene_id':sid,'world':w,'configuration_sha256':sha(d/'config.json'),'state_audit_sha256':sha(d/'state-audit.json'),'attempt_sha256':sha(F/sid/(w+'-attempt.json'))})
 out={'at':datetime.datetime.now().astimezone().isoformat(),'status':'pass','configurations':256,'worlds':512,'fixed_seed_draw_order_matches':True,'palette_scope':'four colours frozen in configs; no re-selection','sources_rehashed':sources,'checks':checks,'no_renderer_or_model_import':True}
 (ROOT/'configuration-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['checks','sources_rehashed']}))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback
  (ROOT/'configuration-audit-failure.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2));raise
