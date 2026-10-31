"""Score-blind full-cohort construction. All gates precede new model outcomes."""
import csv,os,sys,traceback,json
import numpy as np
import torch
from matched_common import *
os.environ.update(SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',XFORMERS_DISABLED='1')
sys.path.insert(0,str(ROOT/'code'))

def main():
 verify();dest=OUT/'inputs';dest.mkdir(exist_ok=False)
 try:
  torch.set_num_threads(4)
  from torchvision.transforms import functional as tvf
  from scripts.run_jepa_wm_pusht_controlled_displacement import simulate_target
  from scripts.run_lewm_pusht_role_color_audit import render_sequence,INTERVENTIONS
  with np.load(ROOT/'data/pusht-controlled256-bundle.npz',allow_pickle=False) as z:b={k:z[k] for k in z.files}
  if len(b['base_id'])!=256 or len(set(b['episode']))!=256:raise ValueError('Expected 256 distinct original episodes')
  refs={m:{(r['base_id'],r['condition']):r for r in csv.DictReader((ROOT/f'data/{m}-reference-scores.csv').open())} for m in MODELS}
  interventions={x.name:x for x in INTERVENTIONS};manifest=[]
  for i,bid in enumerate(b['base_id']):
   bid=str(bid);pixels=torch.from_numpy(b['pixels'][i]);actions=torch.from_numpy(b['action'][i]);states=torch.from_numpy(b['state'][i]);block=actions[2].reshape(5,2)
   if not torch.isfinite(actions).all() or not torch.isfinite(states).all():raise ValueError('Invalid original states/actions')
   cells={'official_clean':(pixels.clone(),actions.clone(),states.clone())};start_errors=[];endpoint_error=None
   for scale in [0.,.25,.75,1.]:
    start,finish,frame=simulate_target(states[2],block*scale);start_errors.append(float((start-states[2]).abs().max()))
    if scale==1.:
     endpoint_error=float((finish-states[3]).abs().max())
     if endpoint_error>2e-4:raise ValueError('Original official-endpoint state gate failed')
     continue
    pix=pixels.clone();pix[-1]=frame;st=states.clone();st[-1]=finish;act=actions.clone()
    if scale==0.:act[2]=block.reshape(-1)*0.;name='lawful_scale_0p0'
    else:act[2]=(-block*scale).reshape(-1);name='opposite_scale_'+str(scale).replace('.','p')
    cells[name]=(pix,act,st)
   if max(start_errors)>2e-5:raise ValueError('Original reset-start state gate failed')
   static,zero,static_states=cells['lawful_scale_0p0']
   for dose in [30,60]:
    persistent=torch.stack([tvf.adjust_hue(f,dose/360.) for f in static]);target=static.clone();target[-1]=persistent[-1];act=zero.clone();act[2]=torch.zeros_like(act[2])
    cells[f'appearance_hue_{dose:03d}']=(target,act,static_states.clone());cells[f'persistent_hue_{dose:03d}']=(persistent,act.clone(),static_states.clone())
   default,actual=render_sequence(states,None);cells['rerender_default']=(default,actions.clone(),actual)
   for role in ROLES:
    variant,realized=render_sequence(states,interventions[role])
    if not torch.equal(realized,actual):raise ValueError('Palette changed realized simulator state')
    target=default.clone();target[-1]=variant[-1]
    cells[role+'_persistent']=(variant,actions.clone(),actual.clone());cells[role+'_target']=(target,actions.clone(),actual.clone())
   if set(cells)!=set(CONDITIONS):raise ValueError('Wrong condition grid')
   rows=[]
   for name in CONDITIONS:
    pix,act,st=cells[name];arr=pix.numpy();target_hash=ahash(arr[-1])
    if tuple(pix.shape)!=(4,3,224,224) or pix.dtype!=torch.uint8:raise ValueError('Invalid image grid')
    if name in OLD:
     for m in MODELS:
      if target_hash!=refs[m][bid,name]['target_sha256']:raise ValueError('Archived target hash mismatch '+m+' '+bid+' '+name)
    parent=PARENTS.get(name);row={'condition':name,'raw_context_sha256':ahash(arr[:3]),'target_sha256':target_hash,'actions_sha256':ahash(act.numpy()),'states_sha256':ahash(st.numpy()),'parent':parent}
    if parent:
     base=cells[parent]
     if name!='rerender_default' and (not torch.equal(act,base[1]) or not torch.equal(st,base[2])):raise ValueError('Appearance changed actions or realized states')
     row.update(target_rgb_l1=float((pix[-1].double()-base[0][-1].double()).abs().mean()/255),context_rgb_l1=float((pix[:3].double()-base[0][:3].double()).abs().mean()/255))
     target_only=name.endswith('_target') or name.startswith('appearance_hue_')
     if target_only and not all(torch.equal(v[:3],w[:3]) for v,w in zip(cells[name],base)):raise ValueError('Target-only context/action/state changed')
    rows.append(row)
   for persistent in ['persistent_hue_030','persistent_hue_060']+[r+'_persistent' for r in ROLES]:
    target=persistent.replace('persistent_hue_','appearance_hue_') if persistent.startswith('persistent_hue_') else persistent.replace('_persistent','_target')
    if not torch.equal(cells[persistent][0][-1],cells[target][0][-1]):raise ValueError('Scopes have unequal targets')
   path=dest/f'{i:04d}.npz';np.savez_compressed(path,pixels=np.stack([cells[n][0].numpy() for n in CONDITIONS]),actions=np.stack([cells[n][1].numpy() for n in CONDITIONS]),states=np.stack([cells[n][2].numpy() for n in CONDITIONS]))
   manifest.append({'base_id':bid,'episode':int(b['episode'][i]),'start_step':int(b['start_step'][i]),'path':str(path),'sha256':sha(path),'reset_start_max_error':max(start_errors),'official_endpoint_max_error':endpoint_error,'role_replay_vs_stored_state_max_error':float((actual-states).abs().max()),'rows':rows})
   save(OUT/'input-progress.json',{'completed':i+1,'planned':256,'at':now()});print('INPUT',i+1,flush=True)
  save(OUT/'input-seal.json',{'status':'pass','base_configurations':256,'cells':256*len(CONDITIONS),'conditions':CONDITIONS,'manifest':manifest,'freeze_sha256':sha(OUT/'freeze.json'),'at':now()})
 except BaseException as e:save(OUT/'input-failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'at':now()});raise
if __name__=='__main__':main()
