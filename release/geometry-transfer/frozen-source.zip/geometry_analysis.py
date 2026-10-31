"""Frozen dev-only OLS transfer; final labels never fit coefficients."""
import numpy as np
LAYERS=('block_09','block_19','block_29','encoder_final','native_target')
def tables(rows,features):
 assert features.shape==(len(rows),5,1408) and np.isfinite(features).all()
 ids=[r['clip_id'] for r in rows];assert len(set(ids))==len(ids);idx={k:i for i,k in enumerate(ids)};by={(r['scene_id'],r['world'],r['physical_condition'],r['appearance_condition']):r for r in rows};assert len(by)==len(rows)
 tasks={'physical_margin':[],'appearance_score_change':[]}
 def add(task,a,b,base,indicator):
  av=features[idx[a['clip_id']]];bv=features[idx[b['clip_id']]];norm=np.array([np.linalg.norm(v) for v in bv]);assert (norm>1e-12).all();d=np.array([float(np.linalg.norm(av[k]-bv[k])/np.linalg.norm(bv[k])) for k in range(5)])
  tasks[task].append({'scene_id':a['scene_id'],'world':a['world'],'appearance':a['appearance_condition'],'branch':a['physical_condition'],'y':a['score']-b['score'],'base':base+[a['frame_features'][k]-b['frame_features'][k] for k in ('static','motion','shuffled')],'displacement':d.tolist(),'indicator':indicator})
 for sid in sorted({r['scene_id'] for r in rows}):
  for w in ['solid','gap']:
   for app in ['clean','persistent_hue','abrupt_event']:
    add('physical_margin',by[sid,w,'inconsistent',app],by[sid,w,'lawful',app],[float(w=='gap'),float(app=='persistent_hue'),float(app=='abrupt_event')],float(w=='gap'))
   for branch in ['lawful','inconsistent']:
    for app in ['persistent_hue','abrupt_event']:
     a=by[sid,w,branch,app];add('appearance_score_change',a,by[sid,w,branch,'clean'],[float(w=='gap'),float(a['trajectory']=='fall'),float(app=='abrupt_event')],float(app=='abrupt_event'))
 return tasks

def design(table,li,kind,permutation=None):
 base=np.array([r['base'] for r in table],float);d=np.array([r['displacement'][li] for r in table],float);indicator=np.array([r['indicator'] for r in table],float)
 if permutation is not None:
  scenes=sorted({r['scene_id'] for r in table});cells=len(table)//len(scenes);assert all(table[j*cells]['scene_id']==sid for j,sid in enumerate(scenes));d=d.reshape(len(scenes),cells)[permutation].reshape(-1)
 extra=np.c_[d,d*indicator]
 x=base if kind=='baseline' else np.c_[base[:,:3],extra] if kind=='geometry_only' else np.c_[base,extra]
 return np.c_[np.ones(len(table)),x]

def fit(table):
 y=np.array([r['y'] for r in table]);models={}
 for li,layer in enumerate(LAYERS):
  models[layer]={}
  for kind in ['baseline','geometry_only','combined']:
   x=design(table,li,kind);beta,_,rank,singular=np.linalg.lstsq(x,y,rcond=None);models[layer][kind]={'beta':beta.tolist(),'rank':int(rank),'training_min':x.min(axis=0).tolist(),'training_max':x.max(axis=0).tolist()}
 return {'training_y_mean':float(y.mean()),'models':models,'training_scenes':sorted({r['scene_id'] for r in table})}

def evaluate(table,coeff):
 scenes=sorted({r['scene_id'] for r in table});n=len(scenes);cells=len(table)//n;y=np.array([r['y'] for r in table]);den=((y-coeff['training_y_mean'])**2).reshape(n,cells).mean(axis=1);assert den.mean()>0
 rng=np.random.default_rng(2026102801);idx=rng.integers(0,n,size=(50000,n));denboot=den[idx].mean(axis=1);assert (denboot>0).all();result={};predictions={}
 perms=np.array([np.random.default_rng(2026102803+j).permutation(n) for j in range(1000)])
 for li,layer in enumerate(LAYERS):
  result[layer]={};sse={}
  for kind,m in coeff['models'][layer].items():
   x=design(table,li,kind);p=x@np.array(m['beta']);predictions[layer+'/'+kind]=p.tolist();e=(p-y)**2;sse[kind]=e.reshape(n,cells).mean(axis=1);boot=1-sse[kind][idx].mean(axis=1)/denboot
   result[layer][kind]={'mse':float(e.mean()),'R2_training_mean_reference':float(1-e.mean()/den.mean()),'descriptive95':np.quantile(boot,[.025,.975]).tolist(),'out_of_training_range_rows':int(((x<np.array(m['training_min']))|(x>np.array(m['training_max']))).any(axis=1).sum()),'by_world_appearance':{w+'/'+a:float(e[[r['world']==w and r['appearance']==a for r in table]].mean()) for w in ['solid','gap'] for a in sorted({r['appearance'] for r in table})}}
  gains=(sse['baseline']-sse['combined']);boot=gains[idx].mean(axis=1)/denboot;null=[];beta=np.array(coeff['models'][layer]['combined']['beta'])
  for perm in perms:
   p=design(table,li,'combined',perm)@beta;null.append(float(1-np.square(p-y).mean()/den.mean()))
  result[layer]['incremental_R2']={'mean':float(gains.mean()/den.mean()),'descriptive95':np.quantile(boot,[.025,.975]).tolist(),'simultaneous99_5':np.quantile(boot,[.0025,.9975]).tolist(),'family_size':10}
  result[layer]['shuffled_geometry_control']={'permutations':1000,'seed_rule':'2026102803+j; identical permutation across condition/layer/task','R2_mean':float(np.mean(null)),'R2_95_range':np.quantile(null,[.025,.975]).tolist(),'not_causal_or_confirmatory_p_value':True}
 return {'configurations':n,'rows':len(table),'draws':50000,'bootstrap_seed':2026102801,'scope':'post-hoc fixed-development-fit transfer, conditional on trained coefficients','layers':result},predictions
