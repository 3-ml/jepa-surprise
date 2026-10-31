"""Post-run, read-only development audit. Never excludes, replaces or rescales a cell."""
import argparse, csv, datetime, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

APPEARANCES=('clean','persistent_hue','abrupt_event')
WORLDS=('solid','gap')
BRANCHES=('lawful','inconsistent')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def require(ok,message):
    if not ok:raise ValueError(message)
def check_rows(rows):
    ids=[r['clip_id'] for r in rows];require(len(set(ids))==len(ids),'duplicate clip')
    scenes=sorted({r['scene_id'] for r in rows})
    require(len(scenes)==24 and len(rows)==288,'incomplete grid')
    expected={(s,w,b,a) for s in scenes for w in WORLDS for b in BRANCHES for a in APPEARANCES}
    require({(r['scene_id'],r['world'],r['physical_condition'],r['appearance_condition']) for r in rows}==expected,'wrong grid')
    residual=[]
    for r in rows:
        require(r['split']=='development','wrong split')
        t=np.asarray(r['temporal_tubelet_l1']);require(t.shape==(18,) and np.isfinite(t).all() and np.isfinite(r['score']),'invalid score')
        residual.append(abs(float(t.mean())-r['score']))
    require(max(residual)<1e-7,'scalar/tubelet reduction mismatch')
    return scenes,max(residual)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    root=a.root;rows=load(root/'scores/all-scores.json');scenes,scalar_residual=check_rows(rows)
    lock=load(root/'run-lock.json');seal=load(root/'render-seal.json')
    for name,h in lock['source_sha256'].items():
        require(sha(root/'source'/name)==h,'frozen source mismatch '+name)
        require(sha(a.repo/'scripts/wmpai2026/vasi'/name)==h,'live source mismatch '+name)
    require(sha(root/'configs.json')==lock['config_sha256'],'config changed')
    require(sha(root/'protocol.md')==lock['protocol_sha256'],'protocol changed')
    require(set(scenes)=={c['scene_id'] for c in load(root/'configs.json')['scenes']},'scene selection changed')
    provenance=load(root/'scores/provenance.json')
    require(provenance['status']=='complete' and provenance['scored_cells']==288,'incomplete provenance')
    require(provenance['render_seal_sha256']==sha(root/'render-seal.json'),'seal changed')
    checkpoint=Path(__import__("os").environ['JEPA_VJEPA2G_CHECKPOINT'])
    require(sha(checkpoint)==provenance['checkpoint_sha256'],'checkpoint hash mismatch')
    sys.path.insert(0,str(a.repo/'scripts'))
    import torch
    import run_intphys2_dense as core
    torch.set_num_threads(2)
    require(core.revision(Path(__import__("os").environ['JEPA_VJEPA2_SOURCE']))==provenance['model_revision'],'model source revision mismatch')
    lookup={r['clip_id']:r for r in rows};checks=[];canonical=[];world_checks=[]
    contexts={};predictors={};states={};all_files={}
    for candidate in seal['candidates']:
        sid=candidate['scene_id'];require(candidate['status']=='eligible','unexpected exclusion')
        trajectories={}
        for world in WORLDS:
            data=root/sid/world;require(sha(data/'manifest.json')==candidate['worlds'][world]['manifest_sha256'],'manifest hash mismatch')
            manifest=load(data/'manifest.json');stateaudit=load(data/'state-audit.json')
            require(stateaudit['status']=='pass' and stateaudit['lawful_repeat_exact'],'state audit failed')
            for branch in BRANCHES:
                trajectories[world,branch]=load(data/(branch+'-states.json'))['trace']
            raw_reference={}
            for c in manifest['clips']:
                if c['physical_condition']=='lawful':raw_reference[c['appearance_condition']]=c
            for clip in manifest['clips']:
                branch=clip['physical_condition'];app=clip['appearance_condition'];cid=f'{sid}__{world}__{branch}__{app}'
                row=lookup[cid];require(load(root/'scores'/(cid+'.json'))==row,'individual/all-scores mismatch')
                require(row['source_manifest_sha256']==sha(data/'manifest.json'),'row manifest mismatch')
                require(sha(data/clip['state_trace_path'])==row['state_sha']==clip['state_trace_sha256'],'state hash mismatch')
                require(states.setdefault((sid,world,branch),row['state_sha'])==row['state_sha'],'appearance changed state')
                paths=[data/p for p in clip['source_frame_paths']];raw=[data/p for p in clip['raw_source_frame_paths']]
                require(len(paths)==48 and len(raw)==48,'frame count')
                for frame_paths,hashes in [(paths,clip['source_frame_sha256']),(raw,clip['raw_source_frame_sha256'])]:
                    for p,h in zip(frame_paths,hashes):
                        require(sha(p)==h,'PNG changed '+str(p));all_files[str(p.relative_to(root))]=h
                require(clip['source_frame_sha256'][12:]==clip['raw_source_frame_sha256'][12:],'future changed during canonicalization')
                require(row['source_frame_sha256']==clip['source_frame_sha256'],'scored PNG hashes mismatch')
                require([root/p for p in row['source_frame_paths']]==paths,'scored path order mismatch')
                ref=raw_reference['persistent_hue' if app=='persistent_hue' else 'clean']
                raw_context=np.stack([np.asarray(Image.open(p).convert('RGBA')) for p in raw[:12]])
                ref_context=np.stack([np.asarray(Image.open(data/p).convert('RGBA')) for p in ref['raw_source_frame_paths'][:12]])
                delta=np.abs(raw_context.astype('int16')-ref_context.astype('int16'))
                maximum=int(delta.max());mean=float(delta.mean()/255);fraction=float(np.any(delta,axis=-1).mean())
                require(maximum<=1 and mean<=1e-6 and fraction<=1e-4,'canonical bound exceeded')
                canonical.append({'clip_id':cid,'max_uint8':maximum,'normalized_mean_abs':mean,'changed_pixel_fraction':fraction})
                rgb=np.stack([np.asarray(Image.open(p).convert('RGB')) for p in paths])
                norm=core.normalize_video(torch.from_numpy(rgb),256)
                context=hashlib.sha256(norm[:12].contiguous().numpy().tobytes()).hexdigest()
                target=hashlib.sha256(norm[12:].contiguous().numpy().tobytes()).hexdigest()
                require(context==row['context_sha'] and target==row['target_input_sha256'],'normalized input mismatch')
                key=(sid,world,app)
                require(contexts.setdefault(key,context)==context,'paired context mismatch')
                require(predictors.setdefault(context,row['predictor_sha256'])==row['predictor_sha256'],'same-context predictor mismatch')
                require(row['min_target_area_fraction']>=0.0025,'visibility area threshold')
                checks.append({'clip_id':cid,'input_hashes_verified':True,'future_bytes_unchanged':True,'state_sha':row['state_sha']})
        fields=('position','quaternion_xyzw','linear_velocity','angular_velocity')
        for aa,bb in [(('solid','lawful'),('gap','inconsistent')),(('solid','inconsistent'),('gap','lawful'))]:
            t1,t2=trajectories[aa],trajectories[bb]
            require(len(t1)==len(t2) and all(x[k]==y[k] for x,y in zip(t1,t2) for k in fields),'cross-world mismatch')
        world_checks.append({'scene_id':sid,'crossed_trajectories_exact':True})
        print('audited',sid,flush=True)
    control_file=load(root/'scores/controls.json');controls=control_file['rows']
    reference=lookup['dev24-001__solid__lawful__clean']
    repeat=next(c for c in controls if c['name']=='repeat')
    require(repeat['exact'] and repeat['score']==reference['score'] and repeat['temporal_tubelet_l1']==reference['temporal_tubelet_l1'],'repeat failed')
    order=control_file['source_frame_permutation']
    require(sorted(order)==list(range(18)),'bad target-tubelet permutation')
    maps={'hold':list(range(12))+[11]*36,'shuffle':list(range(12))+[12+2*k+j for k in order for j in (0,1)],'repeat':list(range(48))}
    rgb=np.stack([np.asarray(Image.open(root/p).convert('RGB')) for p in reference['source_frame_paths']])
    for c in controls:
        norm=core.normalize_video(torch.from_numpy(rgb[maps[c['name']]].copy()),256)
        require(hashlib.sha256(norm[:12].contiguous().numpy().tobytes()).hexdigest()==c['context_sha']==reference['context_sha'],'control context hash mismatch')
        require(hashlib.sha256(norm[12:].contiguous().numpy().tobytes()).hexdigest()==c['target_input_sha256'],'control target hash mismatch')
        require(c['predictor_sha256']==reference['predictor_sha256'],'control predictor mismatch')
        if 'delta_from_clean' in c:require(c['delta_from_clean']==c['score']-reference['score'],'control delta mismatch')
    save(a.out/'control-original-source-frame-map.json',{'note':'Original source_frame_permutation field is an 18-element TARGET TUBELET order, not a 48-frame permutation. Original file preserved.',
         'target_tubelet_order':order,'original_source_frame_maps':maps,'input_hashes_recomputed':True})
    parity=load(root/'scores/execution-parity.json');require(parity['status']=='pass','reference failed')
    summary=load(root/'analysis/summary.json');independent={}
    for app in APPEARANCES:
        v={(r['scene_id'],r['world'],r['physical_condition']):r['score'] for r in rows if r['appearance_condition']==app}
        margins={world:np.array([v[s,world,'inconsistent']-v[s,world,'lawful'] for s in scenes]) for world in WORLDS}
        margins['geometry_balanced']=(margins['solid']+margins['gap'])/2
        independent[app]={}
        for label,arr in margins.items():
            reported=summary['per_appearance'][app][label+'_margin']
            require(abs(arr.mean()-reported['mean'])<1e-14,'summary mean mismatch')
            require(float((arr>0).mean())==reported['frac_pos'],'summary sign mismatch')
            independent[app][label]={'mean':float(arr.mean()),'positive_count':int((arr>0).sum()),'n':len(arr)}
    report={'status':'pass','scope':'post-score audit, no exclusions/replacements','completed_at':datetime.datetime.now().astimezone().isoformat(),
        'scenes':len(scenes),'cells':len(checks),'unique_png_paths_hashed':len(all_files),'checkpoint_sha256_verified':provenance['checkpoint_sha256'],
        'control_inputs_and_exact_repeat_verified':True,'max_scalar_vs_tubelet_mean_difference':scalar_residual,'min_target_area_fraction':min(r['min_target_area_fraction'] for r in rows),
        'canonicalized_cells_with_nonzero_roundoff':sum(c['max_uint8']>0 for c in canonical),
        'max_raw_context_uint8_difference':max(c['max_uint8'] for c in canonical),
        'max_raw_context_normalized_mean_abs':max(c['normalized_mean_abs'] for c in canonical),
        'max_raw_context_changed_pixel_fraction':max(c['changed_pixel_fraction'] for c in canonical),
        'score_inference_seconds_total':sum(r['seconds'] for r in rows),'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows),
        'independent_recomputed_margins':independent,'source_sha256':sha(Path(__file__)),
        'limitations':['Does not rerun GPU scoring or independently reconstruct physics in a second simulator.',
        'All-positive/all-negative empirical-bootstrap proportion intervals are degenerate; do not interpret as population certainty.',
        'Shared procedural template; no natural-population or equivalence claim.']}
    save(a.out/'audit.json',report);save(a.out/'clip-checks.json',checks);save(a.out/'crossed-state-checks.json',world_checks)
    save(a.out/'context-roundoff.json',canonical)
    with (a.out/'external-png-sha256.json').open('w') as f:json.dump(all_files,f,sort_keys=True)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
