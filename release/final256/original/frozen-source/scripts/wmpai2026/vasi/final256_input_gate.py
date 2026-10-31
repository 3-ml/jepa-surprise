"""Whole-set input/freshness gate before any final score is computed."""
import argparse, hashlib, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from final256_common import load, save, sha, check_sources
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'scripts'))
import run_intphys2_dense as core

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    check_sources(REPO,load(root/'run-lock.json'));seal=load(root/'render-seal.json')
    old=set(load(root/'inspected-registry.json')['normalized_context_sha256']);owner={};pairs={};state={};rows=[]
    torch.set_num_threads(2);files=0;maximum=0;changed_cells=0
    for candidate in seal['candidates']:
        if candidate['status']!='eligible':continue
        sid=candidate['scene_id']
        for world in ('solid','gap'):
            data=root/sid/world;manifest=load(data/'manifest.json')
            if sha(data/'manifest.json')!=candidate['worlds'][world]['manifest_sha256']:raise ValueError('manifest changed')
            reference={c['appearance_condition']:c for c in manifest['clips'] if c['physical_condition']=='lawful'}
            for c in manifest['clips']:
                if c['split']!='confirmatory' or c['scene_id']!=sid:raise ValueError('wrong split/id')
                branch=c['physical_condition'];app=c['appearance_condition'];cid=f'{sid}__{world}__{branch}__{app}'
                paths=[data/p for p in c['source_frame_paths']];raw=[data/p for p in c['raw_source_frame_paths']]
                for pp,hh in ((paths,c['source_frame_sha256']),(raw,c['raw_source_frame_sha256'])):
                    if len(pp)!=48 or len(hh)!=48 or [sha(p) for p in pp]!=hh:raise ValueError('source PNG/hash mismatch')
                    files+=48
                if c['source_frame_sha256'][12:]!=c['raw_source_frame_sha256'][12:]:raise ValueError('future bytes changed')
                ref=reference['persistent_hue' if app=='persistent_hue' else 'clean']
                x=np.stack([np.asarray(Image.open(p).convert('RGBA')) for p in raw[:12]])
                y=np.stack([np.asarray(Image.open(data/p).convert('RGBA')) for p in ref['raw_source_frame_paths'][:12]])
                delta=np.abs(x.astype('int16')-y.astype('int16'));mx=int(delta.max())
                if mx>1 or delta.mean()/255>1e-6 or np.any(delta,axis=-1).mean()>1e-4:raise ValueError('roundoff bound failed')
                maximum=max(maximum,mx);changed_cells+=int(mx>0)
                rgb=np.stack([np.asarray(Image.open(p).convert('RGB')) for p in paths]);norm=core.normalize_video(torch.from_numpy(rgb),256)
                ctx=hashlib.sha256(norm[:12].contiguous().numpy().tobytes()).hexdigest();target=hashlib.sha256(norm[12:].contiguous().numpy().tobytes()).hexdigest()
                if ctx in old or owner.setdefault(ctx,sid)!=sid:raise ValueError('whole-set input freshness collision; stop before scores')
                if pairs.setdefault((sid,world,app),ctx)!=ctx:raise ValueError('paired context mismatch')
                actual=sha(data/c['state_trace_path'])
                if actual!=c['state_trace_sha256'] or state.setdefault((sid,world,branch),actual)!=actual:raise ValueError('state/appearance mismatch')
                if c['target_area_min_target_fraction']<.0025:raise ValueError('visibility failed')
                rows.append({'clip_id':cid,'context_sha':ctx,'target_input_sha256':target,'state_sha':actual,'source_manifest_sha256':sha(data/'manifest.json')})
        save(root/'input-gate-progress.json',{'completed_cells':len(rows),'expected_cells':seal['eligible_count']*12,'last_scene':sid})
    if len(rows)!=seal['eligible_count']*12:raise ValueError('cell count mismatch')
    save(root/'input-gate-manifest.json',rows)
    save(root/'input-gate.json',{'status':'pass','cells':len(rows),'hashed_png_paths':files,'max_context_roundoff_uint8':maximum,
        'nonzero_context_roundoff_cells':changed_cells,'all_future_PNGs_unchanged':True,
        'all_inputs_checked_before_any_final_score':True,'input_manifest_sha256':sha(root/'input-gate-manifest.json'),
        'render_seal_sha256':sha(root/'render-seal.json'),'freeze_sha256':sha(root/'freeze.json')})
if __name__=='__main__':main()
