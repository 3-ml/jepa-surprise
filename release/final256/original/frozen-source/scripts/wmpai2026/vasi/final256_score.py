"""Frozen native final256 scoring; disk-block execution, atomic resume, no interim summaries."""
import argparse, csv, datetime, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import score_development24 as native
from final256_common import APPEARANCES, WORLDS, BRANCHES, load, save, sha, check_sources
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]

def make_scorer():
    if sha(native.CHECKPOINT)!=native.CHECKPOINT_SHA or native.core.revision(native.MODEL_ROOT)!=native.MODEL_REV:raise ValueError('checkpoint/model source mismatch')
    scorer=native.Scorer.__new__(native.Scorer);scorer.spec=native.core.MODEL_SPECS['vjepa2-g']
    scorer.models=native.disk.DiskOffload(native.CHECKPOINT,native.MODEL_ROOT,scorer.spec)
    scorer.encoder,scorer.target,scorer.predictor=scorer.models.models
    scorer.cache={};scorer.context_predictors={}
    from src.masks.utils import apply_masks
    scorer.apply_masks=apply_masks
    return scorer

def reference_gate(scorer):
    ref=next(r for r in csv.DictReader(open(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'contact-development-scores-1/scores.csv'))) if r['physical_condition']=='lawful' and r['appearance_condition']=='clean')
    rgb=native.pilot.load_clip(Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'contact-development-1/lawful/clean/frames')))
    a=scorer.score(rgb);scorer.cache.clear();b=scorer.score(rgb)
    difference=max(abs(a['score']-float(ref['score'])),float(np.max(np.abs(np.asarray(a['temporal_tubelet_l1'])-np.asarray(json.loads(ref['tubelet_scores_json']))))))
    if difference>1e-6 or a['score']!=b['score'] or a['temporal_tubelet_l1']!=b['temporal_tubelet_l1']:raise ValueError('old-reference parity/repeat failed')
    scorer.cache.clear();scorer.context_predictors.clear()
    return {'status':'pass','max_reference_difference':difference,'reference_tolerance':1e-6,'uncached_repeat_exact':True,'old_development_fixture_only':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path);ap.add_argument('--preflight-out',type=Path);a=ap.parse_args()
    if a.preflight_out:
        if a.preflight_out.exists():raise ValueError('preflight output exists')
        scorer=make_scorer()
        try:save(a.preflight_out,reference_gate(scorer))
        finally:scorer.models.close()
        return
    root=a.root;lock=load(root/'run-lock.json');check_sources(REPO,lock)
    seal=load(root/'render-seal.json');excluded=load(root/'inspected-registry.json')
    gate=load(root/'input-gate.json')
    if gate['status']!='pass' or gate['render_seal_sha256']!=sha(root/'render-seal.json') or gate['freeze_sha256']!=sha(root/'freeze.json') or gate['input_manifest_sha256']!=sha(root/'input-gate-manifest.json'):raise ValueError('whole-set input gate failed')
    input_rows={r['clip_id']:r for r in load(root/'input-gate-manifest.json')}
    if not seal['sealed_before_scoring'] or seal['eligible_count']<231:raise ValueError('score-blind eligibility gate')
    output=root/'scores';output.mkdir(exist_ok=True)
    if (output/'provenance.json').exists():
        old=load(output/'provenance.json')
        if old.get('status')=='complete':return
        attempts=output/'prior-attempts';attempts.mkdir(exist_ok=True)
        save(attempts/(str(time.time_ns())+'.json'),old)
    provenance={'status':'started','started_at':datetime.datetime.now().astimezone().isoformat(),
        'scope':'frozen final256; earlier joint statistics draft not jointly signed','checkpoint_sha256':native.CHECKPOINT_SHA,
        'model_revision':native.MODEL_REV,'render_seal_sha256':sha(root/'render-seal.json'),'run_lock_sha256':sha(root/'run-lock.json'),
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'frames':48,'context':12,'resolution':256,'batch':1,
        'dtype':'fp32 weights; bf16 autocast/subtraction; fp32 L1 reduction','target_encoder_residency':'disk-block',
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32}
    save(output/'provenance.json',provenance);scorer=None
    try:
        scorer=make_scorer();save(output/'execution-parity.json',reference_gate(scorer))
        rows=[];contexts={};predictors={};state_hashes={};old_contexts=set(excluded['normalized_context_sha256'])
        owner={};eligible=[c for c in seal['candidates'] if c['status']=='eligible'];control_sid=eligible[0]['scene_id']
        for candidate in eligible:
            sid=candidate['scene_id']
            for world in WORLDS:
                data=root/sid/world;manifest=load(data/'manifest.json');scorer.cache.clear();scorer.context_predictors.clear()
                if sha(data/'manifest.json')!=candidate['worlds'][world]['manifest_sha256']:raise ValueError('render manifest changed')
                for clip in manifest['clips']:
                    branch=clip['physical_condition'];app=clip['appearance_condition'];cid=f'{sid}__{world}__{branch}__{app}'
                    if clip['split']!='confirmatory' or clip['scene_id']!=sid:raise ValueError('wrong rendered split/id')
                    paths=[data/p for p in clip['source_frame_paths']]
                    if [sha(p) for p in paths]!=clip['source_frame_sha256']:raise ValueError('PNG changed')
                    if sha(data/clip['state_trace_path'])!=clip['state_trace_sha256']:raise ValueError('state changed')
                    rgb=native.pilot.load_clip(paths[0].parent);norm=native.core.normalize_video(torch.from_numpy(rgb.copy()),256)
                    ctx=native.tsha(norm[:12]);target=native.tsha(norm[12:]);del norm
                    if input_rows[cid]['context_sha']!=ctx or input_rows[cid]['target_input_sha256']!=target or input_rows[cid]['state_sha']!=clip['state_trace_sha256']:raise ValueError('pre-score input gate mismatch')
                    if ctx in old_contexts:raise ValueError('normalized context overlaps inspected data')
                    if owner.setdefault(ctx,sid)!=sid:raise ValueError('normalized context duplicated across final scenes')
                    file=output/(cid+'.json')
                    if file.exists():
                        row=load(file)
                        if row['context_sha']!=ctx or row['target_input_sha256']!=target or row['source_manifest_sha256']!=sha(data/'manifest.json') or row['clip_id']!=cid or row['output_status']!='complete':raise ValueError('resume input/row mismatch')
                    else:
                        row=scorer.score(rgb)
                        small=np.stack([np.asarray(Image.fromarray(f).resize((64,64),Image.Resampling.BILINEAR)) for f in rgb])/255.
                        features=native.baselines.frame_features(small,12,rng=np.random.default_rng(20261025))
                        stop=(branch=='lawful')==(world=='solid')
                        row.update(scene_id=sid,base_trajectory_id=sid,world=world,physical_condition=branch,appearance_condition=app,
                            trajectory='stop' if stop else 'fall',split='confirmatory',clip_id=cid,state_sha=clip['state_trace_sha256'],
                            source_manifest_sha256=sha(data/'manifest.json'),source_frame_sha256=clip['source_frame_sha256'],
                            source_frame_paths=[str(p.relative_to(root)) for p in paths],min_target_area_fraction=clip['target_area_min_target_fraction'],
                            target_tubelet_source_frames=[[12+2*j,13+2*j] for j in range(18)],
                            frame_features=features,output_status='complete')
                        save(file,row)
                    if row['context_sha']!=ctx or row['target_input_sha256']!=target:raise ValueError('scorer input mismatch')
                    if contexts.setdefault((sid,world,app),ctx)!=ctx:raise ValueError('paired context mismatch')
                    if predictors.setdefault(ctx,row['predictor_sha256'])!=row['predictor_sha256']:raise ValueError('predictor mismatch')
                    if state_hashes.setdefault((sid,world,branch),row['state_sha'])!=row['state_sha']:raise ValueError('appearance state mismatch')
                    if not np.isfinite(row['score']) or len(row['temporal_tubelet_l1'])!=18 or not np.isfinite(row['temporal_tubelet_l1']).all():raise ValueError('invalid saved score')
                    rows.append(row);save(output/'progress.json',{'completed_cells':len(rows),'total_cells':len(eligible)*12,'last_clip':cid,'at':datetime.datetime.now().astimezone().isoformat()})
                    if sid==control_sid and world=='solid' and branch=='lawful' and app=='clean' and not (output/'controls.json').exists():
                        shuffled,order=native.pilot.target_shuffle(rgb,'final256-control-v1',sid);controls=[]
                        for name,arr in [('hold',native.pilot.target_hold(rgb)),('shuffle',shuffled)]:
                            controls.append({'name':name,**scorer.score(arr)})
                        scorer.cache.clear();repeat=scorer.score(rgb)
                        if repeat['score']!=row['score'] or repeat['temporal_tubelet_l1']!=row['temporal_tubelet_l1']:raise ValueError('final control repeat mismatch')
                        controls.append({'name':'repeat','exact':True,**repeat})
                        save(output/'controls.json',{'scene_id':sid,'target_tubelet_order':order,'original_shuffle_frame_map':list(range(12))+[12+2*k+j for k in order for j in (0,1)],'hold_original_frame_map':list(range(12))+[11]*36,'rows':controls})
                scorer.cache.clear();torch.cuda.empty_cache()
        save(output/'all-scores.json',rows);save(output/'missing-scenes.json',[c for c in seal['candidates'] if c['status']!='eligible'])
        provenance.update(status='complete',completed_at=datetime.datetime.now().astimezone().isoformat(),scored_cells=len(rows),eligible_scenes=len(eligible),all_scores_sha256=sha(output/'all-scores.json'))
    except BaseException as exc:provenance.update(status='failed',error=repr(exc));raise
    finally:
        if scorer is not None:scorer.models.close()
        save(output/'provenance.json',provenance)
if __name__=='__main__':main()
