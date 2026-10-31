"""Development-only EMA-target geometry on the exact already-scored input windows.

Fixed layers/analysis before extraction; association only, not a corrected score,
causal mediation, or confirmatory selection of a favorable layer.
"""
import argparse, datetime, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'alex'))
import a3_a4_lib as lib
import score_development24 as native
LAYERS=('block_09','block_19','block_29','encoder_final','native_target')

def pool_target(tensor,indices):return tensor.index_select(1,indices).detach().float().mean(dim=1).cpu()
def relative(a,b):
    n=np.linalg.norm(b)
    if n<=1e-12:raise ValueError('zero reference-feature norm')
    return float(np.linalg.norm(a-b)/n)
def analyse(rows,features):
    index={r['clip_id']:i for i,r in enumerate(rows)}
    by={(r['scene_id'],r['world'],r['physical_condition'],r['appearance_condition']):r for r in rows}
    scenes=sorted({r['scene_id'] for r in rows});physical=[];appearance=[]
    def vector(r,li):return features[index[r['clip_id']],li]
    for sid in scenes:
        for world in ('solid','gap'):
            for app in ('clean','persistent_hue','abrupt_event'):
                lawful=by[sid,world,'lawful',app];inconsistent=by[sid,world,'inconsistent',app]
                physical.append({'scene_id':sid,'world':world,'appearance':app,
                    'y':inconsistent['score']-lawful['score'],
                    'base':[float(world=='gap'),float(app=='persistent_hue'),float(app=='abrupt_event')]+
                           [inconsistent['frame_features'][k]-lawful['frame_features'][k] for k in ('static','motion','shuffled')],
                    'displacement':[relative(vector(inconsistent,li),vector(lawful,li)) for li in range(len(LAYERS))]})
            for branch in ('lawful','inconsistent'):
                clean=by[sid,world,branch,'clean']
                for app in ('persistent_hue','abrupt_event'):
                    changed=by[sid,world,branch,app]
                    appearance.append({'scene_id':sid,'world':world,'appearance':app,'branch':branch,
                        'y':changed['score']-clean['score'],
                        'base':[float(world=='gap'),float(changed['trajectory']=='fall'),float(app=='abrupt_event')]+
                               [changed['frame_features'][k]-clean['frame_features'][k] for k in ('static','motion','shuffled')],
                        'displacement':[relative(vector(changed,li),vector(clean,li)) for li in range(len(LAYERS))]})
    result={}
    for task,table in [('physical_margin',physical),('appearance_score_change',appearance)]:
        base=np.array([r['base'] for r in table]);y=np.array([r['y'] for r in table]);groups=np.array([r['scene_id'] for r in table])
        result[task]={}
        for li,layer in enumerate(LAYERS):
            d=np.array([r['displacement'][li] for r in table])
            # Geometry-by-world/appearance interaction is specified before extraction.
            indicator=np.array([float(r['world']=='gap') if task=='physical_margin' else float(r['appearance']=='abrupt_event') for r in table])
            b,f,g=lib.nested_gain(base,np.c_[d,d*indicator],y,groups)
            result[task][layer]={'shortcut_only_loso_r2':b,'with_geometry_loso_r2':f,'incremental_r2':g}
    return {'scope':'exploratory same-model association on 24 inspected development configurations',
            'unit':'base configuration, all worlds/branches/appearances held out together','layers':list(LAYERS),
            'analysis':'Alex leave-one-scene-out OLS; no p-values, no best-layer selection','results':result,
            'limitations':['EMA target features share the score target encoder; an association can be mechanical.',
                'Features average over the 36-frame future; not a local event representation or causal intervention.',
                'The balanced margin never substitutes for reporting both within-world physical margins.',
                'No trained appearance removal, random-subspace/null intervention or final-test evidence.']},physical,appearance

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((a.root/'scores/all-scores.json').read_text())
    if len(rows)!=288 or {r['split'] for r in rows}!={'development'}:raise ValueError('unexpected input set')
    audit=json.loads((a.root/'completion-audit-v1/audit.json').read_text())
    if audit['status']!='pass':raise ValueError('audit gate')
    metadata={'status':'started','scope':'development only','started_at':datetime.datetime.now().astimezone().isoformat(),
        'layers':list(LAYERS),'encoder':'EMA target_encoder from the exact native-score checkpoint',
        'pooling':'float32 mean of target-frame tokens 12..47; intermediate/encoder_final unnormalized, native_target per-token layer-normalized',
        'frames':48,'context':12,'resolution':256,'batch':1,'dtype':'fp32 weights; bf16 autocast; fp32 pooling',
        'execution_version':'v2 disk-block target, no GPU-resident full target; v1 startup MemoryError preserved',
        'input_scores_sha256':native.sha(a.root/'scores/all-scores.json'),'audit_sha256':native.sha(a.root/'completion-audit-v1/audit.json'),
        'checkpoint_sha256':native.CHECKPOINT_SHA,'model_revision':native.MODEL_REV,
        'source_sha256':{str(p):native.sha(p) for p in [Path(__file__),HERE/'score_development24.py',HERE/'disk_offload_vjepa2.py',HERE/'stream_vjepa2_checkpoint.py',HERE.parent/'alex/a3_a4_lib.py']},
        'physical_features':'gap/persistent/abrupt indicators + three frame-statistic differences; add relative geometry distance and distance*gap',
        'appearance_features':'gap/fall/abrupt indicators + three frame-statistic differences; add relative geometry distance and distance*abrupt'}
    native.save(out/'protocol-lock.json',metadata)
    scorer=None;handles=[]
    try:
        # Use the already-validated low-VRAM disk-block path to reduce residency.
        # Exact host-allocation failure cause is unproven. No arithmetic changes.
        if native.sha(native.CHECKPOINT)!=native.CHECKPOINT_SHA or native.core.revision(native.MODEL_ROOT)!=native.MODEL_REV:
            raise ValueError('checkpoint/source mismatch')
        scorer=native.Scorer.__new__(native.Scorer)
        scorer.spec=native.core.MODEL_SPECS['vjepa2-g']
        scorer.models=native.disk.DiskOffload(native.CHECKPOINT,native.MODEL_ROOT,scorer.spec)
        scorer.encoder,scorer.target,scorer.predictor=scorer.models.models
        scorer.cache={};scorer.context_predictors={}
        from src.masks.utils import apply_masks
        scorer.apply_masks=apply_masks
        import csv
        reference=next(r for r in csv.DictReader(open(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'contact-development-scores-1/scores.csv'))) if r['physical_condition']=='lawful' and r['appearance_condition']=='clean')
        parity=scorer.score(native.pilot.load_clip(Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'contact-development-1/lawful/clean/frames'))))
        error=max(abs(parity['score']-float(reference['score'])),float(np.max(np.abs(np.array(parity['temporal_tubelet_l1'])-np.array(json.loads(reference['tubelet_scores_json']))))))
        if error>1e-6:raise ValueError('native-score reference gate failed')
        native.save(out/'native-reference-parity.json',{'status':'pass','max_difference':error})
        scorer.cache.clear();scorer.context_predictors.clear()
        enc,pred,full=native.core.token_masks(12,scorer.spec,1);indices=pred[0];captured={}
        def hook(name):
            def capture(_m,_inputs,output):captured[name]=pool_target(output,indices)
            return capture
        for layer in LAYERS[:3]:handles.append(scorer.target.blocks[int(layer.split('_')[1])].register_forward_hook(hook(layer)))
        def infer(norm):
            captured.clear();x=norm.permute(1,0,2,3).unsqueeze(0).contiguous().to('cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                final=scorer.target(x,full)
                captured['encoder_final']=pool_target(final,indices)
                raw=scorer.apply_masks(final,[pred],concat=False)[0]
                captured['native_target']=F.layer_norm(raw,(raw.size(-1),)).detach().float().mean(dim=1).cpu()
            result=np.stack([captured[layer].numpy()[0] for layer in LAYERS])
            if result.shape!=(5,1408) or not np.isfinite(result).all():raise ValueError('feature shape/finiteness')
            return result
        feats=[];index=[];start=time.perf_counter();(out/'clips').mkdir()
        for i,row in enumerate(rows):
            paths=[a.root/p for p in row['source_frame_paths']]
            if [native.sha(p) for p in paths]!=row['source_frame_sha256']:raise ValueError('input PNG mismatch')
            norm=native.core.normalize_video(torch.from_numpy(native.pilot.load_clip(paths[0].parent)),256)
            if native.tsha(norm[:12])!=row['context_sha'] or native.tsha(norm[12:])!=row['target_input_sha256']:raise ValueError('normalized input mismatch')
            feature=infer(norm)
            if i==0:
                repeated=infer(norm)
                if not np.array_equal(feature,repeated):raise ValueError('feature exact-repeat failed')
                native.save(out/'feature-repeat.json',{'status':'pass','max_difference':0.0,'clip_id':row['clip_id']})
            p=out/'clips'/(row['clip_id']+'.npy');np.save(p,feature);feats.append(feature)
            index.append({'row':i,'clip_id':row['clip_id'],'scene_id':row['scene_id'],'context_sha':row['context_sha'],
                'target_input_sha256':row['target_input_sha256'],'feature_sha256':native.sha(p)})
            native.save(out/'progress.json',{'completed_cells':i+1,'total':288,'seconds':time.perf_counter()-start})
        matrix=np.stack(feats);np.save(out/'features.npy',matrix);native.save(out/'feature-index.json',index)
        summary,physical,appearance=analyse(rows,matrix)
        native.save(out/'summary.json',summary);native.save(out/'physical-displacements.json',physical);native.save(out/'appearance-displacements.json',appearance)
        metadata.update(status='complete',completed_at=datetime.datetime.now().astimezone().isoformat(),elapsed_seconds=time.perf_counter()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),feature_sha256=native.sha(out/'features.npy'))
        print(json.dumps(summary),flush=True)
    except BaseException as exc:
        metadata.update(status='failed',error=repr(exc));raise
    finally:
        for h in handles:h.remove()
        if scorer is not None:scorer.models.close()
        native.save(out/'provenance.json',metadata)
if __name__=='__main__':main()
