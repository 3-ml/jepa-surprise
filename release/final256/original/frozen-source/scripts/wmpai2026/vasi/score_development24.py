"""Native batch-one scorer for sealed development24; no labels used for tuning."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'scripts'))
sys.path.insert(0,str(REPO/'scripts/wmpai2026/alex'))
import run_intphys2_dense as core
import run_vjepa2_kubric_construct_validity as pilot
import disk_offload_vjepa2 as disk
import a3_a4_lib as baselines

CHECKPOINT=Path(__import__("os").environ['JEPA_VJEPA2G_CHECKPOINT'])
MODEL_ROOT=Path(__import__("os").environ['JEPA_VJEPA2_SOURCE'])
CHECKPOINT_SHA='67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e'
MODEL_REV='204698b45b3712590f06245fbfba32d3be539812'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def tsha(t):return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def save(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)


class Scorer:
    def __init__(self):
        if sha(CHECKPOINT)!=CHECKPOINT_SHA or core.revision(MODEL_ROOT)!=MODEL_REV:raise ValueError('wrong checkpoint/source')
        self.spec=core.MODEL_SPECS['vjepa2-g']
        self.models=disk.DiskOffload(CHECKPOINT,MODEL_ROOT,self.spec)
        self.encoder,self.target,self.predictor=self.models.models
        # Stage target blocks once directly onto GPU (no simultaneous full CPU model copies).
        off=self.models.offloads[1]
        off.close()
        for block in self.target.blocks:off._stage(block,None)
        self.cache={}
        self.context_predictors={}
        from src.masks.utils import apply_masks
        self.apply_masks=apply_masks

    def score(self,rgb):
        norm=core.normalize_video(torch.from_numpy(rgb.copy()),256)
        ctx=tsha(norm[:12]);target_hash=tsha(norm[12:])
        torch.cuda.reset_peak_memory_stats()
        x=norm.permute(1,0,2,3).unsqueeze(0).contiguous().to('cuda')
        enc,pred,full=core.token_masks(12,self.spec,1)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            if ctx in self.cache:
                prediction=self.cache[ctx]
            else:
                encoded=self.encoder(x,enc)
                prediction=self.predictor(encoded,enc,pred)
                self.cache[ctx]=prediction.detach().clone()
                del encoded
            all_target=self.target(x,full)
            raw=self.apply_masks(all_target,[pred],concat=False)[0]
            target=F.layer_norm(raw,(raw.size(-1),))
            error=(prediction-target).abs().float()
            if not all(bool(torch.isfinite(t).all()) for t in (prediction,target,error)):
                raise RuntimeError('nonfinite model tensor')
            scalar=float(error.mean())
            temporal=error.mean(-1).reshape(1,18,16,16).mean((-1,-2))[0].cpu().tolist()
            prediction_sha=tsha(prediction.contiguous().view(torch.int16))
        if self.context_predictors.setdefault(ctx,prediction_sha)!=prediction_sha:raise ValueError('predictor context invariant')
        torch.cuda.synchronize()
        row={'score':scalar,'temporal_tubelet_l1':temporal,'context_sha':ctx,'target_input_sha256':target_hash,
             'predictor_sha256':prediction_sha,'seconds':time.perf_counter()-start,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        del x,all_target,raw,target,error,prediction
        return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    seal=json.loads((root/'render-seal.json').read_text())
    lock=json.loads((root/'run-lock.json').read_text())
    for name,digest in lock['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('source changed after run lock: '+name)
    output=root/'scores';output.mkdir(exist_ok=False)
    provenance={'status':'started','checkpoint_sha256':CHECKPOINT_SHA,'model_revision':MODEL_REV,
                'render_seal_sha256':sha(root/'render-seal.json'),'torch':torch.__version__,
                'gpu':torch.cuda.get_device_name(),'dtype':'fp32 weights; bf16 autocast/subtraction; fp32 error reduction',
                'resident_target_encoder':True,'context_cache':'normalized context SHA256 only, cleared per world',
                'activation_checkpointing':False,'frames':48,'context':12,'resolution':256,'batch':1,
                'parity_max_abs_tubelet_atol':1e-6,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32':torch.backends.cudnn.allow_tf32}
    save(output/'provenance.json',provenance)
    scorer=None
    try:
        scorer=Scorer()
        # Must match existing single-scene *development* score before revealing any new batch score.
        refroot=Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'contact-development-1'))
        refcsv=Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'contact-development-scores-1/scores.csv'))
        reference=next(r for r in csv.DictReader(refcsv.open()) if r['physical_condition']=='lawful' and r['appearance_condition']=='clean')
        observed=scorer.score(pilot.load_clip(refroot/'lawful/clean/frames'))
        td=np.max(np.abs(np.asarray(observed['temporal_tubelet_l1'])-np.asarray(json.loads(reference['tubelet_scores_json']))))
        delta=abs(observed['score']-float(reference['score']))
        parity={'scalar_abs_difference':delta,'max_abs_tubelet_difference':float(td),'tolerance':1e-6,
                'reference_scores_sha256':sha(refcsv),'status':'pass' if max(delta,td)<=1e-6 else 'fail'}
        save(output/'execution-parity.json',parity)
        if parity['status']!='pass':raise RuntimeError('resident-target parity gate failed; no new batch scores generated')
        print('PARITY PASS',parity,flush=True)
        scorer.cache.clear();scorer.context_predictors.clear()
        rows=[];missing=[];controls=[]
        for candidate in seal['candidates']:
            sid=candidate['scene_id'];scene=root/sid
            if candidate['status']!='eligible':
                missing.append(candidate);continue
            for world in ('solid','gap'):
                data=scene/world
                if sha(data/'manifest.json')!=candidate['worlds'][world]['manifest_sha256']:
                    raise ValueError('render manifest changed')
                manifest=json.loads((data/'manifest.json').read_text());scorer.cache.clear();scorer.context_predictors.clear()
                paired_context={};state_hashes={}
                for clip in manifest['clips']:
                    branch=clip['physical_condition'];ap=clip['appearance_condition']
                    paths=[data/p for p in clip['source_frame_paths']]
                    if [sha(p) for p in paths]!=clip['source_frame_sha256']:raise ValueError('frame hash mismatch')
                    if sha(data/clip['state_trace_path'])!=clip['state_trace_sha256']:raise ValueError('state hash mismatch')
                    if state_hashes.setdefault(branch,clip['state_trace_sha256'])!=clip['state_trace_sha256']:raise ValueError('appearance altered state')
                    rgb=pilot.load_clip(paths[0].parent)
                    row=scorer.score(rgb)
                    if paired_context.setdefault(ap,row['context_sha'])!=row['context_sha']:raise ValueError('paired normalized context mismatch')
                    small=np.stack([np.asarray(Image.fromarray(f).resize((64,64),Image.Resampling.BILINEAR)) for f in rgb])/255.0
                    features=baselines.frame_features(small,12,rng=np.random.default_rng(20261025))
                    stop=(branch=='lawful')==(world=='solid')
                    row.update(scene_id=sid,base_trajectory_id=sid,world=world,physical_condition=branch,
                               appearance_condition=ap,trajectory='stop' if stop else 'fall',
                               split='development',clip_id=f'{sid}__{world}__{branch}__{ap}',
                               state_sha=clip['state_trace_sha256'],source_manifest_sha256=sha(data/'manifest.json'),
                               source_frame_sha256=clip['source_frame_sha256'],source_frame_paths=[str(p.relative_to(root)) for p in paths],
                               min_target_area_fraction=clip['target_area_min_target_fraction'],frame_features=features,
                               output_status='complete')
                    save(output/(row['clip_id']+'.json'),row);rows.append(row)
                    print(json.dumps({k:row[k] for k in ('clip_id','score','seconds')}),flush=True)
                    save(output/'progress.json',{'completed_cells':len(rows),'last_clip':row['clip_id']})
                    # Prespecified control fixture, never additional independent scenes.
                    if sid=='dev24-001' and world=='solid' and branch=='lawful' and ap=='clean':
                        shuffled,permutation=pilot.target_shuffle(rgb,'dev24-control-v1',sid)
                        for name,control in [('hold',pilot.target_hold(rgb)),('shuffle',shuffled)]:
                            controlrow=scorer.score(control)
                            controls.append({'scene_id':sid,'name':name,**controlrow,
                                             'delta_from_clean':controlrow['score']-row['score']})
                        scorer.cache.clear()  # repeat must recompute both encoders, not reuse predictions
                        repeat=scorer.score(rgb)
                        if repeat['score']!=row['score'] or repeat['temporal_tubelet_l1']!=row['temporal_tubelet_l1']:
                            raise RuntimeError('nonidentical zero-intervention repeat')
                        controls.append({'scene_id':sid,'name':'repeat','exact':True,**repeat})
                        save(output/'controls.json',{'source_frame_permutation':permutation,'rows':controls})
                scorer.cache.clear();torch.cuda.empty_cache()
        save(output/'all-scores.json',rows);save(output/'missing-scenes.json',missing)
        provenance.update(status='complete',scored_cells=len(rows),eligible_scenes=len(rows)//12,missing_scenes=len(missing))
    except BaseException as exc:
        provenance.update(status='failed',error=repr(exc));raise
    finally:
        if scorer is not None:scorer.models.close()
        save(output/'provenance.json',provenance)

if __name__=='__main__':main()
