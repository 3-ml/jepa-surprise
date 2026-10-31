"""Batch-one native V-JEPA2-G scores for one score-blind validated development fixture.

No confirmatory interpretation; immutable per-clip writes plus paired-context,
repeat, finite, source/input hash checks. Hold/shuffle controls remain separate.
"""
from __future__ import annotations
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

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import run_intphys2_dense as core
import run_vjepa2_kubric_construct_validity as pilot
import disk_offload_vjepa2 as disk
import vasi_manifest

CHECKPOINT_SHA='67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e'
MODEL_REV='204698b45b3712590f06245fbfba32d3be539812'
PROTOCOL='contact-development-0001-native48-context12-l1-v1'


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


def save(path,data):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--checkpoint',type=Path,default=Path(__import__("os").environ['JEPA_VJEPA2G_CHECKPOINT']))
    parser.add_argument('--model-root',type=Path,default=Path(__import__("os").environ['JEPA_VJEPA2_SOURCE']))
    args=parser.parse_args()
    manifest_path=args.data/'manifest.json'
    if sha(manifest_path)!=args.manifest_sha256:raise ValueError('render manifest hash mismatch')
    manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='pass' or manifest['clip_count']!=6:raise ValueError('render validity did not pass')
    review=json.loads((args.data/'pre-score-review.json').read_text())
    if review['manifest_sha256']!=args.manifest_sha256 or review['decision']!='allow_development_scoring':
        raise ValueError('missing pre-score visual/trace review')
    if core.revision(args.model_root)!=MODEL_REV or sha(args.checkpoint)!=CHECKPOINT_SHA:
        raise ValueError('model identity mismatch')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA bf16 required')
    args.output.mkdir(parents=True,exist_ok=False)
    spec=core.MODEL_SPECS['vjepa2-g']
    inputs={};metadata={};normalized={}
    for clip in manifest['clips']:
        key=(clip['physical_condition'],clip['appearance_condition'])
        paths=[args.data/p for p in clip['source_frame_paths']]
        if len(paths)!=48:raise ValueError('wrong frame count')
        if [sha(p) for p in paths]!=clip['source_frame_sha256']:raise ValueError('source PNG digest mismatch')
        if sha(args.data/clip['state_trace_path'])!=clip['state_trace_sha256']:raise ValueError('state digest mismatch')
        rgb=pilot.load_clip(paths[0].parent)
        inputs[key]=rgb;metadata[key]=clip
        normalized[key]=core.normalize_video(torch.from_numpy(rgb.copy()),256)
    for appearance in ('clean','persistent_hue','abrupt_event'):
        if not torch.equal(normalized['lawful',appearance][:12],normalized['inconsistent',appearance][:12]):
            raise ValueError('postprocessing changed paired context')
    baseline=inputs['lawful','clean']
    shuffled,permutation=pilot.target_shuffle(baseline,'contact-development-v1','dev-contact-0001')
    controls={('control','target_hold'):pilot.target_hold(baseline),('control','target_shuffle'):shuffled,
              ('control','zero_intervention_repeat'):baseline.copy()}
    for key,rgb in controls.items():
        normalized[key]=core.normalize_video(torch.from_numpy(rgb.copy()),256)
    provenance={'status':'started','scope':'single_scene_development_only',
                'script_sha256':sha(__file__),'loader_sha256':sha(Path(disk.__file__)),
                'reader_sha256':sha(Path(__file__).with_name('stream_vjepa2_checkpoint.py')),
                'core_sha256':sha(Path(core.__file__)),'project_revision':core.revision(ROOT),
                'render_manifest_sha256':args.manifest_sha256,'checkpoint_sha256':CHECKPOINT_SHA,
                'model_revision':MODEL_REV,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
                'precision':'FP32 weights; bf16 autocast; original bf16 subtraction then FP32 reduction',
                'score_definition':'mean absolute predictor minus layer-normalized EMA target tokens',
                'model_spec':spec,'context_frames':12,'batch_size':1,'activation_checkpointing':False,
                'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                'shuffle_seed':'contact-development-v1','shuffle_permutation':permutation,
                'zero_repeat_scalar_atol':1e-6,'hardware_memory_free_bytes_before':torch.cuda.mem_get_info()[0]}
    save(args.output/'provenance.json',provenance)
    models=None
    try:
        started=time.perf_counter()
        models=disk.DiskOffload(args.checkpoint,args.model_root,spec)
        encoder,target_encoder,predictor=models.models
        provenance['model_setup_seconds']=time.perf_counter()-started
        provenance['strict_key_shape_dtype_checks']=models.detail
        from src.masks.utils import apply_masks
        rows=[];diagnostics=[];predictor_by_context={}
        for key,norm in normalized.items():
            branch,appearance=key
            clip_id='dev-contact-0001__'+branch+'__'+appearance
            context_sha=tensor_sha(norm[:12]);target_sha=tensor_sha(norm[12:])
            torch.cuda.reset_peak_memory_stats()
            x=norm.permute(1,0,2,3).unsqueeze(0).contiguous().to('cuda')
            enc_mask,pred_mask,full_mask=core.token_masks(12,spec,1)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                all_targets=target_encoder(x,full_mask)
                raw=apply_masks(all_targets,[pred_mask],concat=False)[0]
                targets=F.layer_norm(raw,(raw.size(-1),))
                del raw,all_targets
                encoded=encoder(x,enc_mask)
                predictions=predictor(encoded,enc_mask,pred_mask)
                error=(predictions-targets).abs().float()
                if not all(bool(torch.isfinite(t).all()) for t in (predictions,targets,error)):
                    raise ValueError('nonfinite model output')
                score=float(error.mean())
                temporal=error.mean(-1).reshape(1,18,16,16).mean((-1,-2))[0].cpu().tolist()
                predictor_sha=tensor_sha(predictions.contiguous().view(torch.int16))
            torch.cuda.synchronize()
            previous=predictor_by_context.setdefault(context_sha,predictor_sha)
            if previous!=predictor_sha:raise ValueError('identical context gave different predictor output')
            common={'scene_id':'dev-contact-0001','clip_id':clip_id,'split':'development',
                    'physical_condition':branch,'appearance_condition':appearance,'score':score,
                    'context_sha':context_sha,'context_input_sha256':context_sha,
                    'target_input_sha256':target_sha,'predictor_sha256':predictor_sha,
                    'tubelet_scores_json':json.dumps(temporal),
                    'tubelet_source_frames_json':json.dumps([[i,i+1] for i in range(12,48,2)]),
                    'inference_seconds':time.perf_counter()-start,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'checkpoint_sha256':CHECKPOINT_SHA,'score_protocol_id':PROTOCOL,'output_status':'complete'}
            if branch!='control':
                clip=metadata[key]
                own_clean=inputs[branch,'clean']
                difference=np.abs(inputs[key][12:].astype(np.float32)-own_clean[12:].astype(np.float32))/255
                with np.load(args.data/branch/appearance/'passes.npz') as passes:
                    target_mask=passes['segmentation'][12:,...,0]==1
                row={**common,'base_trajectory_id':'dev-contact-0001-initial-state',
                     'scene_seed':'none-fixed-initial-configuration','appearance_dose':0 if appearance=='clean' else 30,
                     'appearance_dose_unit':'degrees_hue','event_frame_0based':12,
                     'first_physical_position_divergence_frame_ceil':15,
                     'source_frame_indices_json':json.dumps(list(range(48))),
                     'source_frame_paths_json':json.dumps(clip['source_frame_paths']),
                     'source_frame_sha256_json':json.dumps(clip['source_frame_sha256']),
                     'state_trace_sha256':clip['state_trace_sha256'],'state_sha':clip['state_trace_sha256'],
                     'validity_verdict':'score_blind_development_checks_passed','score_blind_exclusion_reason':'',
                     'target_visibility_min_fraction':clip['target_area_min_target_fraction'],
                     'target_area_min_fraction':clip['target_area_min_target_fraction'],
                     'target_rgb_l1':float(difference[target_mask].mean()),
                     'full_frame_rgb_l1_vs_same_branch_clean':float(difference.mean()),
                     'changed_pixel_fraction':float(np.any(difference>0,axis=-1).mean()),
                     'flow_summary_json':json.dumps({'status':'not_computed_in_first_fixture'})}
                rows.append(row)
            else:
                row={**common,'control_source':'lawful/clean'}
                diagnostics.append(row)
            save(args.output/(clip_id+'.json'),row)
            print(json.dumps({'clip':clip_id,'score':score,'seconds':common['inference_seconds']}),flush=True)
            del x,encoded,predictions,targets,error
            torch.cuda.empty_cache()
        errors=vasi_manifest.validate(rows,expected_scenes=['dev-contact-0001'])
        if errors:raise ValueError(errors)
        sys.path.insert(0,str(ROOT/'scripts/wmpai2026/alex'))
        import a2_scene_stats
        if a2_scene_stats.validate(rows):raise ValueError(a2_scene_stats.validate(rows))
        clean=next(r for r in rows if r['physical_condition']=='lawful' and r['appearance_condition']=='clean')
        repeat=next(r for r in diagnostics if r['appearance_condition']=='zero_intervention_repeat')
        if abs(clean['score']-repeat['score'])>1e-6:raise ValueError('zero-intervention score repeat failed')
        if clean['tubelet_scores_json']!=repeat['tubelet_scores_json']:
            raise ValueError('zero-intervention temporal repeat not bit-exact')
        with (args.output/'scores.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        save(args.output/'controls.json',diagnostics)
        summary={'scope':'single_scene_development_only_no_statistical_inference',
                 'scene_count':1,'paired_context_predictor_match':True,'zero_intervention_repeat_exact':True,
                 'margins_inconsistent_minus_lawful':{
                     ap:next(r['score'] for r in rows if r['physical_condition']=='inconsistent' and r['appearance_condition']==ap)
                        -next(r['score'] for r in rows if r['physical_condition']=='lawful' and r['appearance_condition']==ap)
                     for ap in ('clean','persistent_hue','abrupt_event')},
                 'controls_minus_lawful_clean':{r['appearance_condition']:r['score']-clean['score'] for r in diagnostics}}
        save(args.output/'summary.json',summary)
        provenance['status']='complete';provenance['scored_clips']=len(rows)+len(diagnostics)
    except BaseException as exc:
        provenance['status']='failed';provenance['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if models is not None:models.close()
        save(args.output/'provenance.json',provenance)

if __name__=='__main__':main()
