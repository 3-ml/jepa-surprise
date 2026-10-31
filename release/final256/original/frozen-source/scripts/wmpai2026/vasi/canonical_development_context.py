"""Score-blind v2 renderer-roundoff handling; original frames are never overwritten.

Only identical-state contexts with <=1 uint8 level difference, normalized mean
absolute difference <=1e-6, and changed-pixel fraction <=1e-4 may be canonicalized.
All 36 future frames must remain byte-identical to their own raw render.
"""
import copy
import hashlib
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def roundoff_check(actual,canonical):
    if actual.shape!=canonical.shape or actual.dtype!=np.uint8 or canonical.dtype!=np.uint8:
        raise ValueError('context shape/type mismatch')
    difference=np.abs(actual.astype(np.int16)-canonical.astype(np.int16))
    metrics={'max_uint8':int(difference.max()),'normalized_mean_abs':float(difference.mean()/255),
             'changed_pixel_fraction':float(np.any(difference>0,axis=-1).mean())}
    if metrics['max_uint8']>1 or metrics['normalized_mean_abs']>1e-6 or metrics['changed_pixel_fraction']>1e-4:
        raise ValueError('context difference exceeds pre-score renderer-roundoff limits: '+str(metrics))
    return metrics


def canonicalize(root,clips):
    root=Path(root)
    by_key={(c['physical_condition'],c['appearance_condition']):c for c in clips}
    result=[];audit=[]
    for original in clips:
        branch=original['physical_condition'];appearance=original['appearance_condition']
        reference_ap='persistent_hue' if appearance=='persistent_hue' else 'clean'
        reference=by_key['lawful',reference_ap]
        raw_paths=[root/p for p in original['source_frame_paths']]
        canonical_paths=[root/p for p in reference['source_frame_paths']]
        actual=np.stack([np.asarray(Image.open(p).convert('RGBA')) for p in raw_paths[:12]])
        canonical=np.stack([np.asarray(Image.open(p).convert('RGBA')) for p in canonical_paths[:12]])
        metrics=roundoff_check(actual,canonical)
        out=root/'model-inputs'/branch/appearance/'frames';out.mkdir(parents=True,exist_ok=False)
        for i in range(48):
            src=canonical_paths[i] if i<12 else raw_paths[i]
            dst=out/f'rgba_{i:05d}.png';shutil.copyfile(src,dst)
            if sha(dst)!=sha(src):raise ValueError('model-input copy changed bytes')
        clip=copy.deepcopy(original)
        clip['raw_source_frame_paths']=original['source_frame_paths']
        clip['raw_source_frame_sha256']=original['source_frame_sha256']
        clip['source_frame_paths']=[(out/f'rgba_{i:05d}.png').relative_to(root).as_posix() for i in range(48)]
        clip['source_frame_sha256']=[sha(root/p) for p in clip['source_frame_paths']]
        if clip['source_frame_sha256'][12:]!=original['source_frame_sha256'][12:]:raise ValueError('future changed')
        result.append(clip)
        audit.append({'branch':branch,'appearance':appearance,'canonical_reference':f'lawful/{reference_ap}',
                      'raw_context_vs_reference':metrics,'future_bytes_unchanged':True})
    return result,audit
