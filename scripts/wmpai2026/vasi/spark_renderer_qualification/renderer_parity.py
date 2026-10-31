"""Exact, predeclared renderer comparison; lossless encoding bytes are diagnostic."""
from pathlib import Path
import hashlib, json
import numpy as np
from PIL import Image

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def compare_arrays(reference, observed):
    a=np.asarray(reference);b=np.asarray(observed)
    same_shape=a.shape==b.shape
    finite=bool(a.size and b.size and np.isfinite(a).all() and np.isfinite(b).all())
    equal=bool(same_shape and a.dtype==b.dtype and finite and np.array_equal(a,b))
    difference=None
    changed=None
    if same_shape and finite:
        difference=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
        changed=int(np.count_nonzero(a!=b))
    return {'pass':equal,'reference_shape':list(a.shape),'observed_shape':list(b.shape),'reference_dtype':str(a.dtype),'observed_dtype':str(b.dtype),'finite_nonempty':finite,'max_abs_difference':difference,'different_elements':changed}

def compare_cell(reference, observed):
    reference=Path(reference);observed=Path(observed)
    old=json.loads((reference/'clip.json').read_text());new=json.loads((observed/'clip.json').read_text())
    frames=[]
    for index in range(48):
        name=f'rgba_{index:05d}.png';a=reference/'frames'/name;b=observed/'frames'/name
        with Image.open(a) as image:x=np.asarray(image).copy()
        with Image.open(b) as image:y=np.asarray(image).copy()
        row=compare_arrays(x,y)
        row.update(frame=index,reference_png_sha256=sha(a),observed_png_sha256=sha(b))
        row['expected_rgba_shape']=x.shape==(256,256,4) and y.shape==(256,256,4)
        row['png_bytes_equal']=row['reference_png_sha256']==row['observed_png_sha256']
        row['reference_integrity']=row['reference_png_sha256']==old['source_frame_sha256'][index]
        row['observed_integrity']=row['observed_png_sha256']==new['source_frame_sha256'][index]
        row['pass']=row['pass'] and row['expected_rgba_shape'] and row['reference_integrity'] and row['observed_integrity']
        frames.append(row)
    assert sha(reference/'passes.npz')==old['passes_sha256'], 'reference pass file changed'
    assert sha(observed/'passes.npz')==new['passes_sha256'], 'observed pass file changed'
    with np.load(reference/'passes.npz',allow_pickle=False) as a,np.load(observed/'passes.npz',allow_pickle=False) as b:
        passes={name:compare_arrays(a[name],b[name]) for name in ['segmentation','depth']}
    state=old['state_trace_sha256']==new['state_trace_sha256']
    poses=old['rendered_target_positions']==new['rendered_target_positions']
    area=old['target_area_pixels_per_frame']==new['target_area_pixels_per_frame']
    result={'pass':all(r['pass'] for r in frames) and all(r['pass'] for r in passes.values()) and state and poses and area,'frame_count':48,'rgba_exact_frames':sum(r['pass'] for r in frames),'png_byte_equal_frames':sum(r['png_bytes_equal'] for r in frames),'maximum_rgba_level_difference':max(r['max_abs_difference'] or 0 for r in frames),'different_rgba_elements':sum(r['different_elements'] or 0 for r in frames),'state_trace_hash_equal':state,'poses_equal':poses,'target_area_equal':area,'passes':passes,'frames':frames,'gate':'exact decoded RGBA, segmentation, depth, state/pose/area; PNG encoding equality is diagnostic','numerical_tolerance':0}
    return result
