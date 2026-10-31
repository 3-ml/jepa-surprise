import importlib.util,json,sys
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1]/'scripts/wmpai2026/vasi';sys.path.insert(0,str(HERE))
import final256_common as common
import final256_analyse as analysis

def development():return json.loads((HERE.parents[2]/'protocol/vasi/development24-configs.json').read_text())
def test_fixed_fresh_manifest_deterministic_and_independent_factors():
    a=common.build_manifest(development());b=common.build_manifest(development());assert a==b and len(a['scenes'])==256
    assert len({c['scene_seed'] for c in a['scenes']})==256
    assert {c['split'] for c in a['scenes']}=={'confirmatory'}
    assert len({(c['camera_position'][0],tuple(c['base_color_rgb'])) for c in a['scenes']})==16
    assert all(.6<c['sphere_radius']<.7 and 1.18<c['nominal_contact_time']<1.24 for c in a['scenes'])
    assert len({common.parameter_fingerprint(c) for c in a['scenes']})==256

def test_parameter_fingerprint_does_not_depend_on_seed_or_label():
    c=development()['scenes'][0];d={**c,'scene_seed':0,'scene_id':'new','split':'confirmatory','purpose':'new'}
    assert common.parameter_fingerprint(c)==common.parameter_fingerprint(d)

def synthetic_rows(n=2):
    return [dict(scene_id=f'test-{i}',world=w,physical_condition=b,appearance_condition=a,split='confirmatory',
                 score=.48+i*.0001+(.01 if b=='inconsistent' else 0)+j*.001)
            for i in range(n) for w in common.WORLDS for b in common.BRANCHES for j,a in enumerate(common.APPEARANCES)]

def test_exact_grid_and_no_missing_as_zero():
    rows=synthetic_rows();s,v=analysis.grid(rows);assert v.shape==(2,2,2,3)
    with pytest.raises(ValueError,match='duplicate'):analysis.grid(rows+[rows[0]])
    with pytest.raises(ValueError,match='incomplete'):analysis.grid(rows[:-1])
    rows[0]['split']='development'
    with pytest.raises(ValueError):analysis.grid(rows)

def test_auroc_ties_order_and_binomial_boundaries():
    x=np.zeros((3,8,2,2));assert np.all(analysis.auc_batch(x)==.5)
    x[:,:,:,1]=1;assert np.all(analysis.auc_batch(x)==1)
    assert np.all(analysis.auc_batch(x[:,:,:,::-1])==0)
    assert analysis.binomial_interval(0,24)[1]>0
    assert analysis.binomial_interval(24,24)[0]<1

def test_cross_world_rejects_nonmatching_and_old_trace(tmp_path):
    row={'position':[0,0,1],'quaternion_xyzw':[0,0,0,1],'linear_velocity':[0,0,0],'angular_velocity':[0,0,0]}
    fall={**row,'position':[0,0,-2]}
    for w in common.WORLDS:
        for b in common.BRANCHES:
            trace=[row] if (w=='solid')==(b=='lawful') else [fall]
            common.save(tmp_path/w/(b+'-states.json'),{'trace':trace})
    got=common.crossed_and_fresh(tmp_path,set());assert got['exact_crossed_target_states']
    with pytest.raises(ValueError,match='duplicated'):common.crossed_and_fresh(tmp_path,{got['stop_sha256']})
    common.save(tmp_path/'gap/inconsistent-states.json',{'trace':[fall]})
    with pytest.raises(ValueError,match='cross-world'):common.crossed_and_fresh(tmp_path,set())

def test_full_frozen_analysis_on_synthetic_data(tmp_path,monkeypatch):
    common.save(tmp_path/'run-lock.json',{'source_sha256':{}})
    common.save(tmp_path/'threshold.json',{'threshold':common.THRESHOLD})
    rows=synthetic_rows(256);common.save(tmp_path/'scores/all-scores.json',rows)
    common.save(tmp_path/'scores/provenance.json',{'status':'complete'})
    common.save(tmp_path/'render-seal.json',{'eligible_count':256,'candidates':[{'scene_id':f'test-{i}','status':'eligible'} for i in range(256)]})
    monkeypatch.setattr(sys,'argv',['analysis','--root',str(tmp_path)])
    analysis.main();report=common.load(tmp_path/'analysis/summary.json')
    assert report['eligible_scenes']==256 and report['consistent_positive_mean_ordering']
    assert all(abs(s['mean']-.01)<1e-12 for s in report['primary'].values())
    assert report['secondary']['clean']['solid']['positive_count']==256
    assert report['utility_equivalence_or_causal_gate'].startswith('not assessed')
