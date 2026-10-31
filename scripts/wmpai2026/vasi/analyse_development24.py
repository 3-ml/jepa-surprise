"""Fixed exploratory scene-grouped analysis; never generates final scenes."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'scripts/wmpai2026/alex'))
import a2_scene_stats as stats
import a3_a4_lib as shortcuts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);root=ap.parse_args().root
    rows=json.loads((root/'scores/all-scores.json').read_text())
    missing=json.loads((root/'scores/missing-scenes.json').read_text())
    scenes=sorted({r['scene_id'] for r in rows})
    out=root/'analysis';out.mkdir(exist_ok=False)
    if not rows:raise ValueError('no completed rows')
    for world in ('solid','gap'):
        errors=stats.validate([r for r in rows if r['world']==world])
        if errors:raise ValueError(errors)
    summary={'scope':'development_only_conditional_on_shared_procedural_template',
             'attempted_scenes':24,'complete_scenes':len(scenes),'missing_scenes':missing,
             'bootstrap_draws':10000,'bootstrap_seed':20261025,'per_appearance':{},'shortcut_diagnostics':{},
             'confirmatory_status':'not_frozen_or_started; practical effect and precision target remain unset'}
    table=[];balanced={}
    lookup={(r['scene_id'],r['world'],r['physical_condition'],r['appearance_condition']):r['score'] for r in rows}
    for appearance in ('clean','persistent_hue','abrupt_event'):
        solid={s:lookup[s,'solid','inconsistent',appearance]-lookup[s,'solid','lawful',appearance] for s in scenes}
        gap={s:lookup[s,'gap','inconsistent',appearance]-lookup[s,'gap','lawful',appearance] for s in scenes}
        b={s:(solid[s]+gap[s])/2 for s in scenes};balanced[appearance]=b
        sd=float(np.std(list(b.values()),ddof=1)) if len(scenes)>1 else None
        means=[float(np.mean([v for s,v in b.items() if s!=held])) for held in scenes] if len(scenes)>1 else []
        summary['per_appearance'][appearance]={
            'solid_margin':stats.pair_stats(solid,n_boot=10000,seed=20261025),
            'gap_margin':stats.pair_stats(gap,n_boot=10000,seed=20261025),
            'geometry_balanced_margin':stats.pair_stats(b,n_boot=10000,seed=20261025),
            'between_configuration_sd_balanced':sd,
            'leave_one_config_out_mean_range':[min(means),max(means)] if means else None,
            'planning_only_normal_halfwidth_by_n':{str(n):1.96*sd/np.sqrt(n) for n in (32,64,96,128,192,256)} if sd is not None else {},
            'native_clip_auroc':stats.auroc([r['score'] for r in rows if r['appearance_condition']==appearance and r['physical_condition']=='inconsistent'],
                                          [r['score'] for r in rows if r['appearance_condition']==appearance and r['physical_condition']=='lawful'])}
        for s in scenes:table.append({'scene_id':s,'appearance':appearance,'solid_margin':solid[s],'gap_margin':gap[s],'balanced_margin':b[s]})
        selected=[r for r in rows if r['appearance_condition']==appearance]
        if len(scenes)>=4:
            X=np.array([[r['frame_features'][k] for k in ('static','motion','shuffled')]+[float(r['world']=='gap'),float(r['trajectory']=='fall')] for r in selected])
            y=np.array([r['score'] for r in selected]);g=np.array([r['scene_id'] for r in selected])
            flag=np.array([float(r['physical_condition']=='inconsistent') for r in selected])
            if np.var(y)>0:
                base,full,gain=shortcuts.nested_gain(X,flag,y,g)
                summary['shortcut_diagnostics'][appearance]={'method':'Alex leave-one-base-configuration-out OLS',
                    'base_features':['static','motion','shuffled','gap_indicator','fall_indicator'],
                    'native_score_base_r2':base,'native_score_with_physical_flag_r2':full,'flag_r2_gain':gain,
                    'interpretation':'association only; physical flag is geometry-by-trajectory interaction, not independent causal evidence'}
    summary['paired_appearance_effect_on_balanced_margin']={a:stats.pair_stats(
        {s:balanced[a][s]-balanced['clean'][s] for s in scenes},n_boot=10000,seed=20261025)
        for a in ('persistent_hue','abrupt_event')}
    summary['limitations']=['shared scene template, selected grid; bootstrap not a natural-population confidence guarantee',
      'different backgrounds/support geometry intentionally supply causal context; image target trajectories not necessarily identical pixels due occlusion/shadows',
      'mean score across full image/future may dilute event effects',
      'motion/position nuisance reduced by crossed design, not proven fully removed',
      'no equivalence or trained threshold/causal geometry claim',
      'negative or inconclusive results preserved without seed substitution']
    with (out/'scene-margins.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete_scenes':len(scenes),'missing_scenes':len(missing),
                     'balanced_margins':{a:r['geometry_balanced_margin'] for a,r in summary['per_appearance'].items()}}),flush=True)

if __name__=='__main__':main()
