"""Pre-frozen final256 analysis. No outcome-dependent sample extension or metric choice."""
import argparse, datetime, json, shutil
from pathlib import Path
import numpy as np
from scipy.stats import beta, rankdata
from final256_common import APPEARANCES, WORLDS, BRANCHES, THRESHOLD, load, save, sha, check_sources
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
PRIMARY=('clean_solid','clean_gap','persistent_solid','persistent_gap')

def grid(rows):
    scenes=sorted({r['scene_id'] for r in rows});lookup={}
    for r in rows:
        key=(r['scene_id'],r['world'],r['physical_condition'],r['appearance_condition'])
        if key in lookup or r['split']!='confirmatory':raise ValueError('duplicate/wrong split')
        lookup[key]=r['score']
    expected={(s,w,b,a) for s in scenes for w in WORLDS for b in BRANCHES for a in APPEARANCES}
    if set(lookup)!=expected:raise ValueError('incomplete grid')
    values=np.array([[[[lookup[s,w,b,a] for a in APPEARANCES] for b in BRANCHES] for w in WORLDS] for s in scenes])
    if not np.isfinite(values).all():raise ValueError('nonfinite scores')
    return scenes,values

def auc_batch(values):
    # shape (bootstrap draws, configurations, worlds, branches), branch 1 positive.
    z=values.reshape(len(values),-1);labels=np.tile([False,True],z.shape[1]//2)
    ranks=rankdata(z,method='average',axis=1);n=labels.sum()
    return (ranks[:,labels].sum(axis=1)-n*(n+1)/2)/(n*n)

def binomial_interval(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    check_sources(REPO,load(root/'run-lock.json'))
    provenance=load(root/'scores/provenance.json');rows=load(root/'scores/all-scores.json');seal=load(root/'render-seal.json')
    if provenance['status']!='complete':raise ValueError('score completion gate')
    if load(root/'threshold.json')['threshold']!=THRESHOLD:raise ValueError('threshold changed')
    scenes,v=grid(rows);n=len(scenes)
    if n!=seal['eligible_count'] or n<231 or len(rows)!=12*n:raise ValueError('unexpected final denominator')
    out=root/'analysis'
    if out.exists():
        if (out/'summary.json').exists() and load(out/'summary.json').get('source_scores_sha256')==sha(root/'scores/all-scores.json'):return
        archive=root/'analysis-attempts';archive.mkdir(exist_ok=True);shutil.move(str(out),str(archive/datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    out.mkdir()
    m=v[:,:,1,:]-v[:,:,0,:];p=np.c_[m[:,0,0],m[:,1,0],m[:,0,1],m[:,1,1]]
    rng=np.random.default_rng(202610256);B=50000;S=10000
    primary=np.empty((B,4));mboot=np.empty((B,2,3));bboot=np.empty((B,3))
    fboot=np.empty((S,3,3));missboot=np.empty_like(fboot);aboot=np.empty_like(fboot)
    for start in range(0,B,250):
        size=min(250,B-start);idx=rng.integers(0,n,(size,n))
        primary[start:start+size]=p[idx].mean(axis=1);mboot[start:start+size]=m[idx].mean(axis=1)
        bboot[start:start+size]=m[idx].mean(axis=(1,2))
        if start<S:
            x=v[idx]
            for ai in range(3):
                for wi in range(3):
                    selected=x[:,:,:, :,ai] if wi==2 else x[:,:,wi:wi+1,:,ai]
                    fboot[start:start+size,wi,ai]=(selected[:,:,:,0]>=THRESHOLD).mean(axis=(1,2))
                    missboot[start:start+size,wi,ai]=(selected[:,:,:,1]<THRESHOLD).mean(axis=(1,2))
                    aboot[start:start+size,wi,ai]=auc_batch(selected)
    ci=np.quantile(primary,[.00625,.99375],axis=0)
    result={'status':'complete','scope':'frozen final; eligible sampled procedural population only; earlier joint draft not jointly signed',
        'completed_at':datetime.datetime.now().astimezone().isoformat(),'attempted_scenes':256,'eligible_scenes':n,'scored_clips':len(rows),
        'missing_scenes':[c for c in seal['candidates'] if c['status']!='eligible'],
        'source_scores_sha256':sha(root/'scores/all-scores.json'),'run_lock_sha256':sha(root/'run-lock.json'),
        'bootstrap_seed':202610256,'primary_draws':B,'secondary_rate_auc_draws':S,
        'primary_interval':'98.75% percentile scene-bootstrap intervals, Bonferroni for four means at familywise alpha .05',
        'reporting_precision_target':.001,'threshold':THRESHOLD,'primary':{},'secondary':{},
        'utility_equivalence_or_causal_gate':'not assessed; no validated utility ceiling, equivalence margin or causal intervention'}
    for i,name in enumerate(PRIMARY):
        mean=float(p[:,i].mean());low,high=ci[:,i]
        result['primary'][name]={'mean':mean,'simultaneous_interval':[float(low),float(high)],
            'direction':'positive' if low>0 else ('negative' if high<0 else 'unresolved'),
            'maximum_half_extent':max(mean-float(low),float(high)-mean),
            'reporting_precision_met':bool(max(mean-low,high-mean)<=.001)}
    result['consistent_positive_mean_ordering']=all(e['direction']=='positive' for e in result['primary'].values())
    for ai,appearance in enumerate(APPEARANCES):
        entry={}
        for wi,world in enumerate((*WORLDS,'balanced_worlds')):
            margin=m[:,wi,ai] if wi<2 else m[:,:,ai].mean(axis=1)
            boot=mboot[:,wi,ai] if wi<2 else bboot[:,ai]
            selected=v[:,:, :,ai][None] if wi==2 else v[:,wi:wi+1,:,ai][None]
            fa=float((selected[:,:,:,0]>=THRESHOLD).mean());miss=float((selected[:,:,:,1]<THRESHOLD).mean())
            k=int((margin>0).sum())
            entry[world]={'mean_margin':float(margin.mean()),'descriptive_95_mean_interval':np.quantile(boot,[.025,.975]).tolist(),
                'median_margin':float(np.median(margin)),'positive_count':k,'ties':int((margin==0).sum()),
                'positive_fraction':k/n,'descriptive_clopper_pearson_interval':binomial_interval(k,n),
                'false_alarm_rate':fa,'false_alarm_95_interval':np.quantile(fboot[:,wi,ai],[.025,.975]).tolist(),
                'miss_rate':miss,'miss_95_interval':np.quantile(missboot[:,wi,ai],[.025,.975]).tolist(),
                'balanced_accuracy':1-(fa+miss)/2,'balanced_accuracy_95_interval':np.quantile(1-(fboot[:,wi,ai]+missboot[:,wi,ai])/2,[.025,.975]).tolist(),
                'auroc':float(auc_batch(selected)[0]),'auroc_95_interval':np.quantile(aboot[:,wi,ai],[.025,.975]).tolist()}
        result['secondary'][appearance]=entry
    result['paired_appearance_change_in_balanced_margin']={APPEARANCES[ai]:{'mean':float((m[:,:,ai].mean(axis=1)-m[:,:,0].mean(axis=1)).mean()),
        'descriptive_95_interval':np.quantile(bboot[:,ai]-bboot[:,0],[.025,.975]).tolist()} for ai in (1,2)}
    result['limitations']=['Conditional single procedural family, not natural-video population prevalence.',
        'Four primary mean intervals have multiplicity adjustment; secondary counts/calibration/abrupt results are descriptive.',
        'Small or unresolved balanced means do not establish equivalence or absence of physical knowledge.',
        'No threshold refit, score inversion, window selection, layer selection or outcome-dependent extension.',
        'Sample eligibility may condition the population; all excluded candidates are reported.']
    save(out/'summary.json',result)
    print('FINAL ANALYSIS COMPLETE; no further experiment queued',flush=True)
if __name__=='__main__':main()
