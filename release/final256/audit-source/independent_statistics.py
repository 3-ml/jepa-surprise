"""CPU-only independent reconstruction: bootstrap frequency weights, tie-group AUC.
Never imports the frozen analysis implementation or computes new model scores.
"""
import numpy as np
from scipy.stats import beta
APPS=('clean','persistent_hue','abrupt_event'); WORLDS=('solid','gap','balanced_worlds')
THRESHOLD=0.4858863055706024

def weighted_auc(x,counts):
    # x: config, world, branch; counts: bootstrap draw, config.
    n,w,_=x.shape;values=x.reshape(-1);positive=np.tile([False,True],n*w)
    order=np.argsort(values,kind='stable');starts=np.r_[0,1+np.flatnonzero(np.diff(values[order]))]
    weights=np.repeat(counts,w*2,axis=1)[:,order]
    pos=np.add.reduceat(weights*positive[order],starts,axis=1)
    neg=np.add.reduceat(weights*(~positive[order]),starts,axis=1)
    return (pos*(np.cumsum(neg,axis=1)-neg+.5*neg)).sum(axis=1)/(counts.sum(axis=1)*w)**2

def calculate(rows):
    ids=sorted({r['scene_id'] for r in rows});n=len(ids);lookup={(r['scene_id'],r['world'],r['physical_condition'],r['appearance_condition']):r['score'] for r in rows}
    assert len(lookup)==len(rows)==n*12
    v=np.array([[[[lookup[s,w,b,a] for a in APPS] for b in ['lawful','inconsistent']] for w in WORLDS[:2]] for s in ids],dtype=np.float64)
    margins=v[:,:,1,:]-v[:,:,0,:];by_world=np.concatenate([margins,margins.mean(axis=1,keepdims=True)],axis=1)
    p=np.column_stack([margins[:,0,0],margins[:,1,0],margins[:,0,1],margins[:,1,1]])
    B=50000;S=10000;rng=np.random.default_rng(202610256)
    pm=np.empty((B,4));bm=np.empty((B,3,3));fa=np.empty((S,3,3));miss=np.empty_like(fa);auc=np.empty_like(fa)
    for start in range(0,B,250):
        idx=rng.integers(0,n,(min(250,B-start),n));counts=np.zeros((len(idx),n),dtype=np.float64)
        for j,indices in enumerate(idx):counts[j]=np.bincount(indices,minlength=n)
        pm[start:start+len(idx)]=counts@p/n;bm[start:start+len(idx)]=(counts@by_world.reshape(n,-1)/n).reshape(-1,3,3)
        if start<S:
            for wi in range(3):
                for ai in range(3):
                    x=v[:,:,:,ai] if wi==2 else v[:,wi:wi+1,:,ai]
                    fa[start:start+len(idx),wi,ai]=counts@(x[:,:,0]>=THRESHOLD).mean(axis=1)/n
                    miss[start:start+len(idx),wi,ai]=counts@(x[:,:,1]<THRESHOLD).mean(axis=1)/n
                    auc[start:start+len(idx),wi,ai]=weighted_auc(x,counts)
    primary={};secondary={}
    for j,key in enumerate(['clean_solid','clean_gap','persistent_solid','persistent_gap']):
        mean=float(p[:,j].mean());lo,hi=np.quantile(pm[:,j],[.00625,.99375]);extent=max(mean-lo,hi-mean)
        primary[key]={'mean':mean,'simultaneous_interval':[float(lo),float(hi)],'direction':'positive' if lo>0 else ('negative' if hi<0 else 'unresolved'),'maximum_half_extent':float(extent),'reporting_precision_met':bool(extent<=.001)}
    for ai,app in enumerate(APPS):
        secondary[app]={}
        for wi,world in enumerate(WORLDS):
            z=by_world[:,wi,ai];k=int((z>0).sum());x=v[:,:,:,ai] if wi==2 else v[:,wi:wi+1,:,ai]
            f=float((x[:,:,0]>=THRESHOLD).mean());m=float((x[:,:,1]<THRESHOLD).mean())
            secondary[app][world]={'mean_margin':float(z.mean()),'descriptive_95_mean_interval':np.quantile(bm[:,wi,ai],[.025,.975]).tolist(),'median_margin':float(np.median(z)),'positive_count':k,'ties':int((z==0).sum()),'positive_fraction':k/n,'descriptive_clopper_pearson_interval':[0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))],'false_alarm_rate':f,'false_alarm_95_interval':np.quantile(fa[:,wi,ai],[.025,.975]).tolist(),'miss_rate':m,'miss_95_interval':np.quantile(miss[:,wi,ai],[.025,.975]).tolist(),'balanced_accuracy':1-(f+m)/2,'balanced_accuracy_95_interval':np.quantile(1-(fa[:,wi,ai]+miss[:,wi,ai])/2,[.025,.975]).tolist(),'auroc':float(weighted_auc(x,np.ones((1,n)))[0]),'auroc_95_interval':np.quantile(auc[:,wi,ai],[.025,.975]).tolist()}
    change={APPS[ai]:{'mean':float((by_world[:,2,ai]-by_world[:,2,0]).mean()),'descriptive_95_interval':np.quantile(bm[:,2,ai]-bm[:,2,0],[.025,.975]).tolist()} for ai in [1,2]}
    return {'primary':primary,'secondary':secondary,'paired_appearance_change_in_balanced_margin':change,'consistent_positive_mean_ordering':all(r['direction']=='positive' for r in primary.values())}

def compare(expected,actual):
    gaps=[]
    def visit(a,b,path):
        if isinstance(a,dict):
            assert set(a)==set(b),path
            for k in a:visit(a[k],b[k],path+'/'+k)
        elif isinstance(a,list):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i))
        elif isinstance(a,(bool,str)) or a is None:assert a==b,path
        else:
            gap=abs(float(a)-float(b));assert np.isfinite(gap) and gap<=5e-14,(path,a,b,gap);gaps.append(gap)
    visit(expected,actual,'')
    return {'numeric_values_checked':len(gaps),'maximum_absolute_difference':max(gaps,default=0),'audit_arithmetic_tolerance':5e-14}
