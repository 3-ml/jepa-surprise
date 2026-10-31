import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from study_common import *
import numpy as np
from geometry_analysis import tables,fit
root=BASE/'development24-v2';g=root/'geometry-development-v2';rows=load(root/'scores/all-scores.json');features=np.load(g/'features.npy',allow_pickle=False);index=load(g/'feature-index.json');assert len(rows)==288 and features.shape==(288,5,1408)
assert sha(g/'features.npy')==load(g/'provenance.json')['feature_sha256']
for i,(r,v) in enumerate(zip(rows,index)):assert r['clip_id']==v['clip_id'] and sha(g/'clips'/(r['clip_id']+'.npy'))==v['feature_sha256'] and np.array_equal(features[i],np.load(g/'clips'/(r['clip_id']+'.npy'),allow_pickle=False))
t=tables(rows,features)
for task,file in [('physical_margin','physical-displacements.json'),('appearance_score_change','appearance-displacements.json')]:
 old=load(g/file);assert len(old)==len(t[task])
 for a,b in zip(t[task],old):
  for k in ['scene_id','world','appearance','y','base','displacement']:assert a[k]==b[k],k
save(ROOT/'geometry-coefficients.json',{'at':now(),'scope':'development24 fit only; final scores already inspected but not read/fitted by this script','source_hashes':{str(p):sha(p) for p in [root/'scores/all-scores.json',g/'features.npy',g/'feature-index.json',g/'provenance.json']},'tasks':{k:fit(v) for k,v in t.items()},'development_table_reconstruction':'exact'})
print('Development-only coefficient freeze created; both old diagnostic tables reconstructed exactly')
