import os,traceback
from study_common import *
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';wait_permit('auditG')
 import numpy as np
 from geometry_analysis import tables,evaluate
 complete=load(ROOT/'geometry-complete.json');assert complete['status']=='pass';g=ROOT/'geometry';assert sha(g/'features.npy')==complete['features_sha256'];matrix=np.load(g/'features.npy',allow_pickle=False);index=load(g/'feature-index.json');rows=load(BASE/'final256-v1/scores/all-scores.json');assert matrix.shape==(3072,5,1408) and len(index)==3072
 for i,(row,record) in enumerate(zip(rows,index)):
  assert row['clip_id']==record['clip_id'];p=g/'clips'/(row['clip_id']+'.npy');assert sha(p)==record['feature_sha256'];assert matrix[i].tobytes()==np.load(p,allow_pickle=False).tobytes()
 calculated=tables(rows,matrix);coeff=load(ROOT/'geometry-coefficients.json');assert sha(ROOT/'geometry-coefficients.json')==complete['coefficient_sha256'];out=ROOT/'analysis-G';out.mkdir(exist_ok=False);results={}
 for task,table in calculated.items():
  assert table==load(g/(task+'-table.json'));assert not set(coeff['tasks'][task]['training_scenes'])&{r['scene_id'] for r in table};result,predictions=evaluate(table,coeff['tasks'][task]);results[task]=result;save(out/(task+'-predictions.json'),predictions)
 save(out/'summary.json',{'at':now(),'scope':'post-hoc development-only fit transfer; fixed-coefficient uncertainty; not mediation','results':results});save(out/'complete.json',{'at':now(),'status':'pass','summary_sha256':sha(out/'summary.json'),'full_feature_matrix_and_index_reverified':True})
if __name__=='__main__':
 try:main()
 except BaseException as e:save(ROOT/'auditG-failure.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
