"""Export only full-dataset action column, with no torch import/GPU work."""
from pathlib import Path
import datetime,hashlib,json,os,time,traceback
ROOT=Path(__import__("os").path.join(__import__("os").environ['JEPA_VASI_WORKSPACE'], 'pusht-spark-preparation/action-normalization'))
SOURCE=Path(__import__("os").environ['JEPA_LEWM_PUSHT_HDF5'])
EXPECTED='b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,data):(ROOT/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
def main():
 ROOT.mkdir(exist_ok=False);started=time.time()
 save('launch.json',{'pid':os.getpid(),'at':datetime.datetime.now().astimezone().isoformat(),'gpu_work':False,'source_sha256':sha(__file__)})
 try:
  stat=SOURCE.stat();digest=sha(SOURCE)
  if digest!=EXPECTED:raise ValueError('Source dataset mismatch')
  import h5py,numpy as np,hdf5plugin
  with h5py.File(SOURCE,'r',swmr=True,rdcc_nbytes=8<<20) as f:
   column=f['action']
   if column.size*column.dtype.itemsize>512<<20:raise ValueError('Unexpectedly large action column; stop before allocating')
   array=column[:]
  if SOURCE.stat().st_mtime_ns!=stat.st_mtime_ns or SOURCE.stat().st_size!=stat.st_size:raise ValueError('Dataset changed during export')
  out=ROOT/'full-dataset-actions.npy';np.save(out,array,allow_pickle=False)
  save('provenance.json',{'status':'complete','at':datetime.datetime.now().astimezone().isoformat(),'source':str(SOURCE),'dataset_sha256':digest,'dataset_bytes':stat.st_size,'action_shape':list(array.shape),'action_dtype':str(array.dtype),'actions_sha256':sha(out),'actions_bytes':out.stat().st_size,'nan_rows':int(np.isnan(array).any(axis=1).sum()),'seconds':time.time()-started,'gpu_work':False,'purpose':'Exact raw full-column input to original Torch mean/std formula, not subset-derived normalization'})
 except BaseException as e:save('failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
