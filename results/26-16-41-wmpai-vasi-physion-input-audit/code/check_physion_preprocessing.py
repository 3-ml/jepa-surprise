"""Exact archived-reference input check on CPU, before model qualification."""
from pathlib import Path
import json,os,traceback
import torch
from transformers import VJEPA2VideoProcessor
from physion_common import *
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(4);cv2.setNumThreads(1)
 with (ROOT/'preprocessing-parity-lock.json').open('x') as f:json.dump({'at':now(),'pid':os.getpid()},f)
 expected=json.loads((ROOT/'preprocessing-complete.json').read_text());assert sha(ROOT/'preprocessing-fixtures.npz')==expected['archive_sha256']
 processor=VJEPA2VideoProcessor.from_pretrained(ROOT/'model',local_files_only=True)
 records=[]
 with np.load(ROOT/'preprocessing-fixtures.npz',allow_pickle=False) as fixtures:
  for row in expected['records']:
   path=ROOT/row['relative_path'];assert sha(path)==row['source_sha256'];frames,_=video_frames(path);clip=frames[:16]
   actual=processor([clip],return_tensors='pt')['pixel_values_videos'].numpy();reference=fixtures[f"inputs_{row['index']:02d}"]
   result={'index':row['index'],'raw_rgb_exact':array_sha(clip)==row['rgb_sha256'],'input_exact':array_sha(actual)==row['input_sha256'],'shape':list(actual.shape),'maximum_abs_input_difference':float(np.max(np.abs(actual-reference))),'changed_input_elements':int(np.count_nonzero(actual!=reference))}
   records.append(result);save(ROOT/'preprocessing-parity-progress.json',{'at':now(),'records':records})
   assert result['raw_rgb_exact'] and result['input_exact'] and actual.shape==(1,16,3,256,256), 'Exact preprocessing gate failed; no model cells allowed'
 assert not torch.cuda.is_initialized()
 save(ROOT/'preprocessing-parity.json',{'at':now(),'status':'pass','records':records,'cuda_initialized':False,'checkpoint_loaded':False,'processor_version':processor.__class__.__name__})
if __name__=='__main__':
 try:main()
 except BaseException as e:save(ROOT/'preprocessing-parity-failure.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
