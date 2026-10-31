"""CPU-only audit precedes all model work; no score-driven segment selection."""
import json,os,traceback
from pathlib import Path
from physion_common import *
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 with (ROOT/'input-audit-lock.json').open('x') as f:json.dump({'at':now(),'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21]},f)
 cv2.setNumThreads(1)
 pop=json.loads((ROOT/'population.json').read_text());validate_population(pop)
 observations=[]
 for i,row in enumerate(pop):
  path=ROOT/row['relative_path'];assert sha(path)==row['source_sha256']
  frames,fps=video_frames(path);n=len(frames)//16;assert n>0
  windows=[{'segment':j,'start':16*j,'end_inclusive':16*j+15,**motion(frames[16*j:16*j+16])} for j in range(n)]
  item={'ordinal':i,'video_id':row['video_id'],'pair_id':row['pair_id'],'scenario':row['scenario'],'layout_group':row['layout_group'],'relative_path':row['relative_path'],'source_sha256':row['source_sha256'],'frame_count':len(frames),'fps':fps,'shape':list(frames.shape),'decoded_sha256':array_sha(frames),'segments':n,'middle_segment':n//2,'dropped_tail_frames':len(frames)%16,'windows':windows}
  append(ROOT/'input-audit.jsonl',item);observations.append(item)
  save(ROOT/'input-audit-progress.json',{'at':now(),'done':i+1,'total':192,'model_inference':False})
  del frames
 save(ROOT/'input-audit-complete.json',{'at':now(),'videos':192,'pairs':96,'segments':sum(r['segments'] for r in observations),'planned_A_cells':4*sum(r['segments'] for r in observations),'planned_B_cells':960,'input_audit_sha256':sha(ROOT/'input-audit.jsonl'),'population_sha256':sha(ROOT/'population.json'),'model_inference':False,'gate':'all hashes, complete frame decode and official pair membership passed'})
if __name__=='__main__':
 try:main()
 except BaseException as e:save(ROOT/'input-audit-failure.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
