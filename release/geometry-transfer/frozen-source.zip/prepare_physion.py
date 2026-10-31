"""Whole-population Desktop CPU canonical seal before new scientific forwards."""
import os,time,traceback
from study_common import *
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';wait_permit('prepare');started=time.monotonic()
 from physion_common import video_frames,validate_population,static_variant,appearance_variant,motion,A_CONDITIONS,B_CONDITIONS,array_sha
 import numpy as np,torch,cv2,PIL,transformers
 from transformers import VJEPA2VideoProcessor
 from tensor_store import save_tensor
 from canonical_codec import load_tensor as old_load
 torch.set_num_threads(1);cv2.setNumThreads(1)
 assert cv2.__version__=='4.13.0' and np.__version__=='1.26.4' and PIL.__version__=='12.1.1' and transformers.__version__=='5.13.1'
 processor=VJEPA2VideoProcessor.from_pretrained(OLD/'model',local_files_only=True)
 def process(clip):return processor([clip],return_tensors='pt')['pixel_values_videos'].numpy()
 controls=lines(INPUTS/'control-manifest.jsonl');assert len(controls)==337
 cache=None;frames=None;fixtures={r['relative_path']:r for r in load(BASE/'reviewer-physion-20261026-v1/preprocessing-complete.json')['records']};seen=set()
 for i,r in enumerate(controls):
  deadline()
  if cache!=r['relative_video']:
   path=ROOT/r['relative_video'];assert sha(path)==r['source_sha256'];frames,_=video_frames(path);cache=r['relative_video']
  clip=static_variant(frames[r['source_start']:r['source_start']+16],r['condition']);x=process(clip);assert x.tobytes()==old_load(INPUTS/r['file'],r).tobytes()
  if r['source_start']==0 and r['condition']=='original' and cache in fixtures:
   assert array_sha(clip)==fixtures[cache]['rgb_sha256'] and array_sha(x)==fixtures[cache]['input_sha256'];seen.add(cache)
  if i%16==0:save(ROOT/'preparation-progress.json',{'at':now(),'phase':'337_old_input_checks','done':i+1})
 assert len(seen)==10;save(ROOT/'old-input-recheck.json',{'at':now(),'status':'pass','controls':337,'prior_fixtures':10,'model_forwards':0})
 pop=load(ROOT/'population.json');validate_population(pop);(ROOT/'inputs').mkdir(exist_ok=False);a_count=b_count=0
 for i,entry in enumerate(pop):
  deadline();path=ROOT/entry['relative_path'];assert sha(path)==entry['source_sha256'];frames,fps=video_frames(path);n=len(frames)//16;assert n>0;middle=n//2
  audit={'ordinal':i,**entry,'frame_count':len(frames),'fps':fps,'shape':list(frames.shape),'decoded_sha256':array_sha(frames),'segments':n,'middle_segment':middle,'dropped_tail_frames':len(frames)%16,'windows':[{'segment':j,'start':j*16,'end_inclusive':j*16+15,**motion(frames[j*16:j*16+16])} for j in range(n)]}
  append(ROOT/'input-audit.jsonl',audit)
  for study,segments,conditions,transform in [('B',[middle],B_CONDITIONS,appearance_variant),('A',range(n),A_CONDITIONS,static_variant)]:
   b_records={}
   for j in segments:
    clip=frames[j*16:j*16+16]
    for c in conditions:
     deadline();raw=transform(clip,c);x=process(raw);name=f'inputs/{study}-{i:03d}-{j:03d}-{c}.npz';info=save_tensor(ROOT/name,x)
     mapping=list(range(j*16,j*16+16)) if not c.startswith('static_') else [j*16+{'static_first':0,'static_middle':8,'static_last':15}[c]]*16
     r={'study':study,'ordinal':i,'video_id':entry['video_id'],'pair_id':entry['pair_id'],'scenario':entry['scenario'],'layout_group':entry['layout_group'],'segment':j,'start':j*16,'condition':c,'source_frame_mapping':mapping,'source_sha256':entry['source_sha256'],'raw_rgb_sha256':array_sha(raw),'raw_context_sha256':array_sha(raw[:12]),'raw_future_sha256':array_sha(raw[12:]),'file':name,**info};append(ROOT/(study+'-input-manifest.jsonl'),r)
     if study=='B':b_records[c]=r;b_count+=1
     else:a_count+=1
    if study=='B':
     for kind in ['hue','brightness']:
      assert b_records['persistent_'+kind]['future_sha256']==b_records['target_only_'+kind]['future_sha256'];assert b_records['target_only_'+kind]['context_sha256']==b_records['original']['context_sha256']
      assert b_records['persistent_'+kind]['raw_future_sha256']==b_records['target_only_'+kind]['raw_future_sha256'];assert b_records['target_only_'+kind]['raw_context_sha256']==b_records['original']['raw_context_sha256']
   save(ROOT/'preparation-progress.json',{'at':now(),'phase':'full_population_canonical_inputs','videos_completed':i+1,'A_cells':a_count,'B_cells':b_count,'elapsed_seconds':time.monotonic()-started})
  del frames
 assert a_count==20708 and b_count==960 and not torch.cuda.is_initialized()
 save(ROOT/'input-seal.json',{'at':now(),'status':'pass','videos':192,'pairs':96,'segments':5177,'A_cells':a_count,'B_cells':b_count,'old_controls_exact':337,'old_fixtures_exact':10,'files':{n:sha(ROOT/n) for n in ['population.json','input-audit.jsonl','A-input-manifest.jsonl','B-input-manifest.jsonl','old-input-recheck.json']},'elapsed_seconds':time.monotonic()-started,'cuda_initialized':False,'freeze_sha256':sha(ROOT/'freeze.json')})
if __name__=='__main__':
 try:main()
 except BaseException as e:save(ROOT/'prepare-failure.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
