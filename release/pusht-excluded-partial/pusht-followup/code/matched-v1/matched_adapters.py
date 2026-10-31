"""Frozen native FP32 PushT adapters; visual errors only, no refitting."""
from pathlib import Path
from types import SimpleNamespace
import json,os,sys
import numpy as np
import torch
ROOT=Path.home()/'jepa-wmpai/pusht-followup'
sys.path.insert(0,str(ROOT/'code'))
os.environ.update(TORCH_HOME=str(ROOT/'torch-cache'),HF_HOME=str(ROOT/'hf-cache'),XFORMERS_DISABLED='1',MPLBACKEND='Agg',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy')

def load(name):
 torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 if not torch.cuda.is_available():raise RuntimeError('Authorized CUDA required')
 device=torch.device('cuda')
 if name=='lewm':
  import stable_pretraining as spt
  from scripts.run_lewm_prediction_error_audit import load_model,normalize_action_blocks
  model,*_=load_model('pusht',device)
  stats=json.loads((ROOT/'data/action-normalization.json').read_text());mean=torch.tensor(stats['action_mean']);std=torch.tensor(stats['action_std'])
  transform=spt.data.transforms.Compose(spt.data.transforms.ToImage(**spt.data.dataset_stats.ImageNet,source='pixels',target='pixels'),spt.data.transforms.Resize(224,source='pixels',target='pixels'))
  def forward(pixels,actions,states):
   pixels=transform({'pixels':pixels})['pixels'].unsqueeze(0).to(device);actions=normalize_action_blocks(actions,mean,std).unsqueeze(0).to(device)
   z=model.encode({'pixels':pixels,'action':actions});pred=model.predict(z['emb'][:,:3],z['act_emb'][:,:3]);target=z['emb'][:,1:]
   return pred,target
 elif name=='dino':
  from scripts.run_dinowm_pusht_native_error_audit import add_dinowm_safe_globals,load_model
  from scripts.run_dinowm_pusht_controlled_displacement import normalize_action_blocks_for_dinowm,normalize_proprio_for_dinowm,transform_frames,move_predictor_masks_to_device
  add_dinowm_safe_globals(ROOT/'code/dino_wm')
  model,cfg,_,_=load_model(SimpleNamespace(checkpoint_root=ROOT/'checkpoints/dino-layout'),device);move_predictor_masks_to_device(model,device)
  from datasets.img_transforms import default_transform
  transform=default_transform(int(cfg.img_size))
  def forward(pixels,actions,states):
   obs={'visual':transform_frames(transform,pixels).unsqueeze(0).to(device),'proprio':normalize_proprio_for_dinowm(states).unsqueeze(0).to(device)}
   z=model.encode(obs,normalize_action_blocks_for_dinowm(actions,100.).unsqueeze(0).to(device));pred=model.predict(z[:,:model.num_hist]);target=z[:,model.num_pred:]
   if model.concat_dim==0:pred,target=pred[:,:,:-2,:],target[:,:,:-2,:]
   else:
    cut=-(model.proprio_dim+model.action_dim);pred,target=pred[:,:,:,:cut],target[:,:,:,:cut]
   return pred,target
 elif name=='jepa':
  from scripts.run_jepa_wm_pusht_controlled_displacement import load_model,normalize_action_blocks
  args=SimpleNamespace(jepa_wm_root=ROOT/'code/jepa-wms',checkpoint=Path(json.loads((ROOT/'logs/jepa-wm-staged.json').read_text())['path']))
  model,preprocessor,*_=load_model(args,device)
  def forward(pixels,actions,states):
   proprio=states[:,[0,1,5,6]]
   ctx=model.encode({'visual':pixels[:3].unsqueeze(0).to(device),'proprio':proprio[:3].unsqueeze(0).to(device)})
   target=model.encode({'visual':pixels[-1:].unsqueeze(0).to(device),'proprio':proprio[-1:].unsqueeze(0).to(device)})['visual']
   acts=normalize_action_blocks(actions,preprocessor).unsqueeze(0).to(device);features=model.model.encode_act(acts[:,:3])
   pred,_,_=model.model.forward_pred(ctx['visual'],features,ctx['proprio']);return pred,torch.cat([ctx['visual'][:,1:3],target],dim=1)
 else:raise ValueError(name)
 def score(pixels,actions,states):
  with torch.inference_mode():
   pred,target=forward(torch.from_numpy(pixels.copy()),torch.from_numpy(actions.copy()).float(),torch.from_numpy(states.copy()).float())
   temporal=(pred-target).pow(2).flatten(2).mean(dim=2)[0]
   if len(temporal)!=3 or not torch.isfinite(temporal).all():raise ValueError('Invalid native temporal errors')
   return pred[0,-1].float().cpu().numpy().reshape(-1),target[0,-1].float().cpu().numpy().reshape(-1),temporal.cpu().tolist()
 return score
