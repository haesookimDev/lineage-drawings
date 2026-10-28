#!/usr/bin/env python3
"""Fine-tune official YOLOX-Nano on training-only tiles, CPU, no test access."""
import argparse,hashlib,json,math,random,sys,time
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'.benchmark-data/yolox-source'))
from yolox.models import YOLOX,YOLOPAFPN,YOLOXHead


def model():
 m=YOLOX(YOLOPAFPN(.33,.25,in_channels=[256,512,1024],depthwise=True),YOLOXHead(1,.25,in_channels=[256,512,1024],depthwise=True))
 for layer in m.modules():
  if isinstance(layer,torch.nn.BatchNorm2d):layer.eps=1e-3;layer.momentum=.03
 m.head.initialize_biases(.01);return m


class Tiles(torch.utils.data.Dataset):
 def __init__(self):
  self.rows=json.loads((ROOT/'evaluation/v6/TRAINING-DATA.json').read_text())['tiles'];self.images=[cv2.imread(str(ROOT/r['file'])) for r in self.rows]
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  im=self.images[i].copy();boxes=np.array(self.rows[i]['boxes'],np.float32)
  if random.random()<.5:im=im[:,::-1].copy();boxes[:,[0,2]]=640-boxes[:,[2,0]]
  if random.random()<.5:
   contrast=random.uniform(.55,1.15);im=np.clip(255-(255-im.astype(np.float32))*contrast+random.uniform(-12,8),0,255).astype(np.uint8)
  if random.random()<.15:im=cv2.GaussianBlur(im,(3,3),.5)
  targets=np.zeros((150,5),np.float32);boxes=boxes[:150];targets[:len(boxes),1:3]=(boxes[:,:2]+boxes[:,2:])/2;targets[:len(boxes),3:]=boxes[:,2:]-boxes[:,:2]
  return torch.from_numpy(im.transpose(2,0,1).copy()).float(),torch.from_numpy(targets)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=24);ap.add_argument('--steps-per-epoch',type=int,default=96);ap.add_argument('--batch',type=int,default=4);args=ap.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1);random.seed(20261028);np.random.seed(20261028);torch.manual_seed(20261028)
 start=time.perf_counter();ds=Tiles();m=model();initial=ROOT/'.benchmark-data/yolox_nano_coco.pth';state=torch.load(initial,map_location='cpu',weights_only=True)['model'];current=m.state_dict();selected={k:v for k,v in state.items() if k in current and v.shape==current[k].shape};m.load_state_dict(selected,strict=False)
 opt=torch.optim.AdamW(m.parameters(),lr=.0006,weight_decay=.0001);out=ROOT/'.benchmark-data/v6-models';out.mkdir(exist_ok=True);history=[];weights=torch.tensor([2.5 if r['group'] in ('pahala','virden','ruttan') else 1. for r in ds.rows],dtype=torch.double)
 for epoch in range(args.epochs):
  sampler=torch.utils.data.WeightedRandomSampler(weights,args.steps_per_epoch*args.batch,replacement=True);loader=torch.utils.data.DataLoader(ds,batch_size=args.batch,sampler=sampler,num_workers=0);m.train();m.head.use_l1=epoch>=args.epochs-4;totals=[];t=time.perf_counter()
  for step,(x,y) in enumerate(loader):
   index=epoch*args.steps_per_epoch+step;lr=.0006*min(1.,(index+1)/80)*(.15+.85*(1+math.cos(math.pi*epoch/args.epochs))/2)
   for g in opt.param_groups:g['lr']=lr
   loss=m(x,y);value=loss['total_loss'];assert torch.isfinite(value),str(loss);opt.zero_grad(set_to_none=True);value.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step();totals.append(float(value.detach()))
   if step%24==0:print('epoch',epoch+1,'step',step,'loss',round(totals[-1],4),'elapsed',round(time.perf_counter()-start,1),flush=True)
  row={'epoch':epoch+1,'mean_loss':float(np.mean(totals)),'seconds':round(time.perf_counter()-t,3),'lr':lr};history.append(row);print('EPOCH',row,flush=True)
  if (epoch+1)%4==0:torch.save({'model':m.state_dict(),'epoch':epoch+1,'history':history},out/f'epoch-{epoch+1}.pth')
  (ROOT/'evaluation/v6/TRAINING.json').write_text(json.dumps({'method':'YOLOX-Nano 1 generic symbol class; COCO pretrained official weights; all layers fine-tuned on CPU','seed':20261028,'torch':torch.__version__,'opencv':cv2.__version__,'train_tiles':len(ds),'batch':args.batch,'steps_per_epoch':args.steps_per_epoch,'configured_epochs':args.epochs,'weighted_sampling':'Pahala/Virden/Ruttan annotated ROI tiles 2.5x relative weight, other tiles 1x; replacement','augmentation':'horizontal flip, contrast/brightness, mild blur; box transform applied; no test data','initial_weights_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'upstream_commit':'6ddff4824372906469a7fae2dc3206c7aa4bbaee','copied_pretrained_tensors':len(selected),'history':history,'training_seconds_so_far':round(time.perf_counter()-start,3)},indent=2)+'\n')
 print('TRAINED',round(time.perf_counter()-start,2),'seconds',flush=True)
if __name__=='__main__':main()
