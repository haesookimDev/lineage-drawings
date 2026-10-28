"""Train a tiny six-head line/port/junction U-Net, CPU; no test-page labels read."""
from pathlib import Path
import argparse,hashlib,json,random,sys,time
import cv2,numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v7.procedural import scene,CHANNELS

class Net(nn.Module):
 def __init__(self):
  super().__init__()
  def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())
  self.e0=block(3,8);self.e1=block(8,16);self.e2=block(16,24);self.mid=block(24,32);self.d2=block(56,24);self.d1=block(40,16);self.d0=block(24,8);self.head=nn.Conv2d(8,6,1);self.pool=nn.MaxPool2d(2)
 def forward(self,x):
  a=self.e0(x);b=self.e1(self.pool(a));c=self.e2(self.pool(b));d=self.mid(self.pool(c));up=lambda x:nn.functional.interpolate(x,scale_factor=2,mode='nearest');return self.head(self.d0(torch.cat([up(self.d1(torch.cat([up(self.d2(torch.cat([up(d),c],1))),b],1))),a],1)))

def input_tensor(gray,prompt):
 border=np.concatenate((gray[0],gray[-1],gray[:,0],gray[:,-1]));gray=255-gray if np.median(border)<128 else gray;ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,7)
 return np.stack((1-gray.astype(np.float32)/255,ink.astype(np.float32)/255,prompt.astype(np.float32)))

class Data(torch.utils.data.Dataset):
 def __init__(self):
  self.rows=json.loads((ROOT/'evaluation/v7/TRAINING-DATA.json').read_text())['patches'];self.cache=[dict(np.load(ROOT/r['file'])) for r in self.rows]
 def __len__(self):return 1600
 def __getitem__(self,i):
  if random.random()<.5:
   seed=2026102900+random.randrange(1200);im,prompt,m=scene(seed);weight=.9
  else:
   record=self.cache[random.randrange(len(self.cache))];im,prompt,m=record['image'].copy(),record['prompt'].copy(),record['target'].copy();weight=.45
   if random.random()<.5:im=np.fliplr(im).copy();prompt=np.fliplr(prompt).copy();m=np.flip(m,2).copy()
  return torch.from_numpy(input_tensor(im,prompt)),torch.from_numpy(m.astype(np.float32)),weight

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=16);ap.add_argument('--steps',type=int,default=100);args=ap.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20261029);np.random.seed(20261029);random.seed(20261029)
 model=Net();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);data=Data();out=ROOT/'.benchmark-data/v7-models';out.mkdir(exist_ok=True);start=time.perf_counter();history=[]
 for epoch in range(args.epochs):
  model.train();loader=torch.utils.data.DataLoader(data,batch_size=8,num_workers=0,shuffle=True);losses=[];t=time.perf_counter()
  for step,(x,y,weight) in enumerate(loader):
   if step>=args.steps:break
   pred=model(x);pw=torch.tensor([8,12,20,35,35,35],dtype=torch.float32).reshape(1,6,1,1);bce=nn.functional.binary_cross_entropy_with_logits(pred,y,pos_weight=pw,reduction='none').mean((1,2,3));p=pred.sigmoid();dice=1-(2*(p[:,:2]*y[:,:2]).sum((2,3))+1)/(p[:,:2].sum((2,3))+y[:,:2].sum((2,3))+1);loss=(bce*weight.float()).mean()+.35*dice.mean();assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
   if step%25==0:print('epoch',epoch+1,'step',step,'loss',round(losses[-1],4),'elapsed',round(time.perf_counter()-start,1),flush=True)
  row={'epoch':epoch+1,'mean_loss':float(np.mean(losses)),'seconds':round(time.perf_counter()-t,3)};history.append(row);print('EPOCH',row,flush=True)
  if (epoch+1)%4==0:torch.save({'model':model.state_dict(),'history':history},out/f'epoch-{epoch+1}.pth')
  (ROOT/'evaluation/v7/TRAINING.json').write_text(json.dumps({'architecture':'Tiny U-Net3-input/6-logit-heads, from scratch, no pretrained topology weights','channels':CHANNELS,'seed':20261029,'epochs':args.epochs,'steps_per_epoch':args.steps,'batch':8,'optimizer':'AdamW lr0.001 wd0.0001','training':'50% exact procedural/50% training-only weak GraphML patches; per-sample BCE weights0.9/0.45; solid/dash Dice0.35; positive BCE weights8/12/20/35/35/35','history':history,'seconds_so_far':round(time.perf_counter()-start,3),'torch':torch.__version__,'training_sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'evaluation/v7/procedural.py',ROOT/'evaluation/v7/graph_data.py',ROOT/'evaluation/v7/TRAINING-DATA.json']}},indent=2)+'\n')
 print('TRAINED',round(time.perf_counter()-start,2),flush=True)
if __name__=='__main__':main()
