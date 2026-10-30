"""Missing-example fine-tune of the deployed v10/v11 single generic detector."""
import json,hashlib,sys,time,random,math
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v9.train import model
cv2.setNumThreads(1);torch.set_num_threads(4);torch.set_num_interop_threads(1);random.seed(20261030);np.random.seed(20261030);torch.manual_seed(20261030)
rows=json.loads((ROOT/'evaluation/v12/TRAINING-DATA.json').read_text())['tiles'];cache=[cv2.imread(str(ROOT/r['file'])) for r in rows]
class Data(torch.utils.data.Dataset):
 def __len__(self):return len(rows)
 def __getitem__(self,i):
  im=cache[i].copy();b=np.array(rows[i]['boxes'],np.float32).reshape(-1,4)
  if random.random()<.5:im=im[:,::-1].copy();b[:,[0,2]]=640-b[:,[2,0]]
  if rows[i]['group']=='white-rose' and random.random()<.5:
   im=np.rot90(im).copy();b=np.column_stack((b[:,1],640-b[:,2],b[:,3],640-b[:,0]))
  if random.random()<.6:im=np.clip(255-(255-im.astype(np.float32))*random.uniform(.55,1.1)+random.uniform(-8,8),0,255).astype(np.uint8)
  if random.random()<.1:im=cv2.GaussianBlur(im,(3,3),.5)
  t=np.zeros((150,5),np.float32);b=b[:150];t[:len(b),1:3]=(b[:,:2]+b[:,2:])/2;t[:len(b),3:]=b[:,2:]-b[:,:2]
  return torch.from_numpy(im.transpose(2,0,1).copy()).float(),torch.from_numpy(t)
paths=[ROOT/'.benchmark-data/v6-models/epoch-16.pth',ROOT/'.benchmark-data/v9b-models/epoch-8.pth'];states=[torch.load(p,map_location='cpu',weights_only=True)['model'] for p in paths];initial={k:(states[0][k]*.5+states[1][k]*.5 if states[0][k].is_floating_point() else states[1][k]) for k in states[0]};m=model();m.load_state_dict(initial);out=ROOT/'.benchmark-data/v12-models';out.mkdir(exist_ok=True);torch.save({'model':initial},out/'initial.pth');opt=torch.optim.AdamW(m.parameters(),lr=.00006,weight_decay=.0001);weights=torch.tensor([r['sampling_weight'] for r in rows],dtype=torch.double);start=time.perf_counter();history=[]
for epoch in range(8):
 ds=Data();sampler=torch.utils.data.WeightedRandomSampler(weights,32*4,replacement=True);loader=torch.utils.data.DataLoader(ds,batch_size=4,sampler=sampler,num_workers=0);m.train();m.head.use_l1=epoch>=4;losses=[];t=time.perf_counter()
 for step,(x,y) in enumerate(loader):
  loss=m(x,y)['total_loss'];assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step();losses.append(float(loss.detach()))
  if step%16==0:print('epoch',epoch+1,'step',step,'loss',losses[-1],flush=True)
 history.append({'epoch':epoch+1,'mean_loss':float(np.mean(losses)),'seconds':time.perf_counter()-t});print(history[-1],flush=True)
 if epoch in (3,7):torch.save({'model':m.state_dict(),'epoch':epoch+1},out/f'epoch-{epoch+1}.pth')
 (ROOT/'evaluation/v12/TRAINING.json').write_text(json.dumps({'method':'Fine-tune deployed interpolation checkpoint with former White Rose test hard examples and v9 replay; all layers CPU','seed':20261030,'epochs':8,'steps_per_epoch':32,'batch':4,'learning_rate':.00006,'initial_endpoints':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],'new_project_exclusion':['fernald','epa-wtp','peterhead'],'training_manifest_sha256':hashlib.sha256((ROOT/'evaluation/v12/TRAINING-DATA.json').read_bytes()).hexdigest(),'history':history,'seconds_so_far':time.perf_counter()-start,'annotation_basis':'AI pilot, not expert-certified'},indent=2)+'\n')
