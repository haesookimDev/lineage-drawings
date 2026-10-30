"""Joint port/junction/local-line model; real training points + exact scenes."""
import json,hashlib,sys,time,random
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend.joint_context import patches
from backend.topology_context import prepare
from evaluation.v7.procedural import scene
cv2.setNumThreads(1);torch.set_num_threads(4);torch.manual_seed(20261030);rng=np.random.default_rng(20261030);data=[];labels=[];sources=[]
def add(gray,ink,pts,target):
    if not pts:return
    data.extend(patches(gray,ink,pts));labels.extend(target)
manual=json.loads((ROOT/'evaluation/v8r2/manual-development.json').read_text())['sheets']
for key,s in manual.items():
    if s['role']!='training':continue
    p=ROOT/'.benchmark-data'/s['group']/(s['id']+'.png');gray=cv2.imread(str(p),0);_,ink=prepare(__import__('PIL.Image',fromlist=['Image']).fromarray(gray).convert('RGB'));h,w=gray.shape;truth={n['id']:[] for n in s['nodes']}
    for port in s['ports']:truth[port['element_id']].append(np.array(port['point'])*[w,h])
    pts=[];yy=[]
    for e in s['nodes']:
        a,b,c,d=np.array(e['bbox'])*[w,h,w,h]
        for side in ('left','right','top','bottom'):
            for t in np.linspace(.05,.95,8):
                xy=np.array([a if side=='left' else c,b+t*(d-b)]) if side in ('left','right') else np.array([a+t*(c-a),b if side=='top' else d])
                if min((np.linalg.norm(xy-q) for q in truth[e['id']]),default=999)<8:continue
                pts.append({'pixel':xy.tolist(),'side':side});yy.append([0,-1,-1])
        for q in truth[e['id']]:
            side=min(('left','right','top','bottom'),key=lambda v:abs(q[0]-(a if v=='left' else c)) if v in ('left','right') else abs(q[1]-(b if v=='top' else d)))
            for _ in range(5):pts.append({'pixel':(q+rng.uniform(-1,1,2)).tolist(),'side':side});yy.append([1,-1,-1])
    add(gray,ink,pts,yy);pts=[];yy=[]
    for point in s.get('junctions',[])+s.get('point_samples',[]):
        for _ in range(8):pts.append({'pixel':(np.array(point['point'])*[w,h]+rng.uniform(-1,1,2)).tolist()});yy.append([-1,{'none':0,'tee':1,'dot':2,'plain_cross':3}[point['type']],-1])
    add(gray,ink,pts,yy);sources.append({'page':key,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ports':len(s['ports']),'junctions':len(s.get('junctions',[])+s.get('point_samples',[]))})
for seed in range(20301030,20301230):
    gray,prompt,target=scene(seed);ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,7);pts=[];yy=[]
    for ch in (2,3,4,5):
        _,_,_,centers=cv2.connectedComponentsWithStats(target[ch])
        for xy in centers[1:]:
            for _ in range(2):pts.append({'pixel':(xy+rng.uniform(-1.5,1.5,2)).tolist()});yy.append([1,-1,-1] if ch==2 else [-1,ch-2,-1])
    for ch in (0,1):
        yx=np.argwhere(target[ch]>0)
        if len(yx):
            for index in rng.choice(len(yx),min(12,len(yx)),replace=False):pts.append({'pixel':yx[index][::-1].tolist()});yy.append([-1,-1,ch+1])
    for _ in range(8):
        x,y=map(int,rng.integers(16,240,2))
        if target[:,y-6:y+7,x-6:x+7].max()==0:pts.append({'pixel':[x,y]});yy.append([0,0,0])
    add(gray,ink,pts,yy)
class Joint(nn.Module):
    def __init__(self):
        super().__init__();self.body=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(24,32,3,padding=1),nn.ReLU(),nn.AvgPool2d(8));self.head=nn.Linear(32,9)
    def forward(self,x):return self.head(self.body(x).flatten(1))
x=torch.tensor(np.array(data));y=torch.tensor(labels);net=Joint();opt=torch.optim.AdamW(net.parameters(),lr=.001);start=time.perf_counter();history=[];frequencies={};weights=np.zeros(len(y),float)
for col in range(3):
    yy=y[:,col].numpy();valid=yy>=0;counts=np.bincount(yy[valid]);frequencies[str(col)]=counts.tolist()
    for k,n in enumerate(counts):weights[yy==k]+=1/max(n,1)
for epoch in range(10):
    net.train();sampler=torch.utils.data.WeightedRandomSampler(torch.tensor(weights),min(8192,len(x)),replacement=True);loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=128,sampler=sampler);values=[]
    for xx,yy in loader:
        out=net(xx);loss=0.
        for col,(a,b) in enumerate(((0,2),(2,6),(6,9))):
            keep=yy[:,col]>=0
            if keep.any():loss=loss+nn.functional.cross_entropy(out[keep,a:b],yy[keep,col])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();values.append(float(loss.detach()))
    history.append({'epoch':epoch+1,'loss':float(np.mean(values))});print(history[-1],flush=True)
net.eval();out=ROOT/'.benchmark-data/v12-models/joint-context.onnx';dummy=torch.zeros((64,3,32,32));torch.onnx.export(net,dummy,str(out),input_names=['patches'],output_names=['scores'],opset_version=11,dynamo=False)
with torch.no_grad():expected=net(dummy).numpy()
cvnet=cv2.dnn.readNetFromONNX(str(out));cvnet.setInput(dummy.numpy());actual=cvnet.forward().reshape(64,9);assert np.allclose(actual,expected,atol=.002,rtol=.002)
meta={'model':'shared-point-context-v12','file':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'batch_size':64,'heads':['port2','junction4','local_line3'],'annotation_basis':'AI reviewed real points + exact procedural scenes; no new-project sources','enabled_by_default':False};out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');(ROOT/'evaluation/v12/JOINT-TRAINING.json').write_text(json.dumps({'model':meta,'rows':len(x),'class_counts':frequencies,'sources':sources,'procedural_seeds':[20301030,20301229],'history':history,'seconds':time.perf_counter()-start,'opencv_torch_max_abs_difference':float(np.abs(actual-expected).max())},indent=2)+'\n')
