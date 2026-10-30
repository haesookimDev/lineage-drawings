"""Transfer former White Rose pilot to training, crop complete labeled ROIs only."""
import json,hashlib,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import match_boxes
rng=np.random.default_rng(20261030);out=ROOT/'.benchmark-data/v12-tiles';out.mkdir(exist_ok=True);base=json.loads((ROOT/'evaluation/v9/TRAINING-DATA.json').read_text());rows=[{**r,'sampling_weight':1.} for r in base['tiles']];sources=[]
for key,s in json.loads((ROOT/'evaluation/v11/white-rose-labels.json').read_text())['sheets'].items():
 path=ROOT/'.benchmark-data/v11-held/white-rose'/(s['id']+'-4800.png');im=Image.open(path).convert('RGB');w,h=im.size;draw=ImageDraw.Draw(im)
 for box in s['ignore']:draw.rectangle(tuple(round(v*[w,h][i%2]) for i,v in enumerate(box)),fill='white')
 roi=np.round(np.array(s['roi'])*[w,h,w,h]).astype(int);im=im.crop(tuple(roi));a=np.asarray(im)[:,:,::-1].copy();hh,ww=a.shape[:2];boxes=np.array([n['bbox'] for n in s['nodes']])*[w,h,w,h]-np.tile(roi[:2],2)
 pred=json.loads((ROOT/f'evaluation/v11/{key}-v11-geometry.json').read_text())['elements'];matched=set(match_boxes(pred,s['nodes'],.5).values());hard={n['id'] for n in s['nodes']}-matched
 crops=[(0,0,ww,hh,False)]
 for node,box in zip(s['nodes'],boxes):
  cx,cy=(box[:2]+box[2:])/2
  for side in (384,640,1024):
   x=int(np.clip(cx-side*rng.uniform(.3,.7),0,max(0,ww-side)));y=int(np.clip(cy-side*rng.uniform(.3,.7),0,max(0,hh-side)));crops.append((x,y,min(ww,x+side),min(hh,y+side),node['id'] in hard))
 for i,(x,y,right,bottom,is_hard) in enumerate(crops):
  crop=a[y:bottom,x:right].copy();bb=boxes-np.tile([x,y],2);clipped=bb.copy();clipped[:,0::2]=clipped[:,0::2].clip(0,right-x);clipped[:,1::2]=clipped[:,1::2].clip(0,bottom-y);overlap=np.maximum(0,clipped[:,2:]-clipped[:,:2]).prod(1)/np.maximum(1,(boxes[:,2:]-boxes[:,:2]).prod(1));keep=(overlap>=.7)&((clipped[:,2:]-clipped[:,:2]).min(1)>=3)
  for b in clipped[(overlap>0)&~keep]:
   p,q,r,t=np.round(b).astype(int);crop[q:t+1,p:r+1]=255
  selected=clipped[keep];ratio=min(640/crop.shape[1],640/crop.shape[0]);small=cv2.resize(crop,(round(crop.shape[1]*ratio),round(crop.shape[0]*ratio)));canvas=np.full((640,640,3),114,np.uint8);canvas[:small.shape[0],:small.shape[1]]=small;selected*=ratio
  p=out/f'{key}-{i}.png';cv2.imwrite(str(p),canvas);rows.append({'file':str(p.relative_to(ROOT)),'group':'white-rose','id':s['id'],'boxes':selected.tolist(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':'former-v11-test-now-training','sampling_weight':3. if is_hard else 1.5})
 sources.append({'project':'white-rose','page':s['id'],'previous_role':'v11 pilot -> all training in v12','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'symbols':len(boxes),'previous_missed_or_iou_failed':sorted(hard),'annotation_basis':'single AI source annotation, no expert certification'})
result={'seed':20261030,'tiles':rows,'training_projects':'Original v9 replay projects plus White Rose. No Fernald/EPA/Peterhead pixels or labels. No operational uploads.','sources':sources,'replay_tiles':len(base['tiles']),'new_tiles':len(rows)-len(base['tiles']),'labels_sha256':hashlib.sha256((ROOT/'evaluation/v11/white-rose-labels.json').read_bytes()).hexdigest()};(ROOT/'evaluation/v12/TRAINING-DATA.json').write_text(json.dumps(result,indent=2)+'\n');print({'tiles':len(rows),'sources':sources},flush=True)
