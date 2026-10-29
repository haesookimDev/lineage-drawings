"""Replay glyph training plus old test ROIs, never new-project pixels."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
rng=np.random.default_rng(20261029);out=ROOT/'.benchmark-data/v9-tiles';out.mkdir(exist_ok=True);base=json.loads((ROOT/'evaluation/v6/TRAINING-DATA.json').read_text());rows=list(base['tiles']);sources=list(base['sources']);added=[]
for version,folder in [('v6','v6-highres'),('v7','v7-highres'),('v8','v8-pages')]:
 for key,s in json.loads((ROOT/f'evaluation/{version}/new-page-labels.json').read_text())['sheets'].items():
  g,i=s['group'],s['id'];name=i+'-4800.png' if version=='v8' else i+'.png';path=ROOT/'.benchmark-data'/folder/g/name
  with Image.open(path) as raw:im=raw.convert('RGB');im.thumbnail((4800,4800))
  w,h=im.size;draw=ImageDraw.Draw(im)
  for ig in s.get('ignore',[]):draw.rectangle([round(v*[w,h][k%2]) for k,v in enumerate(ig)],fill='white')
  roi=np.array([round(v*[w,h][k%2]) for k,v in enumerate(s['roi'])]);im=im.crop(tuple(roi));boxes=np.array([[v*[w,h][k%2] for k,v in enumerate(n['bbox'])] for n in s['nodes']],np.float32)-np.tile(roi[:2],2);w,h=im.size;arr=np.array(im)[:,:,::-1].copy();tiles=[(0,0,w,h)]*4
  for box in boxes:
   for size in (384,640,896):
    cx,cy=(box[:2]+box[2:])/2;x=int(np.clip(cx-size*rng.uniform(.3,.7),0,max(0,w-size)));y=int(np.clip(cy-size*rng.uniform(.3,.7),0,max(0,h-size)));tiles.append((x,y,min(w,x+size),min(h,y+size)))
  seen=set();used=0
  for k,(a,b,c,d) in enumerate(tiles):
   if (a,b,c,d) in seen and k>=4:continue
   seen.add((a,b,c,d));crop=arr[b:d,a:c];bb=boxes-np.tile([a,b],2);bb[:,0::2]=bb[:,0::2].clip(0,c-a);bb[:,1::2]=bb[:,1::2].clip(0,d-b);inter=np.maximum(0,bb[:,2:]-bb[:,:2]).prod(1);area=(boxes[:,2:]-boxes[:,:2]).prod(1);bb=bb[(inter/np.maximum(1,area)>.7)&((bb[:,2:]-bb[:,:2]).min(1)>=4)];ratio=min(640/crop.shape[1],640/crop.shape[0]);resize=cv2.resize(crop,(int(crop.shape[1]*ratio),int(crop.shape[0]*ratio)));canvas=np.full((640,640,3),114,np.uint8);canvas[:resize.shape[0],:resize.shape[1]]=resize;bb*=ratio
   if not len(bb):continue
   p=out/f'{g}-{i}-{k}.png';cv2.imwrite(str(p),canvas);rows.append({'file':str(p.relative_to(ROOT)),'group':g,'id':i,'boxes':bb.tolist(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':'former-test-now-training'});used+=1
  record={'group':g,'id':i,'symbols':len(boxes),'tiles':used,'source_image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'previous_role':version+' test -> glyph training'};added.append(record);print(record,flush=True)
assert all(x['group'] not in ('winkler','winnipeg') and 'v9-held' not in x['file'] for x in rows)
(ROOT/'evaluation/v9/TRAINING-DATA.json').write_text(json.dumps({'seed':20261029,'sources':sources+added,'tiles':rows,'note':'Replay original glyph training plus explicitly transferred73 previous-test boxes. ROI outside/ignored areas are not negatives. No v8r2 validation3 pages nor new projects used.'},indent=2)+'\n');print('tiles',len(rows),'new symbols',sum(x['symbols'] for x in added))
