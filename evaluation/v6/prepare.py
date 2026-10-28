"""Create bounded labeled training tiles, preserving source provenance."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v6.data import sources,read

def main():
 out=ROOT/'.benchmark-data/v6-tiles';out.mkdir(exist_ok=True);rng=np.random.default_rng(20261028);records=[];source_records=[]
 for group,ident,region in sources('training'):
  im,boxes=read(group,ident,region);w,h=im.size
  if region:
   draw=ImageDraw.Draw(im)
   for ig in region.get('ignore',[]):draw.rectangle([round(v*[w,h][i%2]) for i,v in enumerate(ig)],fill='white')
   roi=np.array([round(v*[w,h][i%2]) for i,v in enumerate(region['roi'])]);im=im.crop(tuple(roi));boxes-=np.tile(roi[:2],2);w,h=im.size
  image=np.asarray(im)[:,:,::-1].copy();tiles=[]
  # Full labeled ROI and centered patches. Context outside partial-label ROIs
  # never becomes a false background target.
  if region:tiles=[(0,0,w,h)]*4
  for a,b,c,d in boxes:
   size=int(rng.choice([640,896,1152]));cx,cy=(a+c)/2,(b+d)/2;x=int(np.clip(cx-size*rng.uniform(.3,.7),0,max(0,w-size)));y=int(np.clip(cy-size*rng.uniform(.3,.7),0,max(0,h-size)));tiles.append((x,y,min(w,x+size),min(h,y+size)))
  rng.shuffle(tiles);tiles=tiles[:32] if not region else tiles[:40]
  used=0
  for j,(a,b,c,d) in enumerate(tiles):
   crop=image[b:d,a:c];bb=boxes-np.tile([a,b],2);bb[:,0::2]=bb[:,0::2].clip(0,c-a);bb[:,1::2]=bb[:,1::2].clip(0,d-b)
   inter=(bb[:,2:]-bb[:,:2]).clip(0).prod(1);area=(boxes[:,2:]-boxes[:,:2]).prod(1);keep=(inter/np.maximum(1,area)>.7)&((bb[:,2:]-bb[:,:2]).min(1)>=4);bb=bb[keep]
   ratio=min(640/crop.shape[1],640/crop.shape[0]);resized=cv2.resize(crop,(int(crop.shape[1]*ratio),int(crop.shape[0]*ratio)));canvas=np.full((640,640,3),114,np.uint8);canvas[:resized.shape[0],:resized.shape[1]]=resized;bb*=ratio
   if not len(bb):continue
   path=out/f'{group}-{ident}-{j}.png';cv2.imwrite(str(path),canvas);records.append({'file':str(path.relative_to(ROOT)),'group':group,'id':ident,'boxes':bb.tolist(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});used+=1
  source_records.append({'group':group,'id':ident,'symbols':len(boxes),'tiles':used,'source_image_sha256':hashlib.sha256((ROOT/'.benchmark-data'/group/(ident+'.png')).read_bytes()).hexdigest()});print(group,ident,'symbols',len(boxes),'tiles',used,flush=True)
 (ROOT/'evaluation/v6/TRAINING-DATA.json').write_text(json.dumps({'seed':20261028,'sources':source_records,'tiles':records,'note':'Derived training-only crops, not independent images; partial real labels restricted to labeled ROIs. Previous v5 test regions explicitly reclassified as training. No new-page test files loaded.'},indent=2)+'\n')
 print('total',len(records),'tiles',sum(r['symbols'] for r in source_records),'source symbols',flush=True)
if __name__=='__main__':main()
