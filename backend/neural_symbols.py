"""Bounded high-resolution YOLOX-Nano inference via the existing OpenCV CPU runtime."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import hashlib,json,math
import cv2,numpy as np

MODEL_DIR=Path(__file__).parent/'assets'
MAX_SOURCE_SIDE=4800
MAX_TILES=36
_LOCK=Lock()

@lru_cache(maxsize=1)
def model():
 path=MODEL_DIR/'symbol-neural.onnx';manifest=MODEL_DIR/'symbol-neural.json'
 if not manifest.exists():return None,None
 meta=json.loads(manifest.read_text())
 if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['weights_sha256']:raise ValueError('신경망 기호 모델의 검증 값이 맞지 않습니다.')
 net=cv2.dnn.readNetFromONNX(str(path));net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV);net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
 return net,meta


def tiles(width,height,size=1024,overlap=.2):
 """Cover the complete bounded source; right/bottom edges never lose coverage."""
 if max(width,height)>MAX_SOURCE_SIDE:raise ValueError('Neural source must be bounded before inference')
 step=max(1,round(size*(1-overlap)))
 def positions(length):
  if length<=size:return [0]
  return sorted(set(list(range(0,length-size+1,step))+[length-size]))
 result=[(x,y,min(width,x+size),min(height,y+size)) for y in positions(height) for x in positions(width)]
 if len(result)>MAX_TILES:raise ValueError('Neural tile budget exceeded')
 return result


def preprocess(image,size=640):
 h,w=image.shape[:2];ratio=min(size/w,size/h);rw,rh=int(w*ratio),int(h*ratio)
 canvas=np.full((size,size,3),114,np.uint8);canvas[:rh,:rw]=cv2.resize(image,(rw,rh),interpolation=cv2.INTER_LINEAR)
 return cv2.dnn.blobFromImage(canvas,1.,(size,size),swapRB=False,crop=False),ratio


def foreground_integral(bgr):
 """One exact RGB foreground pass shared by overlapping inference crops."""
 foreground=(cv2.inRange(bgr,(249,249,249),(255,255,255))==0).astype(np.uint8)
 return cv2.integral(foreground,sdepth=cv2.CV_32S)


def decode(raw,size=640):
 grids=[];strides=[]
 for stride in (8,16,32):
  y,x=np.meshgrid(np.arange(size//stride),np.arange(size//stride),indexing='ij');grids.append(np.stack((x,y),axis=-1).reshape(-1,2));strides.append(np.full((x.size,1),stride))
 grid=np.concatenate(grids);stride=np.concatenate(strides);pred=np.asarray(raw).reshape(-1,6).copy()
 pred[:,:2]=(pred[:,:2]+grid)*stride;pred[:,2:4]=np.exp(np.clip(pred[:,2:4],-15,15))*stride
 return np.column_stack((pred[:,:2]-pred[:,2:4]/2,pred[:,:2]+pred[:,2:4]/2,pred[:,4]*pred[:,5]))


def merge(base,neural,mode='replace'):
 from .geometry import _iou
 if mode=='neural_only':return neural
 if mode=='add':return base+[(b,p) for b,p in neural if not any(_iou(b,a)>.4 for a,_ in base)]
 # High-confidence neural extents replace overlapping older candidates.
 retained=[(b,p) for b,p in base if not any(_iou(b,n)>.3 for n,_ in neural)]
 return retained+neural


def port_hints(bgr):
 """Geometry proposes zoom locations; only the trained network emits boxes."""
 h,w=bgr.shape[:2];scale=min(1.,2400/max(w,h));small=cv2.resize(bgr,None,fx=scale,fy=scale) if scale<1 else bgr;gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY);ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,7)
 horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,15),np.uint8));vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((15,1),np.uint8));residual=cv2.morphologyEx(cv2.subtract(ink,horizontal|vertical),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8));_,_,stats,_=cv2.connectedComponentsWithStats(residual);sh,sw=ink.shape;hints=[]
 for x,y,cw,ch,area in stats[1:]:
  if area<10 or min(cw,ch)<4 or max(cw,ch)>55 or max(cw,ch)/min(cw,ch)>5:continue
  cy,cx=y+ch//2,x+cw//2
  arms=[np.count_nonzero(horizontal[max(0,cy-2):min(sh,cy+3),max(0,x-16):max(0,x-2)]),np.count_nonzero(horizontal[max(0,cy-2):min(sh,cy+3),min(sw,x+cw+2):min(sw,x+cw+16)]),np.count_nonzero(vertical[max(0,y-16):max(0,y-2),max(0,cx-2):min(sw,cx+3)]),np.count_nonzero(vertical[min(sh,y+ch+2):min(sh,y+ch+16),max(0,cx-2):min(sw,cx+3)])]
  if not ((arms[0]>=5 and arms[1]>=5) or (arms[2]>=5 and arms[3]>=5)):continue
  hints.append(([x/sw,y/sh,(x+cw)/sw,(y+ch)/sh],min(max(min(arms[0:2]),min(arms[2:4])),30)))
 return sorted(hints,key=lambda z:(-z[1],-min((z[0][2]-z[0][0])*sw,(z[0][3]-z[0][1])*sh),z[0]))[:150]


def detect(image,threshold=None,meta_override=None,proposal_hints=None):
 net,meta=model()
 if net is None:return []
 meta={**meta,**(meta_override or {})};cut=meta['threshold'] if threshold is None else threshold
 source=image.copy();source.thumbnail((MAX_SOURCE_SIDE,MAX_SOURCE_SIDE));rgb=np.asarray(source);h,w=rgb.shape[:2];bgr=rgb[:,:,::-1].copy();size=meta.get('input_size',640)
 border=np.concatenate((bgr[0],bgr[-1],bgr[:,0],bgr[:,-1]))
 if np.median(border)<128:bgr=255-bgr
 integral=foreground_integral(bgr)
 regions=tiles(w,h,meta.get('tile_size',1024),meta.get('overlap',.2));regions=[(*r,'tile') for r in regions]
 if max(w,h)>meta.get('tile_size',1024):regions.append((0,0,w,h,'overview'))
 found=[];detail_count=0
 def infer_region(region):
  x,y,right,bottom,kind=region
  crop=bgr[y:bottom,x:right]
  # Only empty background is skipped; faint content is retained.
  count=int(integral[bottom,right]-integral[y,right]-integral[bottom,x]+integral[y,x])
  if count/((bottom-y)*(right-x))<.0001:return
  blob,ratio=preprocess(crop,size)
  with _LOCK:net.setInput(blob);raw=net.forward()
  low=min(cut,.08) if meta.get('detail_tiles',0) else cut
  pred=decode(raw,size);pred=pred[np.isfinite(pred).all(1)&(pred[:,4]>=low)]
  for a,b,c,d,score in pred:
   a,c=np.clip([a/ratio,c/ratio],0,right-x)+x;b,d=np.clip([b/ratio,d/ratio],0,bottom-y)+y
   if min(c-a,d-b)<3:continue
   found.append(([float(a/w),float(b/h),float(c/w),float(d/h)],float(score),kind))
 for region in regions:infer_region(region)
 # Revisit small uncertain objects at twice the local sampling resolution.
 # Seeds are predictions only; no filename, project, annotation or OCR ID.
 budget=min(int(meta.get('detail_tiles',0)),max(0,MAX_TILES+1-len(regions))) if max(w,h)>512 else 0;details=[]
 seeds=sorted([(b,s) for b,s,k in found if .08<=s<meta.get('detail_seed_max',.6) and 6<=min((b[2]-b[0])*w,(b[3]-b[1])*h) and max((b[2]-b[0])*w,(b[3]-b[1])*h)<=90],key=lambda z:(-z[1],z[0]))
 hints=port_hints(bgr) if budget and meta.get('detail_hint_mode')=='ports' else []
 # Allocate half the small budget to visible two-sided pipe contacts, including
 # glyphs that generated no low-confidence network proposal at all.
 if budget and meta.get('detail_hint_mode')=='hybrid':
  from .geometry import _iou
  accepted=[b for b,s,_ in found if s>=cut]
  resolved=proposal_hints() if callable(proposal_hints) else proposal_hints
  missing=[b for b in (resolved or []) if not any(_iou(b,a)>.25 for a in accepted) and 5<=min((b[2]-b[0])*w,(b[3]-b[1])*h)<=85]
  # Independent geometry can seed glyphs with no detector proposal at all.
  hints=[(b,1.) for b in missing[:40]]
 seeds=[(b,1.) for b,_ in hints[:max(3,budget//2)*4]]+seeds if hints else seeds
 geometric_budget=max(0,budget//2);geometric_count=0
 for box,score in seeds:
  if len(details)>=budget:break
  if score==1. and geometric_count>=geometric_budget:continue
  cx,cy=(box[0]+box[2])*w/2,(box[1]+box[3])*h/2;tile=int(np.clip(meta.get('detail_tile_size',512),256,640));x=int(np.clip(cx-tile/2,0,max(0,w-tile)));y=int(np.clip(cy-tile/2,0,max(0,h-tile)))
  if any(abs(x-a)<tile*.65 and abs(y-b)<tile*.65 for a,b,_,_,_ in details):continue
  details.append((x,y,min(w,x+tile),min(h,y+tile),'detail'))
  if score==1.:geometric_count+=1
 for region in details:infer_region(region)
 detail_count=len(details);regions+=details;found=[(b,s,k) for b,s,k in found if s>=cut]
 if not found:return []
 boxes=[[b[0]*w,b[1]*h,(b[2]-b[0])*w,(b[3]-b[1])*h] for b,_,_ in found];scores=[s for _,s,_ in found]
 base_ids=[i for i,(_,_,k) in enumerate(found) if k!='detail'];detail_ids=[i for i,(_,_,k) in enumerate(found) if k=='detail']
 def suppress(indices):
  if not indices:return []
  kept=cv2.dnn.NMSBoxes([boxes[i] for i in indices],[scores[i] for i in indices],float(cut)-1e-5,.4,top_k=1000)
  return [indices[int(j)] for j in np.asarray(kept).ravel()]
 kept_base=suppress(base_ids)
 # Refinement adds missing glyphs; it cannot replace good base localization
 # or push existing candidates out of the fixed200-symbol budget.
 from .geometry import _iou
 detail_ids=[i for i in detail_ids if not any(_iou(found[i][0],found[j][0])>.2 for j in kept_base)]
 kept=(kept_base+suppress(detail_ids))[:200];result=[]
 for i in kept:
  box,score,kind=found[int(i)];result.append((box,{'geometry':'neural_symbol','model':meta.get('model','yolox-nano-symbol-v1'),'symbol_detector':'yolox-nano-symbol-v1','symbol_model_sha256':meta['weights_sha256'],'detector_score':score,'requires_review':True,'inference_view':kind,'source_size':[w,h],'tile_count':len(regions),'detail_tile_count':detail_count}))
 return result
