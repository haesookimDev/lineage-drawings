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


def detect(image,threshold=None,meta_override=None):
 net,meta=model()
 if net is None:return []
 meta={**meta,**(meta_override or {})};cut=meta['threshold'] if threshold is None else threshold
 source=image.copy();source.thumbnail((MAX_SOURCE_SIDE,MAX_SOURCE_SIDE));rgb=np.asarray(source);h,w=rgb.shape[:2];bgr=rgb[:,:,::-1].copy();size=meta.get('input_size',640)
 border=np.concatenate((bgr[0],bgr[-1],bgr[:,0],bgr[:,-1]))
 if np.median(border)<128:bgr=255-bgr
 regions=tiles(w,h,meta.get('tile_size',1024),meta.get('overlap',.2));regions=[(*r,'tile') for r in regions]
 if max(w,h)>meta.get('tile_size',1024):regions.append((0,0,w,h,'overview'))
 found=[]
 for x,y,right,bottom,kind in regions:
  crop=bgr[y:bottom,x:right]
  # Only empty background is skipped; faint content is retained.
  if float((crop.min(axis=2)<249).mean())<.0001:continue
  blob,ratio=preprocess(crop,size)
  with _LOCK:net.setInput(blob);raw=net.forward()
  pred=decode(raw,size);pred=pred[np.isfinite(pred).all(1)&(pred[:,4]>=cut)]
  for a,b,c,d,score in pred:
   a,c=np.clip([a/ratio,c/ratio],0,right-x)+x;b,d=np.clip([b/ratio,d/ratio],0,bottom-y)+y
   if min(c-a,d-b)<3:continue
   found.append(([float(a/w),float(b/h),float(c/w),float(d/h)],float(score),kind))
 if not found:return []
 boxes=[[b[0]*w,b[1]*h,(b[2]-b[0])*w,(b[3]-b[1])*h] for b,_,_ in found];scores=[s for _,s,_ in found]
 kept=cv2.dnn.NMSBoxes(boxes,scores,float(cut)-1e-5,.4,top_k=1000);result=[]
 for i in np.asarray(kept).ravel()[:200]:
  box,score,kind=found[int(i)];result.append((box,{'geometry':'neural_symbol','model':'yolox-nano-symbol-v1','symbol_detector':'yolox-nano-symbol-v1','symbol_model_sha256':meta['weights_sha256'],'detector_score':score,'requires_review':True,'inference_view':kind,'source_size':[w,h],'tile_count':len(regions)}))
 return result
