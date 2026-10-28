"""Learned line-style, port and junction maps. No semantic/flow approval."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import hashlib,json
import cv2,numpy as np
from .network import line_ink
MODEL_DIR=Path(__file__).parent/'assets'
CHANNELS=['solid','dashed','port','tee','dot','plain_cross']
_LOCK=Lock()

@lru_cache(maxsize=1)
def model():
 meta_path=MODEL_DIR/'topology.json';path=MODEL_DIR/'topology.onnx'
 if not meta_path.exists():return None,None
 meta=json.loads(meta_path.read_text())
 if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['weights_sha256']:raise ValueError('선·포트 모델의 검증 값이 맞지 않습니다.')
 net=cv2.dnn.readNetFromONNX(str(path));net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV);net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU);return net,meta


def maps(image,elements):
 net,meta=model()
 if net is None:return None,{}
 cv2.setNumThreads(2);gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY);h,w=gray.shape
 if max(h,w)>2400:raise ValueError('Topology view must be bounded to2400px')
 border=np.concatenate((gray[0],gray[-1],gray[:,0],gray[:,-1]));gray=255-gray if np.median(border)<128 else gray;ink=line_ink(gray);prompt=np.zeros((h,w),np.uint8)
 for e in elements:
  if e['kind']=='text' or not e.get('properties',{}).get('connection_eligible',True):continue
  a,b,c,d=[round(v*[w,h][i%2]) for i,v in enumerate(e['bbox'])];cv2.rectangle(prompt,(a,b),(c,d),1,-1)
 size=384;step=320
 def positions(n):return [0] if n<=size else sorted(set(list(range(0,n-size+1,step))+[n-size]))
 regions=[(x,y,min(w,x+size),min(h,y+size)) for y in positions(h) for x in positions(w)]
 if len(regions)>64:raise ValueError('Topology patch budget exceeded')
 acc=np.zeros((6,h,w),np.uint16);count=np.zeros((h,w),np.uint8)
 for x,y,c,d in regions:
  hh,ww=d-y,c-x;input=np.zeros((1,3,size,size),np.float32);input[0,0,:hh,:ww]=1-gray[y:d,x:c].astype(np.float32)/255;input[0,1,:hh,:ww]=ink[y:d,x:c].astype(np.float32)/255;input[0,2,:hh,:ww]=prompt[y:d,x:c]
  with _LOCK:net.setInput(input);raw=net.forward()[0,:,:hh,:ww]
  probs=np.round(255/(1+np.exp(-np.clip(raw,-15,15)))).astype(np.uint16);acc[:,y:d,x:c]+=probs;count[y:d,x:c]+=1
 result=np.zeros((6,h,w),np.uint8)
 for channel in range(6):result[channel]=(acc[channel]//np.maximum(count,1)).astype(np.uint8)
 return result,{'topology_model':'tiny-unet-line-port-v1','topology_model_sha256':meta['weights_sha256'],'recognition_tiles':len(regions),'map_quantization':'uint8 averaged logits-sigmoid maps; scores not calibrated probabilities'}
