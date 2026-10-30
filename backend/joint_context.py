"""Shared visual encoder for port, junction and local line-style evidence."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import hashlib,json
import cv2,numpy as np
DIR=Path(__file__).parent/'assets';LOCK=Lock()
@lru_cache(maxsize=1)
def model():
    p=DIR/'joint-context.json'
    if not p.exists():return None,None
    meta=json.loads(p.read_text());weights=DIR/meta['file']
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('공동 문맥 모델 검증 값 불일치')
    return cv2.dnn.readNetFromONNX(str(weights)),meta

def patches(gray,ink,points):
    out=[]
    for p in points:
        x,y=map(float,p['pixel']);rot={'right':0,'bottom':1,'left':2,'top':3}.get(p.get('side'),0);local=cv2.getRectSubPix(gray,(25,25),(x,y));context=cv2.getRectSubPix(gray,(49,49),(x,y));binary=cv2.getRectSubPix(ink,(49,49),(x,y));channels=[1-cv2.resize(local,(32,32)).astype(np.float32)/255,1-cv2.resize(context,(32,32)).astype(np.float32)/255,cv2.resize(binary,(32,32)).astype(np.float32)/255];out.append(np.stack([np.rot90(c,rot).copy() for c in channels]))
    return np.asarray(out,np.float32).reshape(-1,3,32,32)

def softmax(x):
    x=x-x.max(1,keepdims=True);e=np.exp(x);return e/e.sum(1,keepdims=True)

def predict(gray,ink,points):
    net,meta=model()
    if net is None:return None
    xx=patches(gray,ink,points);parts=[]
    # Fixed padded batches keep the OpenCV graph small and deterministic.
    batch=int(meta.get('batch_size',64))
    for i in range(0,len(xx),batch):
        block=xx[i:i+batch];padded=np.zeros((batch,3,32,32),np.float32);padded[:len(block)]=block
        with LOCK:net.setInput(padded);result=net.forward().reshape(batch,9)[:len(block)]
        parts.append(result)
    output=np.concatenate(parts) if parts else np.zeros((0,9),np.float32)
    return {'port':softmax(output[:,:2]),'junction':softmax(output[:,2:6]),'line':softmax(output[:,6:9]),'model_sha256':meta['sha256']}
