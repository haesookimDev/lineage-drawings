"""Sparse point-context classifiers: image evidence only, no semantic approval."""
from functools import lru_cache
from pathlib import Path
import hashlib,json,math
import cv2,numpy as np
from .network import line_ink
DIR=Path(__file__).parent/'assets'
HOG=cv2.HOGDescriptor((32,32),(16,16),(8,8),(8,8),9)

@lru_cache(maxsize=1)
def model():
 p=DIR/'topology-context.json'
 if not p.exists():return None,None,None
 meta=json.loads(p.read_text());models=[]
 for name in ('port','junction'):
  file=DIR/meta[name+'_file']
  if hashlib.sha256(file.read_bytes()).hexdigest()!=meta[name+'_sha256']:raise ValueError('포트·접속점 모델 검증 값 불일치')
  models.append(cv2.ml.RTrees_load(str(file)))
 return *models,meta

def prepare(image):
 gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY);border=np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]]
 if np.median(border)<128:gray=255-gray
 return gray,line_ink(gray)

def rays(ink,x,y):
 """Ink support along actual arms, independent of Hough split endpoints."""
 h,w=ink.shape;out=[]
 for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
  scores=[]
  for lo,hi in ((4,9),(9,17),(17,26)):
   a=np.arange(lo,hi);xs=np.clip(np.rint(x+dx*a).astype(int),0,w-1);ys=np.clip(np.rint(y+dy*a).astype(int),0,h-1)
   hit=np.zeros(len(a),bool)
   for off in (-1,0,1):hit|=ink[np.clip(ys+(off if dx else 0),0,h-1),np.clip(xs+(off if dy else 0),0,w-1)]>0
   scores.append(float(hit.mean()))
  out+=scores
 return out

def ray_batch(ink,points):
 h,w=ink.shape;xy=np.asarray(points,dtype=float).reshape(-1,2);columns=[]
 for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
  for lo,hi in ((4,9),(9,17),(17,26)):
   a=np.arange(lo,hi);xs=np.clip(np.rint(xy[:,0,None]+dx*a).astype(int),0,w-1);ys=np.clip(np.rint(xy[:,1,None]+dy*a).astype(int),0,h-1);hit=np.zeros(xs.shape,bool)
   for off in (-1,0,1):hit|=ink[np.clip(ys+(off if dx else 0),0,h-1),np.clip(xs+(off if dy else 0),0,w-1)]>0
   columns.append(hit.mean(1))
 return np.column_stack(columns)


def features(gray,ink,points):
 h,w=gray.shape;out=[];all_context=ray_batch(ink,[p['pixel'] for p in points]) if points else []
 for index,p in enumerate(points):
  x,y=p['pixel'];side=p.get('side');rot={'right':0,'bottom':1,'left':2,'top':3}.get(side,0);vec=[]
  for radius in (12,24):
   patch=cv2.getRectSubPix(gray,(radius*2+1,radius*2+1),(float(x),float(y)));patch=np.rot90(patch,rot).copy();patch=cv2.resize(patch,(32,32),interpolation=cv2.INTER_AREA)
   vec.extend((HOG.compute(patch).ravel(),cv2.resize((patch<200).astype(np.float32),(8,8),interpolation=cv2.INTER_AREA).ravel()))
  context=all_context[index].tolist();geometry=p.get('geometry',[0.]*6)
  vec.append(np.array(context+geometry,np.float32));out.append(np.concatenate(vec))
 return np.array(out,np.float32)

def votes(classifier,x,classes):
 if len(x)==0:return np.zeros((0,classes),np.float32)
 raw=classifier.getVotes(x,0);out=np.zeros((len(x),classes),np.float32);labels=raw[0].astype(int);scores=raw[1:].astype(np.float32);den=np.maximum(1,scores.sum(1))
 for k,c in enumerate(labels):
  if 0<=c<classes:out[:,c]=scores[:,k]/den
 return out
