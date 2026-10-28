"""Exact synthetic supervision for line styles, ports and crossing conventions."""
import cv2,numpy as np
CHANNELS=['solid','dashed','port','tee','dot','plain_cross']

def scene(seed,size=256):
 rng=np.random.default_rng(seed);im=np.full((size,size),rng.integers(235,256),np.uint8);m=np.zeros((6,size,size),np.uint8);prompt=np.zeros_like(im);width=int(rng.choice([1,1,2,3]));color=int(rng.integers(10,160));kind=int(rng.integers(0,6));cx,cy=map(int,rng.integers(94,164,2));boxes=[];links=[]
 if kind in (0,1,2):
  endpoints=[(22,cy),(234,cy),(cx,22),(cx,234)];active=[0,1,2] if kind==0 else [0,1,2,3]
  for i in active:links.append((endpoints[i],(cx,cy)))
  if kind==0:cv2.circle(m[3],(cx,cy),4,1,-1)
  elif kind==1:cv2.circle(m[4],(cx,cy),4,1,-1)
  else:cv2.circle(m[5],(cx,cy),4,1,-1)
 elif kind==3:
  endpoints=[(22,cy-14),(234,cy-14),(22,cy+14),(234,cy+14)];links=[(endpoints[0],endpoints[1]),(endpoints[2],endpoints[3])];active=range(4)
 else:
  endpoints=[(22,70),(234,190)];active=[0,1]
  if kind==4:links=[(endpoints[0],(cx,70)),((cx,70),(cx,190)),((cx,190),endpoints[1])]
  else:links=[(endpoints[0],endpoints[1])]
 dashed=bool(rng.random()<.25);dash=int(rng.integers(5,11));gap=int(rng.integers(3,7))
 for a,b in links:
  if dashed:
   delta=np.array(b)-a;length=float(np.linalg.norm(delta));unit=delta/max(length,1)
   for t in range(0,int(length),dash+gap):
    p=tuple(np.round(np.array(a)+unit*t).astype(int));q=tuple(np.round(np.array(a)+unit*min(t+dash,length)).astype(int));cv2.line(im,p,q,color,width);cv2.line(m[1],p,q,1,max(3,width))
  else:cv2.line(im,a,b,color,width);cv2.line(m[0],a,b,1,max(3,width))
 if kind==1:cv2.circle(im,(cx,cy),max(3,width+2),color,-1)
 # Boxes are prompts, silhouettes are diverse and do not encode equipment IDs.
 for i in active:
  x,y=endpoints[i];rx,ry=map(int,rng.integers(8,15,2));a,b,c,d=max(0,x-rx),max(0,y-ry),min(size-1,x+rx),min(size-1,y+ry);boxes.append([a,b,c,d]);im[b:d+1,a:c+1]=255
  shape=int(rng.integers(0,3))
  if shape==0:cv2.ellipse(im,(x,y),(rx,ry),0,0,360,color,width)
  elif shape==1:cv2.rectangle(im,(a,b),(c,d),color,width)
  else:cv2.polylines(im,[np.array([[a,b],[c,d],[c,b],[a,d],[a,b]],np.int32)],False,color,width)
  cv2.rectangle(prompt,(a,b),(c,d),1,-1)
  for p,q in links:
   if (x,y) not in (p,q):continue
   other=np.array(q if p==(x,y) else p,float);delta=other-[x,y];t=min(rx/max(abs(delta[0]),1e-6),ry/max(abs(delta[1]),1e-6));point=tuple(np.round(np.array([x,y])+delta*t).astype(int));cv2.circle(m[2],point,4,1,-1)
 for a,b,c,d in boxes:m[:2,b:d+1,a:c+1]=0
 # Letter/box clutter is negative even when it contains short straight strokes.
 for _ in range(int(rng.integers(2,7))):
  x,y=map(int,rng.integers([10,18],[190,245]));cv2.putText(im,str(int(rng.integers(1,999)))+' PT',(x,y),cv2.FONT_HERSHEY_SIMPLEX,.25,int(rng.integers(60,180)),1,cv2.LINE_AA)
 if rng.random()<.2:im=cv2.GaussianBlur(im,(3,3),.45)
 if rng.random()<.15:im=np.clip(im.astype(float)+rng.normal(0,2,im.shape),0,255).astype(np.uint8)
 if rng.random()<.5:im=np.flip(im,1).copy();prompt=np.flip(prompt,1).copy();m=np.flip(m,2).copy()
 rotations=int(rng.integers(0,4));im=np.rot90(im,rotations).copy();prompt=np.rot90(prompt,rotations).copy();m=np.rot90(m,rotations,axes=(1,2)).copy()
 return im,prompt,m
