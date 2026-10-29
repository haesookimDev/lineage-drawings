"""Bounded pixel-supported recovery of short branches and periodic dashed runs."""
import math
import cv2,numpy as np


def profile(ink,a,b,c,d,radius=1):
 h,w=ink.shape;n=max(2,round(math.hypot(c-a,d-b))+1);x=np.rint(np.linspace(a,c,n)).astype(int);y=np.rint(np.linspace(b,d,n)).astype(int);horizontal=abs(c-a)>=abs(d-b);hit=np.zeros(n,bool)
 for off in range(-radius,radius+1):hit|=ink[np.clip(y+(off if horizontal else 0),0,h-1),np.clip(x+(0 if horizontal else off),0,w-1)]>0
 return hit


def periodic(hit):
 changes=np.flatnonzero(np.diff(np.r_[False,hit,False].astype(np.int8)));runs=changes[1::2]-changes[::2];gaps=changes[2::2]-changes[1:-1:2]
 return bool(len(runs)>=3 and len(gaps)>=2 and np.median(runs)>=3 and 2<=np.median(gaps)<=24 and gaps.max()<=26 and np.std(gaps)/max(1,np.mean(gaps))<=.55 and .22<hit.mean()<.92)


def periodic_batch(hit):
 """Same run/gap predicate as periodic(), without per-intersection Python."""
 hit=np.asarray(hit,dtype=bool)
 if hit.ndim!=2:raise ValueError('Expected rows of sampled line pixels')
 result=np.zeros(len(hit),bool)
 if not len(hit):return result
 changes=np.diff(np.pad(hit.astype(np.int8),((0,0),(1,1))),axis=1)
 counts=(changes==1).sum(1);coverage=hit.mean(1)
 selected=np.flatnonzero((counts>=3)&(coverage>.22)&(coverage<.92))
 if not len(selected):return result
 rows=changes[selected];n=rows.shape[1];positions=np.arange(n)
 # Sorting padded coordinates creates fixed-size, ordered run boundaries.
 starts=np.sort(np.where(rows==1,positions,n),axis=1)
 ends=np.sort(np.where(rows==-1,positions,n),axis=1)
 valid=np.arange(n)[None,:]<counts[selected,None]
 runs=np.where(valid,ends-starts,np.nan)
 gap_valid=np.arange(n-1)[None,:]<(counts[selected]-1)[:,None]
 gaps=np.where(gap_valid,starts[:,1:]-ends[:,:-1],np.nan)
 med=np.nanmedian(gaps,axis=1)
 result[selected]=(np.nanmedian(runs,axis=1)>=3)&(med>=2)&(med<=24)&(np.nanmax(gaps,axis=1)<=26)&(np.nanstd(gaps,axis=1)/np.maximum(1,np.nanmean(gaps,axis=1))<=.55)
 return result


def path_has_periodic_dashes(ink,points,w,h):
 """A lower signal-port gate requires actual repeated dashes on its path."""
 for first,last in zip(points,points[1:]):
  a,b=np.asarray(first)*[w,h];c,d=np.asarray(last)*[w,h]
  if math.hypot(c-a,d-b)>=45 and periodic(profile(ink,a,b,c,d)):
   return True
 return False


def recover(ink,lines):
 """Closing proposes runs; periodic pixel evidence, not closing, accepts them."""
 h,w=ink.shape;extra=[]
 for horizontal in (True,False):
  stroke=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,3) if horizontal else (3,1),np.uint8));closed=cv2.morphologyEx(stroke,cv2.MORPH_CLOSE,np.ones((1,25) if horizontal else (25,1),np.uint8));long=cv2.morphologyEx(closed,cv2.MORPH_OPEN,np.ones((1,45) if horizontal else (45,1),np.uint8));_,_,stats,_=cv2.connectedComponentsWithStats(long)
  for x,y,cw,ch,area in stats[1:]:
   if (ch if horizontal else cw)>4:continue
   a,b,c,d=(x,y+(ch-1)/2,x+cw-1,y+(ch-1)/2) if horizontal else (x+(cw-1)/2,y,x+(cw-1)/2,y+ch-1);hit=profile(ink,a,b,c,d)
   if not periodic(hit):continue
   changes=np.flatnonzero(np.diff(np.r_[False,hit,False].astype(np.int8)));gap=float(np.median(changes[2::2]-changes[1:-1:2]));extension=min(12.,gap/2+2)
   extra.append([max(0,a-extension) if horizontal else a,max(0,b-extension) if not horizontal else b,min(w-1,c+extension) if horizontal else c,min(h-1,d+extension) if not horizontal else d])
   if len(extra)>=500:break
 # A small rasterization gap may lie between a detected box and its visible
 # pipe stub. Extrapolate by3px only; junction arms still require actual ink.
 extended=[]
 for a,b,c,d in lines:
  v=np.array([c-a,d-b],float);length=np.linalg.norm(v)
  if length<5:continue
  v*=3/length;extended.append([float(np.clip(a-v[0],0,w-1)),float(np.clip(b-v[1],0,h-1)),float(np.clip(c+v[0],0,w-1)),float(np.clip(d+v[1],0,h-1))])
 return extended+extra,len(extra)


def short_links(elements,ink,w,h):
 """Two visible dashes between aligned symbol faces can form a short signal.
 No arbitrary empty gap: require ink at both ends and bounded white intervals.
 """
 boxes=[np.array(e['bbox'])*[w,h,w,h] for e in elements if e['kind']!='text' and e.get('properties',{}).get('connection_eligible',True)];out=[]
 for i,a in enumerate(boxes):
  for b in boxes[i+1:]:
   ac=(a[:2]+a[2:])/2;bc=(b[:2]+b[2:])/2
   if abs(ac[0]-bc[0])<=3:
    top,bottom=(a,b) if ac[1]<bc[1] else (b,a);x=(ac[0]+bc[0])/2;line=[x,top[3],x,bottom[1]]
    if top[3]>=bottom[1]:continue
   elif abs(ac[1]-bc[1])<=3:
    left,right=(a,b) if ac[0]<bc[0] else (b,a);y=(ac[1]+bc[1])/2;line=[left[2],y,right[0],y]
    if left[2]>=right[0]:continue
   else:continue
   length=math.hypot(line[2]-line[0],line[3]-line[1])
   if length<12 or length>160:continue
   hit=profile(ink,*line);changes=np.flatnonzero(np.diff(np.r_[False,hit,False].astype(np.int8)));runs=changes[1::2]-changes[::2];gaps=changes[2::2]-changes[1:-1:2]
   if len(runs)<2 or not len(gaps) or np.median(runs)<4 or gaps.max()>24 or not .25<hit.mean()<.94 or not hit[:5].any() or not hit[-5:].any():continue
   if len(gaps)>1 and np.std(gaps)/max(1,np.mean(gaps))>.55:continue
   v=np.array(line[2:])-line[:2];v*=2/max(1,length);out.append([*(np.array(line[:2])-v),*(np.array(line[2:])+v)])
   if len(out)>=500:return out
 return out


def dashed_arms(ink,points):
 """Dash gaps cannot make a real four-way crossing look like a T."""
 h,w=ink.shape;xy=np.asarray(points,float).reshape(-1,2);offsets=np.arange(3,66);result=np.zeros((len(xy),4),bool)
 for k,(dx,dy) in enumerate(((1,0),(-1,0),(0,1),(0,-1))):
  xs=np.rint(xy[:,0,None]+dx*offsets).astype(int);ys=np.rint(xy[:,1,None]+dy*offsets).astype(int);hit=np.zeros(xs.shape,bool)
  for off in (-1,0,1):hit|=ink[np.clip(ys+(off if dx else 0),0,h-1),np.clip(xs+(off if dy else 0),0,w-1)]>0
  result[:,k]=periodic_batch(hit)
 return result
