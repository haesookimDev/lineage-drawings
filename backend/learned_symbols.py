"""CPU proposal detector with trained HOG/ink random-forest objectness.

Only visual candidate localization is predicted. No standard identifiers or
semantic review approvals are generated. Training lives outside the runtime.
"""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import hashlib,json,math
import cv2
import numpy as np

MODEL_DIR=Path(__file__).parent/'assets'
MAX_PROPOSALS=7000
HOG=cv2.HOGDescriptor((32,32),(16,16),(8,8),(8,8),9)


def prepare_gray(gray):
    border=np.concatenate((gray[0],gray[-1],gray[:,0],gray[:,-1]))
    return 255-gray if np.median(border)<128 else gray


def proposals(gray,base=(),include_large=False,dilations=(1,3,7,15)):
    """Bounded multi-scale connected-ink regions; no OCR/GT input is required."""
    gray=prepare_gray(gray);h,w=gray.shape
    ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,7)
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,17),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((17,1),np.uint8))
    residual=cv2.subtract(ink,horizontal|vertical)
    regions=[]
    for dilation in dilations:
        mask=residual if dilation==1 else cv2.dilate(residual,np.ones((dilation,dilation),np.uint8))
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        pad=dilation//2
        for x,y,cw,ch,area in stats[1:]:
            x,y,cw,ch=int(x+pad),int(y+pad),int(cw-2*pad),int(ch-2*pad)
            if area<6 or min(cw,ch)<3 or max(cw,ch)>max(180,min(w,h)*.22) or cw*ch>w*h*.035:continue
            if max(cw,ch)/max(1,min(cw,ch))>9:continue
            cx,cy=x+cw/2,y+ch/2
            # A full component and nearby scales recover split bow-ties, stems,
            # double circles and outlines with missing contour closure.
            for scale in (1.,1.35,1.8):
                regions.append((cx,cy,max(7,cw)*scale,max(7,ch)*scale))
            if max(cw,ch)<=44:
                size=max(12,math.sqrt(cw*ch)*1.5)
                for ratio in (.65,1.,1.5,2.25):
                    regions.append((cx,cy,size*math.sqrt(ratio),size/math.sqrt(ratio)))
    raw=[]
    for box in base:
        a,b,c,d=box;raw.append((a*w,b*h,c*w,d*h))
    if include_large:
        contours,_=cv2.findContours(ink,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,cw,ch=cv2.boundingRect(contour);area=cw*ch
            if min(cw,ch)<30 or max(cw,ch)<75 or not 1500<=area<=w*h*.18:continue
            if max(cw,ch)/min(cw,ch)>9:continue
            raw.append((x-1,y-1,x+cw+1,y+ch+1))
    raw += [(cx-cw/2,cy-ch/2,cx+cw/2,cy+ch/2) for cx,cy,cw,ch in regions]
    # Spatially fair allocation: dense title-block text must not spend the
    # complete proposal budget before the process area is visited.
    cells=defaultdict(list);seen=set()
    for box in raw:
        a,b,c,d=box;a,b,c,d=max(0,round(a)),max(0,round(b)),min(w,round(c)),min(h,round(d))
        if c-a<5 or d-b<5:continue
        key=(a//2,b//2,c//2,d//2)
        if key in seen:continue
        seen.add(key);cells[(int((a+c)/2)//96,int((b+d)/2)//96)].append((a,b,c,d))
    out=[];groups=list(cells.values())
    for i in range(max((len(v) for v in groups),default=0)):
        for group in groups:
            if i<len(group):out.append(group[i])
            if len(out)>=MAX_PROPOSALS:return out
    return out


def features(gray,boxes):
    h,w=gray.shape;out=[]
    for a,b,c,d in boxes:
        a,b,c,d=map(int,(a,b,c,d));cw,ch=c-a,d-b
        px,py=max(1,round(cw*.08)),max(1,round(ch*.08))
        crop=gray[max(0,b-py):min(h,d+py),max(0,a-px):min(w,c+px)]
        if crop.size==0:raise ValueError('Empty detector proposal')
        patch=cv2.resize(crop,(32,32),interpolation=cv2.INTER_AREA)
        patch=cv2.normalize(patch,None,0,255,cv2.NORM_MINMAX)
        grid=cv2.resize((patch<210).astype(np.float32),(8,8),interpolation=cv2.INTER_AREA).ravel()
        hog=HOG.compute(patch).ravel()
        # No page position or project identity is included in model features.
        ext=[gray[max(0,b):min(h,d),max(0,a-5):max(0,a-1)],gray[max(0,b):min(h,d),min(w,c+1):min(w,c+5)],
             gray[max(0,b-5):max(0,b-1),max(0,a):min(w,c)],gray[min(h,d+1):min(h,d+5),max(0,a):min(w,c)]]
        ports=[float((p<210).mean()) if p.size else 0. for p in ext]
        geometry=[math.log(max(.05,cw/ch)),math.log(max(1,cw))/6,math.log(max(1,ch))/6,*ports]
        out.append(np.concatenate((hog,grid,np.asarray(geometry,dtype=np.float32))))
    return np.asarray(out,dtype=np.float32)


@lru_cache(maxsize=1)
def model():
    path=MODEL_DIR/'symbol-objectness.xml.gz';meta_path=MODEL_DIR/'symbol-objectness.json'
    if not meta_path.exists():return None,None
    meta=json.loads(meta_path.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['weights_sha256']:raise ValueError('학습 기호 모델의 검증 값이 맞지 않습니다.')
    return cv2.ml.RTrees_load(str(path)),meta


def probabilities(classifier,x):
    votes=classifier.getVotes(x,0)
    classes=votes[0];data=votes[1:].astype(np.float32)
    return data[:,classes>0].sum(axis=1)/np.maximum(1,data.sum(axis=1))


def score_boxes(image,boxes):
    classifier,meta=model()
    if classifier is None or not boxes:return np.ones(len(boxes),np.float32)
    gray=prepare_gray(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY));h,w=gray.shape
    pixel=[tuple(round(v*[w,h][i%2]) for i,v in enumerate(b)) for b in boxes]
    return probabilities(classifier,features(gray,pixel))


def detect(image,base=(),threshold=None):
    classifier,meta=model()
    if classifier is None:return []
    gray=prepare_gray(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY));h,w=gray.shape
    # The forest also saw exact/jittered large equipment boxes in training.
    # Extra contour proposals supply those extents at inference time.
    boxes=proposals(gray,base,include_large=True)
    if not boxes:return []
    x=features(gray,boxes);scores=probabilities(classifier,x);cut=meta['threshold'] if threshold is None else threshold
    chosen=np.flatnonzero(scores>=cut)
    if len(chosen)==0:return []
    b=np.asarray([boxes[i] for i in chosen],dtype=np.float32);s=scores[chosen]
    xywh=np.column_stack((b[:,:2],b[:,2:]-b[:,:2])).tolist()
    kept=cv2.dnn.NMSBoxes(xywh,s.tolist(),float(cut)-1e-5,.35,top_k=600)
    out=[]
    for index in np.asarray(kept).ravel()[:120]:
        a,b0,c,d=b[int(index)];out.append(([float(a/w),float(b0/h),float(c/w),float(d/h)],
            {'geometry':'learned_symbol','detector_score':float(s[int(index)]),'model':'hog-rtrees-symbol-v1','requires_review':True}))
    return out
