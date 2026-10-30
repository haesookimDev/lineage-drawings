"""Bounded source-legend matching. Scores are proposals, not calibrated truth."""
import cv2,numpy as np
from PIL import Image

def binary(image):
    gray=cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2GRAY)
    if np.median(np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]])<128:gray=255-gray
    return cv2.adaptiveThreshold(gray,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,7).astype(np.float32)

def match(image,templates,threshold=.48,max_candidates=30):
    """Template masks use source pixels; no target annotations or learned IDs."""
    source=image.copy();source.thumbnail((2400,2400));w,h=source.size;ink=binary(source);found=[]
    for template in templates[:8]:
        original=binary(template['image']);orientations=[]
        for turn in (0,1,2,3):
            pattern=np.rot90(original,turn).copy()
            if any(pattern.shape==old.shape and np.array_equal(pattern,old) for _,old in orientations):continue
            orientations.append((turn,pattern))
        for turn,pattern in orientations:
          ph,pw=pattern.shape
          for scale in (.65,.85,1.,1.3,1.7):
            tw,th=max(5,round(pw*scale)),max(5,round(ph*scale))
            if min(tw,th)<6 or max(tw,th)>160 or tw>=w or th>=h:continue
            small=cv2.resize(pattern,(tw,th),interpolation=cv2.INTER_NEAREST);mask=np.ones_like(small)
            if template.get('mode')=='outline' and min(tw,th)>12:cv2.ellipse(mask,(tw//2,th//2),(max(1,tw//4),max(1,th//4)),0,0,360,0,-1)
            small*=mask;area=float(small.sum())
            if area<5:continue
            intersection=cv2.matchTemplate(ink,small,cv2.TM_CCORR);target=cv2.matchTemplate(ink,mask,cv2.TM_CCORR);score=np.clip(intersection/np.maximum(area+target-intersection,1),0,1)
            peaks=(score>=threshold)&(score>=cv2.dilate(score,np.ones((max(3,th//2),max(3,tw//2)),np.uint8))-.00001);ys,xs=np.where(peaks)
            ids=sorted(range(len(xs)),key=lambda i:(-float(score[ys[i],xs[i]]),int(ys[i]),int(xs[i])))[:20]
            for i in ids:
                x,y=int(xs[i]),int(ys[i]);found.append({'bbox':[x/w,y/h,(x+tw)/w,(y+th)/h],'score':float(score[y,x]),'template_id':template['id'],'name':template['name'],'concept_id':template.get('concept_id'),'scale':scale,'rotation':turn*90})
    from .geometry import _iou
    result=[]
    for candidate in sorted(found,key=lambda p:(-p['score'],p['template_id'],p['bbox'])):
        if any(_iou(candidate['bbox'],p['bbox'])>.35 for p in result):continue
        result.append(candidate)
        if len(result)>=max_candidates:break
    return result
