"""Development-only shape templates + geometric circles/continuation markers."""
from functools import lru_cache
from pathlib import Path
import hashlib,json
import cv2
import numpy as np


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.


@lru_cache(maxsize=1)
def prototypes():
    root=Path(__file__).parent/'assets';path=root/'open100-dev-shapes.npz'
    metadata=json.loads((root/'open100-dev-shapes.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['weights_sha256']:
        raise ValueError('형상 모델 파일의 검증 값이 맞지 않습니다.')
    with np.load(path,allow_pickle=False) as weights:
        return [(weights[f't{i}'].copy(),entry['family']) for i,entry in enumerate(metadata['prototypes'])]


def template_candidates(gray):
    h,w=gray.shape;binary=(gray<210).astype(np.float32);ink=(binary*255).astype(np.uint8)
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(12,w//150)),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(12,h//150),1),np.uint8))
    # Search at half resolution, then verify full-resolution NCC only around
    # coarse peaks. The full-resolution acceptance threshold is unchanged.
    # This avoids 219 full-page FFTs and retains the original model assets.
    coarse=cv2.resize(binary,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    found=[]
    for index,(original,family) in enumerate(prototypes()):
        for scale in (.85,1.,1.15):
            th,tw=original.shape;tw,th=round(tw*scale),round(th*scale)
            if min(tw,th)<5 or max(tw,th)>min(w,h)/2:continue
            template=cv2.resize(original,(tw,th))
            small=cv2.resize(template,(max(3,tw//2),max(3,th//2)),interpolation=cv2.INTER_AREA)
            similarity=cv2.matchTemplate(coarse,small,cv2.TM_CCOEFF_NORMED)
            peaks=(similarity>=.55)&(similarity>=cv2.dilate(similarity,np.ones((3,3),np.uint8)))
            yy,xx=np.where(peaks)
            if len(xx)>300:
                best=np.argpartition(similarity[yy,xx],-300)[-300:];yy,xx=yy[best],xx[best]
            for sy,sx in zip(yy,xx):
                x0,y0=max(0,int(sx)*2-3),max(0,int(sy)*2-3)
                x1,y1=min(w,int(sx)*2+tw+5),min(h,int(sy)*2+th+5)
                if x1-x0<tw or y1-y0<th:continue
                fine=cv2.matchTemplate(binary[y0:y1,x0:x1],template,cv2.TM_CCOEFF_NORMED)
                _,score,_,position=cv2.minMaxLoc(fine)
                if score<.78:continue
                x,y=x0+position[0],y0+position[1]
                ports=(horizontal[y:y+th,max(0,x-4):x],horizontal[y:y+th,x+tw:min(w,x+tw+4)],
                       vertical[max(0,y-4):y,x:x+tw],vertical[y+th:min(h,y+th+4),x:x+tw])
                if not any(np.count_nonzero(port)>=3 for port in ports):continue
                found.append(([float(x/w),float(y/h),float((x+tw)/w),float((y+th)/h)],
                              {'geometry':'shape_prototype','match_score':float(score),
                               'prototype':index,'shape_hint':family,'model':'open100-dev-shapes-v1'}))
    found.sort(key=lambda p:p[1]['match_score'],reverse=True);kept=[]
    for item in found:
        if any(iou(item[0],prior[0])>.35 for prior in kept):continue
        kept.append(item)
        if len(kept)>=200:break
    return kept


def interior_parts(box,gray):
    h,w=gray.shape;a,b,c,d=box;cx,cy=(a+c)*w/2,(b+d)*h/2;radius=min((c-a)*w,(d-b)*h)/2
    x1,y1=max(0,round(cx-radius)),max(0,round(cy-radius));x2,y2=min(w,round(cx+radius)+1),min(h,round(cy+radius)+1)
    crop=gray[y1:y2,x1:x2]
    yy,xx=np.indices(crop.shape)
    mask=((crop<210)&((xx-(cx-x1))**2+(yy-(cy-y1))**2<(radius*.75)**2)).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    return sum(int(area)>=1 for x,y,cw,ch,area in stats[1:])


def pointed_loop(box,gray):
    h,w=gray.shape;a,b,c,d=box;x1,y1,x2,y2=map(round,(a*w,b*h,c*w,d*h));cw,ch=x2-x1,y2-y1
    if min(cw,ch)<8 or max(cw,ch)/min(cw,ch)<2.2:return False
    white=(gray[y1:y2,x1:x2]>210).astype(np.uint8)*255
    contours,_=cv2.findContours(white,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:2]:
        if cv2.contourArea(contour)<cw*ch*.45:continue
        polygon=cv2.approxPolyDP(contour,max(.8,cv2.arcLength(contour,True)*.008),True)[:,0,:]
        if not 5<=len(polygon)<=10:continue
        if ch>cw:polygon=polygon[:,::-1]
        xx,yy=polygon[:,0],polygon[:,1];lo,hi=yy.min(),yy.max();mid=(lo+hi)/2
        right,left=polygon[xx>=xx.max()-1],polygon[xx<=xx.min()+1]
        if any(len(side)==1 and abs(side[0,1]-mid)<(hi-lo)*.25 for side in (right,left)):return True
    return False


def refine_symbols(image,legacy):
    gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
    pointed=[(box,{'geometry':'pointed_loop','line_port':True}) for box,properties in legacy
             if properties.get('geometry')=='closed_loop' and pointed_loop(box,gray)]
    # Require both boxed continuation markers and multiple matching shapes.
    # No input filename or dataset identity is used for this style gate.
    if len(pointed)<2:
        return [(box,{**props,'profile':'generic_fallback'}) for box,props in legacy]
    trained=template_candidates(gray)
    if len(trained)<3:
        return [(box,{**props,'profile':'generic_fallback'}) for box,props in legacy]
    circles=[]
    for box,properties in legacy:
        if properties.get('geometry')!='round_loop':continue
        a,b,c,d=box;cx,cy=(a+c)/2,(b+d)/2;area=(c-a)*(d-b)
        if any(q[0]<=cx<=q[2] and q[1]<=cy<=q[3] and area<(q[2]-q[0])*(q[3]-q[1]) for q,_ in trained):continue
        parts=interior_parts(box,gray)
        if parts<2 or any(iou(box,other[0])>.3 for other in circles):continue
        circles.append((box,{**properties,'interior_parts':parts,'connection_eligible':parts>=4}))
    result=trained+circles
    for item in pointed:
        if not any(iou(item[0],other[0])>.3 for other in result):result.append(item)
    return result
