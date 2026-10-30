"""Text-region atlas OCR with reversible page coordinates and bounded fallback."""
from pathlib import Path
import json,math
import cv2,numpy as np
from PIL import Image

def propose(image):
    gray=cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2GRAY);h,w=gray.shape
    if np.median(np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]])<128:gray=255-gray
    ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)
    # Long pipes/frames are retained in the source, removed only from proposals.
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,31),np.uint8));vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((31,1),np.uint8));small=cv2.subtract(ink,horizontal|vertical)
    _,_,stats,_=cv2.connectedComponentsWithStats(small);mask=np.zeros_like(ink);count=0
    for x,y,cw,ch,area in stats[1:]:
        if 3<=ch<=64 and 1<=cw<=100 and 3<=area<=1800 and .06<=cw/ch<=12:
            cv2.rectangle(mask,(max(0,x-2),max(0,y-2)),(min(w-1,x+cw+2),min(h-1,y+ch+2)),255,-1);count+=1
    grouped=cv2.dilate(mask,np.ones((5,17),np.uint8));_,_,stats,_=cv2.connectedComponentsWithStats(grouped);boxes=[]
    for x,y,cw,ch,area in stats[1:]:
        if cw<7 or ch<5:continue
        a,b,c,d=max(0,x-7),max(0,y-6),min(w,x+cw+7),min(h,y+ch+6)
        boxes.append([int(a),int(b),int(c),int(d)])
    # Merge padded overlaps: one source character must not appear in two crops.
    for _ in range(3):
        old=len(boxes);merged=[]
        for box in sorted(boxes,key=lambda b:(b[1],b[0])):
            for i,m in enumerate(merged):
                if box[0]<m[2] and box[2]>m[0] and box[1]<m[3] and box[3]>m[1]:merged[i]=[min(m[0],box[0]),min(m[1],box[1]),max(m[2],box[2]),max(m[3],box[3])];break
            else:merged.append(box)
        boxes=merged
        if len(boxes)==old:break
    return sorted(boxes,key=lambda b:(b[1],b[0])),count

def atlas(image,boxes):
    """Shelf pack unchanged pixels; each entry maps a crop to original pixels."""
    gap=28;width=max(1024,max(b[2]-b[0] for b in boxes)+gap*2);rows=[];x=y=gap;row_h=0
    for box in boxes:
        cw,ch=box[2]-box[0],box[3]-box[1]
        if x+cw+gap>width:x=gap;y+=row_h+gap;row_h=0
        rows.append({'source':box,'atlas':[x,y,x+cw,y+ch]});x+=cw+gap;row_h=max(row_h,ch)
    height=y+row_h+gap
    if width*height>image.width*image.height*.72 or max(width,height)>4800:return None,[]
    result=Image.new('RGB',(width,height),'white')
    for row in rows:result.paste(image.crop(tuple(row['source'])),tuple(row['atlas'][:2]))
    return result,rows

def remap(words,rows,atlas_size,page_size):
    aw,ah=atlas_size;w,h=page_size;result=[]
    for word in words:
        box=np.array(word['bbox'],float)*[aw,ah,aw,ah];cx,cy=(box[:2]+box[2:])/2
        row=next((r for r in rows if r['atlas'][0]<=cx<=r['atlas'][2] and r['atlas'][1]<=cy<=r['atlas'][3]),None)
        if row is None:continue
        a,b,c,d=row['atlas'];sx,sy,_,_=row['source'];x1,y1,x2,y2=box
        if x1<a-1 or y1<b-1 or x2>c+1 or y2>d+1:continue
        mapped=[max(0.,min(1.,(x1-a+sx)/w)),max(0.,min(1.,(y1-b+sy)/h)),max(0.,min(1.,(x2-a+sx)/w)),max(0.,min(1.,(y2-b+sy)/h))]
        result.append({**word,'bbox':mapped,'properties':{**word.get('properties',{}),'ocr_mode':'region_atlas','source_region':[sx/w,sy/h,row['source'][2]/w,row['source'][3]/h]}})
    return sorted(result,key=lambda r:(r['bbox'][1],r['bbox'][0],r['label']))

def masked(image):
    gray=cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2GRAY);h,w=gray.shape
    if np.median(np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]])<128:gray=255-gray
    ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)
    hline=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,35),np.uint8));vline=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((35,1),np.uint8));residual=cv2.subtract(ink,hline|vline);keep=np.zeros_like(ink)
    for pixels in (ink,residual):
        _,labels,stats,_=cv2.connectedComponentsWithStats(pixels)
        eligible=np.zeros(len(stats),np.uint8)
        for i,(x,y,cw,ch,area) in enumerate(stats[1:],1):
            if 3<=ch<=64 and 1<=cw<=100 and 3<=area<=1800 and .06<=cw/ch<=12:eligible[i]=255
        keep|=eligible[labels]
    keep=cv2.dilate(keep,np.ones((11,11),np.uint8));arr=np.asarray(image.convert('RGB')).copy();arr[keep==0]=255
    return Image.fromarray(arr)

def recognize(image,warnings,ocr_fn,diagnostics=None,mode='auto'):
    if mode=='full':return ocr_fn(image,warnings,diagnostics)
    if mode=='mask':
        if diagnostics is not None:diagnostics.update(ocr_mode='region_mask',ocr_source_pixels=image.width*image.height,ocr_processed_pixels=image.width*image.height)
        return ocr_fn(masked(image),warnings,diagnostics)
    boxes,characters=propose(image);pack,rows=atlas(image,boxes) if boxes and len(boxes)<=140 else (None,[])
    if diagnostics is not None:diagnostics.update(ocr_region_count=len(boxes),ocr_mode='region_atlas' if pack else 'full_fallback',ocr_source_pixels=image.width*image.height,ocr_processed_pixels=pack.width*pack.height if pack else image.width*image.height)
    if pack is None:return ocr_fn(image,warnings,diagnostics)
    words=remap(ocr_fn(pack,warnings,diagnostics),rows,pack.size,image.size)
    if characters>=30 and sum(len(w['label']) for w in words)<max(12,characters*.12):
        if diagnostics is not None:diagnostics.update(ocr_mode='full_low_yield_fallback',ocr_processed_pixels=pack.width*pack.height+image.width*image.height)
        return ocr_fn(image,warnings,diagnostics)
    return words
