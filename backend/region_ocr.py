"""OCR only source-defined regions, preserving exact original pixel coordinates."""
import math

def read_regions(image,warnings,ocr_fn,regions,diagnostics=None):
    from .geometry import _iou
    w,h=image.size;boxes=[]
    for box in regions[:8]:
        if len(box)!=4 or not all(math.isfinite(v) for v in box):continue
        a,b,c,d=max(0,math.floor(box[0]*w)),max(0,math.floor(box[1]*h)),min(w,math.ceil(box[2]*w)),min(h,math.ceil(box[3]*h))
        if c-a<8 or d-b<8:continue
        normalized=[a/w,b/h,c/w,d/h]
        if not any(_iou(normalized,old[4])>.9 for old in boxes):boxes.append((a,b,c,d,normalized))
    words=[];pixels=0;hits=[]
    for a,b,c,d,box in boxes:
        crop=image.crop((a,b,c,d));diag={};found=ocr_fn(crop,warnings,diag);pixels+=(c-a)*(d-b);hits.append(diag.get('ocr_cache_hit',False))
        for item in found:
            x1,y1,x2,y2=item['bbox'];mapped=[(a+x1*(c-a))/w,(b+y1*(d-b))/h,(a+x2*(c-a))/w,(b+y2*(d-b))/h]
            if any(item['label']==e['label'] and _iou(mapped,e['bbox'])>.5 for e in words):continue
            words.append({**item,'bbox':mapped,'properties':{**item.get('properties',{}),'ocr_mode':'source_region','source_region':box}})
    if diagnostics is not None:diagnostics.update(ocr_mode='source_regions',ocr_region_count=len(boxes),ocr_source_pixels=w*h,ocr_processed_pixels=pixels,ocr_cache_hit=bool(hits) and all(hits))
    return words
