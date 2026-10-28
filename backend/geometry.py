"""CPU geometric proposals and multi-segment adjacency, without semantic labels.

A supported circle is a geometric observation, not an ISA/ISO classification.
All output still requires review. Raster tracing assumes visible continuous lines;
four-way crossings without a dot follow a conservative non-contact convention.
Dash semantics and hidden ports still require review.
"""
from collections import defaultdict
import math

import cv2
import numpy as np


def _iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/union if union > 0 else 0.


def _box(x, y, width, height, w, h):
    return [max(0.,float(x/w)), max(0.,float(y/h)), min(1.,float((x+width)/w)), min(1.,float((y+height)/h))]


def _axis_masks(bw, w, h, divisor=100, minimum=20):
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1,max(minimum,w//divisor)),np.uint8))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((max(minimum,h//divisor),1),np.uint8))
    return horizontal,vertical


def _round_loops(gray):
    h,w = gray.shape
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1,minDist=16,param1=100,param2=22,
                              minRadius=9,maxRadius=max(12,min(w,h)//25))
    if circles is None:
        return []
    distance = cv2.distanceTransform((gray>200).astype(np.uint8),cv2.DIST_L2,3)
    angles = np.linspace(0,2*np.pi,96,endpoint=False)
    out = []
    for x,y,r in circles[0][:4000]:
        if r*2 > min(w,h)*.15:
            continue
        xs = np.clip(np.rint(x+r*np.cos(angles)).astype(int),0,w-1)
        ys = np.clip(np.rint(y+r*np.sin(angles)).astype(int),0,h-1)
        support = distance[ys,xs] <= 1.8
        # Hough alone hallucinates circles at pipe corners and text. Require ink
        # around the whole perimeter, including each quadrant, before retaining.
        if support.mean() < .8 or min(part.mean() for part in np.array_split(support,4)) < .6:
            continue
        out.append((_box(x-r,y-r,2*r,2*r,w,h),{'geometry':'round_loop','perimeter_support':float(support.mean())}))
    return out


def _closed_loops(bw, texts):
    h,w = bw.shape
    contours,hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    raw = []
    for idx,c in enumerate(contours):
        x,y,cw,ch = cv2.boundingRect(c)
        area = abs(cv2.contourArea(c))
        if hierarchy[0][idx][3] < 0 or min(cw,ch)<7 or max(cw,ch)>min(w,h)*.3 or max(cw,ch)/min(cw,ch)>8 or area<25 or area/(cw*ch)<.25:
            continue
        depth,j = 0,idx
        while hierarchy[0][j][3] >= 0:
            depth += 1
            j = hierarchy[0][j][3]
        if depth%2 != 1:
            continue
        box = _box(x-1,y-1,cw+2,ch+2,w,h)
        # OCR inside a large loop must not erase its symbol outline.
        if max(cw,ch)<22 and any(_iou(box,t['bbox'])>.25 for t in texts):
            continue
        raw.append(box)
        if len(raw) >= 1500:
            break
    filtered = []
    for a in raw:
        area = (a[2]-a[0])*(a[3]-a[1])
        if any(a is not b and b[0]<=a[0] and b[1]<=a[1] and b[2]>=a[2] and b[3]>=a[3]
               and (b[2]-b[0])*(b[3]-b[1]) < 6*area for b in raw):
            continue
        filtered.append(a)
    # Hollow bow-ties are two separate white holes. Join aligned adjacent halves
    # by geometry only, without calling them valves or assigning standard IDs.
    used,merged = set(),[]
    for i,a in enumerate(filtered):
        if i in used:
            continue
        x1,y1,x2,y2 = a
        aw,ah = (x2-x1)*w,(y2-y1)*h
        if max(aw,ah)>45:
            continue
        for j,b in enumerate(filtered):
            if j<=i or j in used:
                continue
            u1,v1,u2,v2 = b
            bw0,bh = (u2-u1)*w,(v2-v1)*h
            gapx,gapy = max(x1-u2,u1-x2,0)*w,max(y1-v2,v1-y2,0)*h
            aligned_h = abs((y1+y2-v1-v2)*h/2)<3 and abs(ah-bh)<4 and gapx<6 and .35<aw/ah<1.2 and .35<bw0/bh<1.2
            aligned_v = abs((x1+x2-u1-u2)*w/2)<3 and abs(aw-bw0)<4 and gapy<6 and .35<ah/aw<1.2 and .35<bh/bw0<1.2
            if aligned_h or aligned_v:
                used.update((i,j))
                merged.append([min(x1,u1),min(y1,v1),max(x2,u2),max(y2,v2)])
                break
    return [a for i,a in enumerate(filtered) if i not in used]+merged


def _has_port(box, horizontal, vertical):
    h,w = horizontal.shape
    x1,y1,x2,y2 = box
    x,y,cw,ch = int(x1*w),int(y1*h),int((x2-x1)*w),int((y2-y1)*h)
    if cw*ch<90:
        return False
    midx,midw = int(x+cw*.25),max(1,int(cw*.5))
    midy,midh = int(y+ch*.25),max(1,int(ch*.5))
    # Sample away from corners: table frames and closed pipe loops often have
    # lines only at corners, while a symbol port continues through a side.
    ports = [horizontal[max(0,midy):min(h,midy+midh),max(0,x-12):max(0,x-2)],
             horizontal[max(0,midy):min(h,midy+midh),min(w,x+cw+2):min(w,x+cw+12)],
             vertical[max(0,y-12):max(0,y-2),max(0,midx):min(w,midx+midw)],
             vertical[min(h,y+ch+2):min(h,y+ch+12),max(0,midx):min(w,midx+midw)]]
    return any(np.count_nonzero(port)>=5 for port in ports)


def legacy_candidates(image, texts):
    """Return normalized shape proposals plus axis line segments, at viewer scale."""
    w,h = image.size
    gray = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
    horizontal,vertical = _axis_masks(bw,w,h)
    candidates = _round_loops(gray)
    for a in _closed_loops(bw,texts):
        if not _has_port(a,horizontal,vertical):
            continue
        cx,cy = (a[0]+a[2])/2,(a[1]+a[3])/2
        if any(b[0]<=cx<=b[2] and b[1]<=cy<=b[3] for b,_ in candidates):
            continue
        candidates.append((a,{'geometry':'closed_loop','line_port':True}))
    if not candidates:
        # Preserve a generic fallback for drawings without supported closed loops.
        mask = bw.copy()
        for text in texts:
            a,b,c,d = text['bbox']
            mask[max(0,int(b*h)-2):min(h,int(d*h)+2),max(0,int(a*w)-2):min(w,int(c*w)+2)] = 0
        symbols = cv2.morphologyEx(cv2.subtract(mask,horizontal|vertical),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        _,_,stats,_ = cv2.connectedComponentsWithStats(symbols)
        for x,y,cw,ch,area in stats[1:]:
            if area<30 or min(cw,ch)<8 or cw>w*.7 or ch>h*.7 or cw*ch>w*h*.25:
                continue
            candidates.append((_box(x,y,cw,ch,w,h),{'geometry':'component_fallback'}))
            if len(candidates)>=200:
                break
    # Separate horizontal/vertical masks avoid losing bends in a Hough segment.
    horizontal,vertical = _axis_masks(bw,w,h,divisor=150,minimum=15)
    segments = []
    for mask,is_horizontal in ((horizontal,True),(vertical,False)):
        _,_,stats,_ = cv2.connectedComponentsWithStats(mask)
        for x,y,cw,ch,_ in stats[1:]:
            if is_horizontal and ch<8:
                segments.append([int(x),float(y+(ch-1)/2),int(x+cw-1),float(y+(ch-1)/2)])
            elif not is_horizontal and cw<8:
                segments.append([float(x+(cw-1)/2),int(y),float(x+(cw-1)/2),int(y+ch-1)])
    return candidates[:200],segments[:5000],len(candidates)>200


def geometric_candidates(image,texts,high_resolution=None):
    from .symbols import refine_symbols
    from .learned_symbols import detect,score_boxes,model
    from .network import axis_segments
    from .neural_symbols import detect as neural_detect,merge,model as neural_model
    # The deployed worker has a 2-CPU limit. Bound CV's native thread pool too.
    cv2.setNumThreads(2)
    legacy,legacy_lines,limited=legacy_candidates(image,texts)
    candidates=refine_symbols(image,legacy)
    proposal_base=[box for box,_ in candidates]
    fallback=not candidates or all(p.get('profile')=='generic_fallback' for _,p in candidates)
    net,neural_meta=neural_model()
    neural_only=net is not None and neural_meta.get('merge_mode')=='neural_only'
    classifier,metadata=(None,None) if neural_only else model()
    if classifier is not None:
        scores=score_boxes(image,[box for box,_ in candidates])
        candidates=[(box,{**props,'objectness_score':float(score)}) for (box,props),score in zip(candidates,scores)
                    if score>=metadata.get('prune_threshold',0.)]
    learned=[] if neural_only else detect(image,proposal_base)
    for box,properties in learned:
        if any(_iou(box,prior)>.4 for prior,_ in candidates):continue
        candidates.append((box,{**properties,'profile':'generic_fallback' if fallback else 'shape_assisted'}))
    if classifier is not None:
        candidates=[(box,{**props,'symbol_detector':'hog-rtrees-symbol-v1','symbol_model_sha256':metadata['weights_sha256']}) for box,props in candidates]
    if net is not None:
        found=neural_detect(high_resolution if high_resolution is not None else image)
        found=[(box,{**props,'profile':'generic_fallback' if fallback else 'shape_assisted'}) for box,props in found]
        candidates=merge(candidates,found,neural_meta.get('merge_mode','replace'))
    return candidates[:200],legacy_lines if fallback else axis_segments(image),limited or len(candidates)>200


def trace_adjacency(elements,segments,w,h,image=None,diagnostics=None):
    """Retain validated adjacency; attach separately reviewable recognition evidence.

    The experimental learned router did not beat v6 on held-out validation.
    Its paths may explain an existing candidate, but cannot add/approve edges.
    """
    details={}
    pairs=_trace_validated(elements,segments,w,h,image,details)
    if image is not None:
        from .topology_features import model
        if model()[0] is not None:
            from .topology import route
            _,recognized,ports=route(elements,segments,w,h,image)
            allowed={tuple(sorted(pair)) for pair in pairs}
            boxes={e['id']:e['bbox'] for e in elements}
            paths=[{**p,'from_bbox':boxes[p['from_id']],'to_bbox':boxes[p['to_id']]} for p in recognized.pop('paths',[]) if tuple(sorted((p['from_id'],p['to_id']))) in allowed]
            details.update({k:v for k,v in recognized.items() if k not in ('algorithm','crossing_policy','connections_limited','pair_budget_reached')})
            details.update(recognition_algorithm='line-port-junction-v7',routing_mode='validated_v6_with_recognition_evidence',
                           recognition_is_advisory=True,paths=paths,
                           recognition_pair_budget_reached=recognized.get('pair_budget_reached',False))
            for element in elements:
                if element['id'] in ports:
                    element.setdefault('properties',{})['ports']=[{k:v for k,v in p.items() if k!='pixel'} for p in ports[element['id']]]
                    element['properties']['topology_bbox']=list(element['bbox'])
    if diagnostics is not None:diagnostics.update(details)
    return pairs


def _trace_validated(elements,segments,w,h,image=None,diagnostics=None):
    symbols=[e for e in elements if e['kind']!='text']
    if symbols and all(e.get('properties',{}).get('profile')=='generic_fallback' for e in symbols):
        from .legacy_network import trace_adjacency as legacy_trace
        base=legacy_trace(elements,segments,w,h)
        if image is None:
            if diagnostics is not None:diagnostics.update({'algorithm':'legacy-components-v2','crossing_policy':'unresolved-review-required'})
            return base
        # Generic drawings retain every existing candidate. A second line pass
        # recovers short visible pipe stubs lost by component-based line masks.
        # This is still an unresolved-crossing candidate path, never auto-approved.
        from .network import axis_segments
        extra=set(legacy_trace(elements,axis_segments(image),w,h))-set(base)
        order={e['id']:i for i,e in enumerate(symbols)}
        extra=sorted(extra,key=lambda pair:tuple(sorted((order[pair[0]],order[pair[1]]))))
        budget=min(100,max(0,500-len(base)))
        if diagnostics is not None:diagnostics.update({'algorithm':'visible-components-v4','crossing_policy':'unresolved-review-required',
            'retained_legacy_connections':len(base),'supplemental_connections':min(len(extra),budget),'connections_limited':len(extra)>budget})
        return base+extra[:budget]
    from .network import port_graph
    result,details=port_graph(elements,segments,w,h,image)
    if diagnostics is not None:diagnostics.update(details)
    return result
