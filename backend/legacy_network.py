"""Preserved v2 tracer for unsupported drawing conventions."""
from collections import defaultdict
import math
import cv2
import numpy as np

def trace_adjacency(elements, segments, w, h):
    """Trace visible line components with symbol interiors cut out as barriers.

    This stops traversal at intermediate equipment instead of predicting the
    transitive closure of an entire process. Return ID pairs; caller assigns
    review state. Four-way crossings remain ambiguous, never auto-approved.
    """
    symbols = [e for e in elements if e['kind']!='text'][:500]
    mask = np.zeros((h,w),np.uint8)
    for segment in segments[:5000]:
        if len(segment)!=4 or not all(math.isfinite(v) for v in segment):
            continue
        x1,y1,x2,y2 = segment
        cv2.line(mask,(round(x1),round(y1)),(round(x2),round(y2)),255,2)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    boxes = []
    for e in symbols:
        a,b,c,d = e['bbox']
        x1,y1,x2,y2 = max(0,int(a*w)),max(0,int(b*h)),min(w,int(c*w)+1),min(h,int(d*h)+1)
        mask[y1:y2,x1:x2] = 0
        boxes.append((x1,y1,x2,y2))
    _,labels,stats,_ = cv2.connectedComponentsWithStats(mask)
    owners = defaultdict(list)
    for e,(x1,y1,x2,y2) in zip(symbols,boxes):
        nearby = set(np.unique(labels[max(0,y1-4):min(h,y2+4),max(0,x1-4):min(w,x2+4)]))-{0}
        for component in nearby:
            if stats[component,cv2.CC_STAT_AREA]>=8:
                owners[int(component)].append(e['id'])
    seen,out = set(),[]
    for ids in owners.values():
        # Dense unresolved line networks must not create an unbounded clique.
        if len(ids)>16:
            continue
        for i,a in enumerate(ids):
            for b in ids[i+1:]:
                pair = tuple(sorted((a,b)))
                if pair not in seen:
                    seen.add(pair)
                    out.append((a,b))
                    if len(out)>=500:
                        return out
    return out
