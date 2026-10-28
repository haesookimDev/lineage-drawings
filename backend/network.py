"""Port graph routing: endpoints/T-junctions/dots connect; plain Xs stay separate.

Policy is a reviewable drafting convention, not a universal engineering rule.
"""
from collections import defaultdict
import math
import cv2
import numpy as np


def merge_segments(segments):
    axes=[[],[]];diagonal=[]
    for s in segments[:5000]:
        if len(s)!=4 or not all(math.isfinite(v) for v in s):continue
        a,b,c,d=map(float,s)
        if abs(d-b)<=2 and abs(c-a)>=5:axes[0].append(((b+d)/2,min(a,c),max(a,c)))
        elif abs(c-a)<=2 and abs(d-b)>=5:axes[1].append(((a+c)/2,min(b,d),max(b,d)))
        elif math.hypot(c-a,d-b)>8:diagonal.append([a,b,c,d])
    result=[]
    for axis,lines in enumerate(axes):
        groups=[]
        for pos,lo,hi in sorted(lines):
            group=next((g for g in reversed(groups[-8:]) if abs(g[0]-pos)<=1.7 and lo<=g[2]+3 and hi>=g[1]-3),None)
            if group:
                group[0]=(group[0]+pos)/2;group[1]=min(group[1],lo);group[2]=max(group[2],hi)
            else:groups.append([pos,lo,hi])
        for pos,lo,hi in groups:result.append([lo,pos,hi,pos] if axis==0 else [pos,lo,pos,hi])
    return (result+diagonal)[:5000]


def axis_segments(image):
    gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY);bw=(gray<210).astype(np.uint8)*255
    segments=[]
    for horizontal in (True,False):
        mask=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,12) if horizontal else (12,1),np.uint8))
        lines=cv2.HoughLinesP(mask,1,np.pi/180,threshold=12,minLineLength=12,maxLineGap=3)
        if lines is not None:
            for a,b,c,d in lines[:,0]:
                if (horizontal and abs(b-d)<=2) or (not horizontal and abs(a-c)<=2):segments.append([float(a),float(b),float(c),float(d)])
    return merge_segments(segments[:5000])


def has_junction_dot(distance,x,y):
    if distance is None:return False
    h,w=distance.shape;x,y=round(x),round(y)
    if not 10<=x<w-10 or not 10<=y<h-10:return False
    arms=[float(distance[y,x+offset]) for offset in (-8,-6,6,8)]+[float(distance[y+offset,x]) for offset in (-8,-6,6,8)]
    arms=[v for v in arms if v>.1]
    return bool(arms) and distance[y,x]>max(1.5,float(np.median(arms))*1.6+.4)


def candidate_pairs(lines):
    """Bounded spatial bins avoid comparing every segment with every other one."""
    bins=defaultdict(list)
    for index,s in enumerate(lines):
        a,b,c,d=s
        x1,x2=math.floor((min(a,c)-3)/64),math.floor((max(a,c)+3)/64)
        y1,y2=math.floor((min(b,d)-3)/64),math.floor((max(b,d)+3)/64)
        if (x2-x1+1)*(y2-y1+1)<=256:
            cells=((x,y) for x in range(x1,x2+1) for y in range(y1,y2+1))
        else:
            steps=max(1,math.ceil(math.hypot(c-a,d-b)/32))
            cells={(math.floor((a+(c-a)*t/steps)/64)+dx,math.floor((b+(d-b)*t/steps)/64)+dy)
                   for t in range(steps+1) for dx in (-1,0,1) for dy in (-1,0,1)}
        for cell in cells:bins[cell].append(index)
    seen=set()
    for group in bins.values():
        for pos,i in enumerate(group):
            for j in group[pos+1:]:
                pair=(min(i,j),max(i,j))
                if pair in seen:continue
                seen.add(pair)
                if len(seen)>100_000:return
                yield pair


def port_graph(elements,segments,w,h,image=None):
    symbols=[e for e in elements if e['kind']!='text' and e.get('properties',{}).get('connection_eligible',True)][:500]
    lines=[np.asarray(s,dtype=float) for s in merge_segments(segments)]
    along=[{} for _ in lines];intervals=[[] for _ in lines]
    adjacent=defaultdict(set);owners=defaultdict(set);ports=defaultdict(set);serial=0
    diagnostics={'algorithm':'port-graph-v3','crossing_policy':'endpoint_or_dot','junction_candidates':0,'unmarked_crossings':0,'pair_budget_reached':False}
    def node(index,t):
        nonlocal serial
        key=round(float(t),5)
        if key not in along[index]:along[index][key]=serial;serial+=1
        return along[index][key]
    def link(a,b):adjacent[a].add(b);adjacent[b].add(a)
    if image is not None:
        gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY);distance=cv2.distanceTransform((gray<210).astype(np.uint8),cv2.DIST_L2,3)
    else:distance=None
    for index,s in enumerate(lines):
        p=s[:2];v=s[2:]-p;length=float(np.linalg.norm(v));node(index,0);node(index,1)
        for element in symbols:
            a,b,c,d=element['bbox'];box=(a*w,b*h,c*w,d*h);lo,hi=0.,1.
            for axis in (0,1):
                if abs(v[axis])<1e-8:
                    if p[axis]<box[axis]-3 or p[axis]>box[axis+2]+3:lo,hi=1,0;break
                else:
                    aa=(box[axis]-p[axis])/v[axis];bb=(box[axis+2]-p[axis])/v[axis]
                    lo=max(lo,min(aa,bb));hi=min(hi,max(aa,bb))
            if lo>hi:continue
            if lo*length>2:
                at=node(index,lo);owners[at].add(element['id']);ports[element['id']].add(at)
            if (1-hi)*length>2:
                at=node(index,hi);owners[at].add(element['id']);ports[element['id']].add(at)
            intervals[index].append((lo,hi))
    pair_count=0
    for i,j in candidate_pairs(lines):
        pair_count+=1
        s=lines[i];p=s[:2];r=s[2:]-p;lr=float(np.linalg.norm(r));q=lines[j][:2];v=lines[j][2:]-q;lv=float(np.linalg.norm(v))
        cross=r[0]*v[1]-r[1]*v[0]
        if abs(cross)<1e-6:continue
        qp=q-p;t=(qp[0]*v[1]-qp[1]*v[0])/cross;u=(qp[0]*r[1]-qp[1]*r[0])/cross
        if not(-2.5/lr<=t<=1+2.5/lr and -2.5/lv<=u<=1+2.5/lv):continue
        t=float(np.clip(t,0,1));u=float(np.clip(u,0,1));point=p+t*r
        if any(a+1e-4<t<b-1e-4 for a,b in intervals[i]) or any(a+1e-4<u<b-1e-4 for a,b in intervals[j]):continue
        a=node(i,t);b=node(j,u)
        endpoint=min(t,1-t)*lr<=3 or min(u,1-u)*lv<=3
        if endpoint or has_junction_dot(distance,*point):
            link(a,b);diagnostics['junction_candidates']+=1
        else:diagnostics['unmarked_crossings']+=1
    diagnostics['pair_budget_reached']=pair_count>=100_000
    for index,s in enumerate(lines):
        ordered=sorted(along[index].items())
        for (ta,a),(tb,b) in zip(ordered,ordered[1:]):
            midpoint=(ta+tb)/2
            if not any(lo+1e-6<midpoint<hi-1e-6 for lo,hi in intervals[index]):link(a,b)
    result=set()
    for start,starts in ports.items():
        todo=list(starts);seen=set(todo)
        while todo:
            at=todo.pop();others=owners[at]-{start}
            if others:
                result.update(tuple(sorted((start,end))) for end in others);continue
            for nxt in adjacent[at]-seen:seen.add(nxt);todo.append(nxt)
    diagnostics['connections_limited']=len(result)>500
    order={e['id']:i for i,e in enumerate(symbols)}
    ranked=sorted(result,key=lambda pair:tuple(sorted((order[pair[0]],order[pair[1]]))))
    return ranked[:500],diagnostics
