"""Visible line paths and explicit symbol ports; X continuity stays separate."""
from collections import defaultdict
import heapq,math
import cv2,numpy as np
from .network import line_ink,merge_segments,candidate_pairs,has_junction_dot
from .topology_features import maps,model


def _peak(channel,x,y,r=5):
 h,w=channel.shape;x,y=round(x),round(y);return float(channel[max(0,y-r):min(h,y+r+1),max(0,x-r):min(w,x+r+1)].max(initial=0))/255


def recognize_ports(elements,ink,prob,w,h,meta):
 result={};port_cut=meta.get('port_threshold',.25)
 for e in elements:
  if e['kind']=='text' or not e.get('properties',{}).get('connection_eligible',True):continue
  a,b,c,d=[round(v*[w,h][i%2]) for i,v in enumerate(e['bbox'])];a,b,c,d=max(0,a),max(0,b),min(w-1,c),min(h-1,d);ports=[]
  if c<=a or d<=b:continue
  for side,origin,normal,lo,hi in [('left',a,(-1,0),b,d),('right',c,(1,0),b,d),('top',b,(0,-1),a,c),('bottom',d,(0,1),a,c)]:
   vertical=normal[0]!=0;positions=np.arange(lo,hi+1);support=np.zeros(len(positions),np.float32)
   lookahead=16
   directional=[]
   for slope in (0,-1,1):
    hits=np.zeros(len(positions),np.float32)
    for dist in range(1,lookahead+1):
     x=np.full(len(positions),origin+normal[0]*dist) if vertical else positions+slope*dist;y=positions+slope*dist if vertical else np.full(len(positions),origin+normal[1]*dist);ok=(x>=0)&(x<w)&(y>=0)&(y<h);hits[ok]+=(ink[y[ok],x[ok]]>0)
    directional.append(hits)
   # Diagonal support needs a strong port head. Axis stems retain the original
   # visible-stroke gate, rather than turning every nearby character into a port.
   boundary=prob[2,np.clip(positions if vertical else np.full(len(positions),origin),0,h-1),np.clip(np.full(len(positions),origin) if vertical else positions,0,w-1)]/255
   support=np.maximum(directional[0],np.where(boundary>=.6,np.maximum(directional[1],directional[2]),0))
   selected=(support>=6).astype(np.uint8);_,_,stats,_=cv2.connectedComponentsWithStats(selected.reshape(-1,1))
   for _,start,_,length,_ in stats[1:]:
    if length>max(12,(hi-lo)*.4):continue
    along=float(lo+start+(length-1)/2);x,y=(float(origin),along) if vertical else (along,float(origin));score=_peak(prob[2],x,y);line_score=max(_peak(prob[0],x+normal[0]*5,y+normal[1]*5,2),_peak(prob[1],x+normal[0]*5,y+normal[1]*5,2))
    if score<port_cut and line_score<meta.get('line_threshold',.2)+.1:continue
    if any(math.hypot(x-p['pixel'][0],y-p['pixel'][1])<5 for p in ports):continue
    ports.append({'pixel':[x,y],'point':[round(x/w,7),round(y/h,7)],'side':side,'score':round(score,4),'visible_support':round(float(support[int(start):int(start+length)].mean()/lookahead),4),'source':'learned_heatmap_and_visible_stem','status':'needs_review'})
  ports=sorted(ports,key=lambda p:(-p['score'],p['pixel']))[:12];result[e['id']]=ports
 return result


def line_segments(image,elements,prob,meta):
 gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY);ink=line_ink(gray);h,w=ink.shape;cut=round(meta.get('line_threshold',.2)*255);mask=((prob[:2].max(0)>=cut)&(ink>0)).astype(np.uint8)*255
 # Prompted symbol interiors are excluded from routing, never transparently
 # passed through. Lines between equipment stop at the first equipment.
 for e in elements:
  if e['kind']=='text' or not e.get('properties',{}).get('connection_eligible',True):continue
  a,b,c,d=[round(v*[w,h][i%2]) for i,v in enumerate(e['bbox'])];mask[max(0,b):min(h,d+1),max(0,a):min(w,c+1)]=0
 # Thin narrow strokes without introducing a heavy skeleton dependency.
 work=mask.copy();skel=np.zeros_like(mask);kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
 for _ in range(8):
  eroded=cv2.erode(work,kernel);skel|=cv2.subtract(work,cv2.dilate(eroded,kernel));work=eroded
  if not np.any(work):break
 lines=[]
 for source,gap in [(skel,3),(np.where(prob[1]>=round(meta.get('dash_threshold',.6)*255),mask,0).astype(np.uint8),10)]:
  found=cv2.HoughLinesP(source,1,np.pi/180,threshold=7,minLineLength=7,maxLineGap=gap)
  if found is not None:lines.extend(found[:,0].astype(float).tolist())
 return merge_segments(lines[:5000]),ink


def route(elements,segments,w,h,image,recognized=None):
 net,meta=model()
 if net is None:return [],{},{}
 prob,features=maps(image,elements) if recognized is None else recognized;lines,ink=line_segments(image,elements,prob,meta);ports=recognize_ports(elements,ink,prob,w,h,meta)
 active={k:v for k,v in ports.items() if v};details={**features,'algorithm':'line-port-junction-v7','crossing_policy':'visible_endpoint_tee_or_dot; plain_X_separate','ports':{k:[{a:b for a,b in p.items() if a!='pixel'} for p in v] for k,v in active.items()},'junctions':[],'line_segments':len(lines),'port_count':sum(map(len,active.values())),'connections_limited':False}
 details['lines']=[]
 for a,b,c,d in lines[:750]:
  xs=np.clip(np.rint(np.linspace(a,c,16)).astype(int),0,w-1);ys=np.clip(np.rint(np.linspace(b,d,16)).astype(int),0,h-1)
  solid,dashed=[float(prob[k,ys,xs].mean()/255) for k in (0,1)]
  details['lines'].append({'points':[[round(a/w,7),round(b/h,7)],[round(c/w,7),round(d/h,7)]],'style':'dashed' if dashed>solid else 'solid','score':round(max(solid,dashed),4),'status':'needs_review'})
 details['lines_display_limited']=len(lines)>750
 along=[{} for _ in lines];coords={};adj=defaultdict(dict);owners=defaultdict(set);starts=defaultdict(set);serial=0
 def node(i,t):
  nonlocal serial
  t=round(float(np.clip(t,0,1)),6)
  if t not in along[i]:
   along[i][t]=serial;s=np.asarray(lines[i]);coords[serial]=s[:2]+t*(s[2:]-s[:2]);serial+=1
  return along[i][t]
 def link(a,b):
  cost=float(np.linalg.norm(coords[a]-coords[b]));adj[a][b]=cost;adj[b][a]=cost
 for i,line in enumerate(lines):node(i,0);node(i,1)
 endpoint_bins=defaultdict(set)
 for i,line in enumerate(lines):
  for p in (line[:2],line[2:]):endpoint_bins[(int(p[0]//32),int(p[1]//32))].add(i)
 for ident,pp in active.items():
  for port in pp:
   p=np.asarray(port['pixel']);near=[]
   nearby=set();cell=(int(p[0]//32),int(p[1]//32))
   for dx in (-1,0,1):
    for dy in (-1,0,1):nearby.update(endpoint_bins.get((cell[0]+dx,cell[1]+dy),()))
   for i in sorted(nearby):
    line=lines[i]
    s=np.asarray(line);v=s[2:]-s[:2];length=np.linalg.norm(v)
    if length<1:continue
    t=float(np.clip(np.dot(p-s[:2],v)/(length*length),0,1));q=s[:2]+t*v;dist=np.linalg.norm(p-q)
    if dist<=meta.get('port_snap',6) and min(t,1-t)*length<=10:near.append((float(dist),i,t))
   if near:
    best=min(p[0] for p in near)
    for dist,i,t in near:
     if dist>best+1.5:continue
     n=node(i,t);owners[n].add(ident);starts[ident].add(n)
 # Junction classification combines learned context with actual ink support.
 distance=cv2.distanceTransform((ink>0).astype(np.uint8),cv2.DIST_L2,3);seen_junctions=set();pair_budget=0
 for i,j in candidate_pairs(lines):
  pair_budget+=1;s=np.asarray(lines[i]);p=s[:2];r=s[2:]-p;q=np.asarray(lines[j][:2]);v=np.asarray(lines[j][2:])-q;lr=np.linalg.norm(r);lv=np.linalg.norm(v);cross=r[0]*v[1]-r[1]*v[0]
  if abs(cross)<1e-6:continue
  qp=q-p;t=float((qp[0]*v[1]-qp[1]*v[0])/cross);u=float((qp[0]*r[1]-qp[1]*r[0])/cross)
  if not(-3/lr<=t<=1+3/lr and -3/lv<=u<=1+3/lv):continue
  point=p+np.clip(t,0,1)*r;x,y=point;endpoint=min(t,1-t)*lr<=3 or min(u,1-u)*lv<=3
  if not(0<=x<w and 0<=y<h):continue
  tee,dot,plain=[_peak(prob[c],x,y) for c in (3,4,5)];bulge=has_junction_dot(distance,x,y)
  inferred=not endpoint and not bulge and dot>=meta.get('inferred_junction_threshold',1.1) and plain<.25
  connected=bool(endpoint or (bulge and dot>=meta.get('junction_threshold',.35)) or inferred)
  typ='tee' if connected and endpoint else 'inferred_junction' if inferred else 'dot' if connected else 'plain_cross';confidence=tee if typ=='tee' else dot if connected else plain
  key=(round(x/5),round(y/5),typ)
  if key not in seen_junctions and len(details['junctions'])<500:
   seen_junctions.add(key);details['junctions'].append({'point':[round(x/w,7),round(y/h,7)],'type':typ,'score':round(confidence,4),'connected':connected,'status':'needs_review','source':'heatmap_and_visible_line_geometry'})
  if connected:link(node(i,t),node(j,u))
 for i in range(len(lines)):
  ordered=sorted(along[i].items())
  for (_,a),(_,b) in zip(ordered,ordered[1:]):link(a,b)
 pairs=set();paths={};order={e['id']:i for i,e in enumerate(elements)}
 for ident,start_nodes in starts.items():
  heap=[(0.,n) for n in sorted(start_nodes)];heapq.heapify(heap);dist={n:0. for n in start_nodes};prev={n:None for n in start_nodes}
  while heap:
   length,n=heapq.heappop(heap)
   if length>dist[n]+1e-6:continue
   other=owners[n]-{ident}
   if other:
    for target in sorted(other,key=lambda k:order[k]):
     key=tuple(sorted((ident,target)));points=[];at=n
     while at is not None:points.append(coords[at]);at=prev[at]
     points.reverse();clean=[]
     for point in points:
      if not clean or np.linalg.norm(point-clean[-1])>.5:clean.append(point)
     if len(clean)>2:
      clean=cv2.approxPolyDP(np.array(clean,np.float32).reshape(-1,1,2),.75,False).reshape(-1,2).tolist()
     # Keep endpoints when unusually complex paths exceed the storage budget.
     if len(clean)>128:clean=[clean[i] for i in np.linspace(0,len(clean)-1,128).astype(int)]
     if key not in paths or length<paths[key][0]:paths[key]=(length,ident,target,[[round(float(p[0]/w),7),round(float(p[1]/h),7)] for p in clean])
     pairs.add(key)
    continue
   for nxt,cost in adj[n].items():
    value=length+cost
    if value<dist.get(nxt,float('inf'))-1e-6:dist[nxt]=value;prev[nxt]=n;heapq.heappush(heap,(value,nxt))
 ranked=sorted(pairs,key=lambda pair:tuple(sorted((order[pair[0]],order[pair[1]]))))[:500];details['connections_limited']=len(pairs)>500;details['pair_budget_reached']=pair_budget>=100000;details['paths']=[{'from_id':paths[k][1],'to_id':paths[k][2],'points':paths[k][3],'source':'visible_line_and_explicit_ports','status':'needs_review'} for k in ranked if len(paths[k][3])>=2]
 return ranked,details,ports
