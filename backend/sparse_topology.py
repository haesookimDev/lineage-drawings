"""Sparse line/port/junction routing with reviewed point-context evidence."""
from collections import defaultdict
import heapq,math
import cv2,numpy as np
from .network import axis_segments,merge_segments,candidate_pairs,has_junction_dot
from .topology_context import prepare,rays,ray_batch,features,votes,model


def segments(image,ink,existing=(),recover=False):
 lines=axis_segments(image);extra=[]
 # Only bridge short repeated gaps on long aligned runs; text remains subject
 # to port, arm and point-context filters. No invisible semantic edges.
 for horizontal in (True,False):
  mask=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,3) if horizontal else (3,1),np.uint8));found=cv2.HoughLinesP(mask,1,np.pi/180,threshold=12,minLineLength=28,maxLineGap=10)
  if found is None:continue
  for a,b,c,d in found[:,0]:
   if not ((horizontal and abs(b-d)<=1) or (not horizontal and abs(a-c)<=1)):continue
   n=round(math.hypot(c-a,d-b))+1;xs=np.clip(np.rint(np.linspace(a,c,n)).astype(int),0,ink.shape[1]-1);ys=np.clip(np.rint(np.linspace(b,d,n)).astype(int),0,ink.shape[0]-1);hit=(ink[ys,xs]>0).astype(np.uint8);changes=np.flatnonzero(np.diff(np.r_[0,hit,0]));runs=changes[1::2]-changes[::2]
   if len(runs)>=3 and .3<float(hit.mean())<.95:extra.append([float(a),float(b),float(c),float(d)])
 lines=merge_segments(lines+extra)
 if recover:
  from .line_recovery import recover as recover_lines
  lines,n=recover_lines(ink,lines);return merge_segments(lines)[:5000],len(extra)+n
 return lines[:5000],len(extra)


def proposals(elements,lines,w,h):
 """Boundary intersections and blocking intervals share one coordinate frame."""
 symbols=[e for e in elements if e['kind']!='text' and e.get('properties',{}).get('connection_eligible',True)];intervals=[[] for _ in lines];raw=[]
 if not symbols:return raw,intervals
 boxes=np.asarray([e['bbox'] for e in symbols])*[w,h,w,h]
 for i,s in enumerate(lines):
  p=np.array(s[:2]);v=np.array(s[2:])-p;length=np.linalg.norm(v);lo=np.zeros(len(symbols));hi=np.ones(len(symbols))
  for axis in (0,1):
   if abs(v[axis])<1e-8:
    invalid=(p[axis]<boxes[:,axis]-2)|(p[axis]>boxes[:,axis+2]+2);lo[invalid]=1;hi[invalid]=0
   else:
    aa=(boxes[:,axis]-p[axis])/v[axis];bb=(boxes[:,axis+2]-p[axis])/v[axis];lo=np.maximum(lo,np.minimum(aa,bb));hi=np.minimum(hi,np.maximum(aa,bb))
  for j in np.flatnonzero(lo<=hi):
   e=symbols[j];box=boxes[j];l,r=float(lo[j]),float(hi[j]);intervals[i].append((l,r))
   for t,sign in ((l,-1),(r,1)):
    if (t*length if sign==-1 else (1-t)*length)<3 or len(raw)>=6000:continue
    q=p+t*v;normal=v*sign;side=('right' if normal[0]>0 else 'left') if abs(normal[0])>=abs(normal[1]) else ('bottom' if normal[1]>0 else 'top');a,b,c,d=box
    frac=(q[1]-b)/max(d-b,1) if side in ('left','right') else (q[0]-a)/max(c-a,1)
    raw.append({'element_id':e['id'],'pixel':q.tolist(),'side':side,'line':i,'t':float(t),'geometry':[float(frac),float(min(frac,1-frac)),math.log(max(c-a,1))/6,math.log(max(d-b,1))/6,float(length/100),0.]})
 return raw[:6000],intervals


def intersections(lines,intervals,ink,recover=False):
 rows=[];distance=cv2.distanceTransform((ink>0).astype(np.uint8),cv2.DIST_L2,3)
 for i,j in candidate_pairs(lines):
  p=np.array(lines[i][:2]);r=np.array(lines[i][2:])-p;q=np.array(lines[j][:2]);v=np.array(lines[j][2:])-q;lr=np.linalg.norm(r);lv=np.linalg.norm(v);cross=r[0]*v[1]-r[1]*v[0]
  if abs(cross)<1e-6:continue
  qp=q-p;t=float((qp[0]*v[1]-qp[1]*v[0])/cross);u=float((qp[0]*r[1]-qp[1]*r[0])/cross)
  if not(-3/lr<=t<=1+3/lr and -3/lv<=u<=1+3/lv):continue
  t,u=float(np.clip(t,0,1)),float(np.clip(u,0,1))
  if any(a+1e-4<t<b-1e-4 for a,b in intervals[i]) or any(a+1e-4<u<b-1e-4 for a,b in intervals[j]):continue
  point=p+t*r
  rows.append({'pixel':point.tolist(),'i':i,'j':j,'t':t,'u':u})
  if len(rows)>=6000:break
 support_all=ray_batch(ink,[p['pixel'] for p in rows]).reshape(-1,4,3) if rows else []
 dash=None
 if recover and rows:
  from .line_recovery import dashed_arms
  dash=dashed_arms(ink,[p['pixel'] for p in rows])
 for index,(p,support) in enumerate(zip(rows,support_all)):
  count=int((support[:,:2].mean(1)>=.55).sum());arm_mask=(support[:,0]>=.75)&(support[:,:2].mean(1)>=.55);strict=int((arm_mask|dash[index]).sum()) if dash is not None else int(arm_mask.sum());dot=has_junction_dot(distance,*p['pixel']);p.update(arms=strict,dot=dot,geometry=[float(count)/4,float(dot),*support[:,0].tolist()])
 return rows


def route(elements,existing,w,h,image,config=None,return_proposals=False,recognized=None):
 port_model,junction_model,meta=model();meta={**(meta or {}),**(config or {})}
 if isinstance(recognized,dict):gray,ink,lines,dashed,raw,intervals,junctions=(recognized[k] for k in ('gray','ink','lines','dashed','raw','intervals','junctions'))
 else:
  gray,ink=prepare(image);lines,dashed=segments(image,ink,existing,recover=meta.get('line_recovery',False))
  if meta.get('short_dash_recovery'):
   from .line_recovery import short_links
   links=short_links(elements,ink,w,h);lines=merge_segments(lines+links);dashed+=len(links)
  raw,intervals=proposals(elements,lines,w,h);junctions=intersections(lines,intervals,ink,recover=meta.get('line_recovery',False))
 if return_proposals:return gray,ink,raw,junctions,lines
 if port_model is None:return [],{},{}
 ps=votes(port_model,features(gray,ink,raw),2)[:,1] if recognized is None else recognized['port_scores'] if isinstance(recognized,dict) else recognized[0];js=votes(junction_model,features(gray,ink,junctions),4) if recognized is None else recognized['junction_scores'] if isinstance(recognized,dict) else recognized[1];accepted=[];ports=defaultdict(list)
 for p,score in zip(raw,ps):
  if score<meta.get('port_threshold',.5):continue
  ident=p['element_id'];x,y=p['pixel']
  if len(ports[ident])>=12 or any(np.linalg.norm(np.array(p['pixel'])-q['pixel'])<5 for q in ports[ident]):continue
  item={**p,'score':round(float(score),4),'point':[round(x/w,7),round(y/h,7)],'status':'needs_review','source':'sparse_context_and_line_contact'};ports[ident].append(item);accepted.append(item)
 details={'algorithm':'sparse-line-port-v8','recognition_algorithm':'sparse-line-port-v8','topology_model':'hog-rtrees-port-junction-v2','topology_model_sha256':meta.get('port_sha256'),'junction_model_sha256':meta.get('junction_sha256'),'recognition_tiles':0,'line_recovery':bool(meta.get('line_recovery')), 'port_count':len(accepted),'ports':{k:[{a:b for a,b in p.items() if a not in ('pixel','line','t','geometry','element_id')} for p in v] for k,v in ports.items()},'junctions':[],'routing_contacts':[],'lines':[],'paths':[],'line_segments':len(lines),'dashed_run_proposals':dashed,'connections_limited':False,'crossing_policy':'actual_arms_and_dot_with_reviewed_context; plain_X_separate'}
 coords={};along=[{} for _ in lines];adj=defaultdict(dict);owners=defaultdict(set);starts=defaultdict(set);port_scores={};serial=0
 def node(i,t):
  nonlocal serial
  t=round(float(t),6)
  if t not in along[i]:
   n=serial;serial+=1;along[i][t]=n;s=np.array(lines[i]);coords[n]=s[:2]+t*(s[2:]-s[:2])
  return along[i][t]
 def link(a,b):
  cost=float(np.linalg.norm(coords[a]-coords[b]));adj[a][b]=cost;adj[b][a]=cost
 for i in range(len(lines)):node(i,0);node(i,1)
 for p in accepted:
  n=node(p['line'],p['t']);ident=p['element_id'];owners[n].add(ident);starts[ident].add(n);port_scores[(ident,n)]=p['score']
 display_candidates=[];ignore=[]
 for e in elements:
  text=e['kind']=='text'
  if text and not any(ch.isalnum() for ch in str(e.get('label',''))):continue
  box=np.array(e['bbox'])*[w,h,w,h]
  if text and (box[2]-box[0]>180 or (box[2]-box[0])*(box[3]-box[1])>3000):continue
  ignore.append(box+np.array([-3,-3,3,3]) if not text else box)
 ignore=np.asarray(ignore).reshape(-1,4)
 for p,score in zip(junctions,js):
  x,y=p['pixel']
  original_arms=round(p['geometry'][0]*4);routing_type='dot' if p['dot'] and original_arms>=3 else 'tee' if original_arms==3 else 'plain_cross' if original_arms==4 else 'bend';routing_class={'tee':1,'dot':2,'plain_cross':3,'bend':0}[routing_type];routing_score=float(score[routing_class]);routing_connected=(routing_type=='bend' and original_arms==2) or (routing_type in ('tee','dot') and routing_score>=meta.get('junction_threshold',.2))
  if meta.get('strict_contacts'):
   # New links use the same actual arms, OCR/glyph clearance and vote threshold
   # as displayed contact evidence. A four-arm crossing without a dot stays open.
   arms=p['arms'];routing_type='dot' if p['dot'] and arms>=3 else 'tee' if arms==3 else 'plain_cross' if arms==4 else 'bend';routing_class={'tee':1,'dot':2,'plain_cross':3,'bend':0}[routing_type];routing_score=float(score[routing_class]);near_ignored=bool(len(ignore) and np.any((ignore[:,0]<=x)&(x<=ignore[:,2])&(ignore[:,1]<=y)&(y<=ignore[:,3])))
   routing_connected=bool((routing_type=='bend' and arms==2) or (routing_type in ('tee','dot') and not near_ignored and routing_score+1e-7>=max(meta.get('junction_threshold',.25),meta.get('junction_display_threshold',.45))))
  if routing_connected:
   link(node(p['i'],p['t']),node(p['j'],p['u']))
   if routing_type!='bend' and len(details['routing_contacts'])<500:details['routing_contacts'].append({'point':[round(x/w,7),round(y/h,7)],'type':routing_type,'score':round(routing_score,4),'status':'needs_review','source':'association_hypothesis_not_confirmed_junction'})
  if len(ignore) and np.any((ignore[:,0]<=x)&(x<=ignore[:,2])&(ignore[:,1]<=y)&(y<=ignore[:,3])):continue
  arms=p['arms'];typ='dot' if p['dot'] and arms>=3 else 'tee' if arms==3 else 'plain_cross' if arms==4 else 'bend';cls={'tee':1,'dot':2,'plain_cross':3,'bend':0}[typ];confidence=float(score[cls]);connected=typ=='bend' and arms==2 or typ in ('tee','dot') and confidence+1e-7>=meta.get('junction_threshold',.2)
  if typ=='bend' or confidence<.25:continue
  display_candidates.append({'point':[round(p['pixel'][0]/w,7),round(p['pixel'][1]/h,7)],'type':typ,'score':round(confidence,4),'connected':bool(connected),'status':'needs_review','source':'actual_arms_and_trained_context'})
 # Merge duplicate association hypotheses for inspection; routing links above
 # are not upgraded into confirmed detections by this display operation.
 contacts=[];contact_bins=defaultdict(list)
 for p in sorted(details['routing_contacts'],key=lambda p:(-p['score'],p['point'])):
  xy=np.array(p['point'])*[w,h];cell=tuple((xy//6).astype(int));near=[q for dx in (-1,0,1) for dy in (-1,0,1) for q in contact_bins.get((cell[0]+dx,cell[1]+dy),())]
  if any(np.linalg.norm(xy-q)<6 for q in near):continue
  contact_bins[cell].append(xy);contacts.append(p)
 details['routing_contacts']=contacts
 # Confidence-ranked spatial suppression is shared by validation and runtime.
 occupied=defaultdict(list)
 for p in sorted(display_candidates,key=lambda p:(-p['score'],p['point'])):
  xy=np.array(p['point'])*[w,h];cell=tuple((xy//6).astype(int));near=[q for dx in (-1,0,1) for dy in (-1,0,1) for q in occupied.get((cell[0]+dx,cell[1]+dy),())]
  if any(np.linalg.norm(xy-q)<6 for q in near):continue
  occupied[cell].append(xy)
  if p['score']>=meta.get('junction_display_threshold',.25) and len(details['junctions'])<500:details['junctions'].append(p)

 for i in range(len(lines)):
  items=sorted(along[i].items())
  for (ta,a),(tb,b) in zip(items,items[1:]):
   if not any(lo+1e-6<(ta+tb)/2<hi-1e-6 for lo,hi in intervals[i]):link(a,b)
 order={e['id']:i for i,e in enumerate(elements)};paths={}
 for ident,nn in starts.items():
  heap=[(0.,n) for n in sorted(nn)];heapq.heapify(heap);dist={n:0. for n in nn};prev={n:None for n in nn};origin={n:n for n in nn}
  while heap:
   length,n=heapq.heappop(heap)
   if length>dist[n]+1e-6:continue
   other=owners[n]-{ident}
   if other:
    for target in sorted(other,key=lambda k:order[k]):
     key=tuple(sorted((ident,target)));points=[];at=n
     while at is not None:points.append(coords[at]);at=prev[at]
     points.reverse();points=cv2.approxPolyDP(np.array(points,np.float32).reshape(-1,1,2),.6,False).reshape(-1,2).tolist() if len(points)>2 else points
     confidence=min(port_scores[(ident,origin[n])],port_scores[(target,n)])
     if len(points)>=2 and (key not in paths or length<paths[key]['length']):paths[key]={'from_id':ident,'to_id':target,'points':[[round(float(x/w),7),round(float(y/h),7)] for x,y in points[:127]+([points[-1]] if len(points)>127 else [])],'length':length,'score':confidence,'source':'sparse_visible_line_and_ports','status':'needs_review'}
    continue
   for nxt,cost in adj[n].items():
    value=length+cost
    if value<dist.get(nxt,float('inf'))-1e-6:dist[nxt]=value;prev[nxt]=n;origin[nxt]=origin[n];heapq.heappush(heap,(value,nxt))
 ranked=sorted(paths,key=lambda k:(-paths[k]['score'],tuple(sorted((order[k[0]],order[k[1]])))))[:500];details['paths']=[paths[k] for k in ranked];details['connections_limited']=len(paths)>500
 for a,b,c,d in lines[:750]:
  n=max(2,round(math.hypot(c-a,d-b)));xs=np.clip(np.rint(np.linspace(a,c,n)).astype(int),0,w-1);ys=np.clip(np.rint(np.linspace(b,d,n)).astype(int),0,h-1);hit=ink[ys,xs]>0;changes=np.flatnonzero(np.diff(np.r_[0,hit,0]));runs=len(changes)//2;coverage=float(hit.mean());style='dashed' if runs>=3 and coverage<.9 else 'solid'
  details['lines'].append({'points':[[round(a/w,7),round(b/h,7)],[round(c/w,7),round(d/h,7)]],'style':style,'score':round(coverage,4),'status':'needs_review'})
 public_ports={k:[{a:b for a,b in p.items() if a not in ('line','t','geometry','element_id')} for p in pp] for k,pp in ports.items()}
 return ranked,details,public_ports
