"""Train sparse point forests from reviewed real labels + separate weak/exact data."""
from pathlib import Path
import sys,json,hashlib,random,time
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend import sparse_topology as route
from backend.topology_context import prepare,features,rays
from backend.network import has_junction_dot
from evaluation.v7.graph_data import sources,load
from evaluation.v7.procedural import scene
cv2.setNumThreads(4);cv2.setRNGSeed(20261030);rng=np.random.default_rng(20261030)
port_x=[];port_y=[];junction_x=[];junction_y=[];manifest=[];started=time.perf_counter()

def geo(ink,p):
 support=np.array(rays(ink,*p)).reshape(4,3);n=int((support[:,:2].mean(1)>=.55).sum());dt=cv2.distanceTransform((ink>0).astype(np.uint8),cv2.DIST_L2,3);return [n/4,float(has_junction_dot(dt,*p)),*support[:,0].tolist()]

def append(gray,ink,points,labels,kind,repeat=1,jitter=False):
 if not points:return
 xx=features(gray,ink,points);yy=np.array(labels,np.int32);(port_x if kind=='port' else junction_x).append(xx);(port_y if kind=='port' else junction_y).append(yy)
 if repeat>1:
  for _ in range(repeat-1):
   pp=[{**p,'pixel':(np.array(p['pixel'])+rng.uniform(-1.2,1.2,2)).tolist()} for p in points];(port_x if kind=='port' else junction_x).append(features(gray,ink,pp));(port_y if kind=='port' else junction_y).append(yy)

manual=json.loads((ROOT/'evaluation/v8/manual-development.json').read_text())['sheets']
for key,s in manual.items():
 if s['role']!='training':continue
 path=ROOT/'.benchmark-data'/s['group']/(s['id']+'.png');im=Image.open(path).convert('RGB');w,h=im.size;gray,ink=prepare(im);els=[{'id':n['id'],'kind':'unknown','bbox':n['bbox'],'properties':{}} for n in s['nodes']];_,_,raw,_,lines=route.route(els,[],w,h,im,return_proposals=True)
 truth={e['id']:[] for e in els}
 for p in s['ports']:truth[p['element_id']].append(np.array(p['point'])*[w,h])
 pp=[];yy=[]
 for p in raw:
  score=min((np.linalg.norm(np.array(p['pixel'])-q) for q in truth[p['element_id']]),default=999);pp.append(p);yy.append(int(score<=5))
 for e in els:
  box=np.array(e['bbox'])*[w,h,w,h];a,b,c,d=box
  # Sample all four boundaries to include outline and near-corner hard negatives.
  for side in ('left','right','top','bottom'):
   for t in np.linspace(.05,.95,10):
    xy=np.array([a if side=='left' else c,b+t*(d-b)]) if side in ('left','right') else np.array([a+t*(c-a),b if side=='top' else d]);dist=min((np.linalg.norm(xy-q) for q in truth[e['id']]),default=999)
    if dist<8:continue
    pp.append({'pixel':xy.tolist(),'side':side,'geometry':[t,min(t,1-t),np.log(max(c-a,1))/6,np.log(max(d-b,1))/6,.3,0.]});yy.append(0)
  for q in truth[e['id']]:
   side=min(('left','right','top','bottom'),key=lambda side:abs(q[0]-(a if side=='left' else c)) if side in ('left','right') else abs(q[1]-(b if side=='top' else d)));t=(q[1]-b)/max(d-b,1) if side in ('left','right') else (q[0]-a)/max(c-a,1)
   for _ in range(8):pp.append({'pixel':q.tolist(),'side':side,'geometry':[t,min(t,1-t),np.log(max(c-a,1))/6,np.log(max(d-b,1))/6,.3,0.]});yy.append(1)
 append(gray,ink,pp,yy,'port',repeat=3)
 pts=s.get('junctions',[])+s.get('point_samples',[]);jp=[{'pixel':(np.array(p['point'])*[w,h]).tolist()} for p in pts]
 for p in jp:p['geometry']=geo(ink,p['pixel'])
 append(gray,ink,jp,[{'none':0,'tee':1,'dot':2,'plain_cross':3}[p['type']] for p in pts],'junction',repeat=18)
 manifest.append({'kind':'manual-real-training','id':key,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'reviewed_ports':len(s['ports']),'reviewed_junction_points':len(pts),'port_training_rows_before_jitter':len(pp)})
 print('manual',key,len(pp),len(jp),flush=True)
# Extra real GraphML junction crops were reviewed against actual pixels.
rows=json.loads((ROOT/'evaluation/v8/reviewed-graph-points.json').read_text())
for ident in range(4):
 p=ROOT/'.benchmark-data/open100'/f'{ident}.png';im=Image.open(p).convert('RGB');im.thumbnail((2400,2400));w,h=im.size;gray,ink=prepare(im);rr=[x for x in rows if x['id']==str(ident)];pts=[{'pixel':(np.array(p['point'])*[w,h]).tolist()} for p in rr]
 for q in pts:q['geometry']=geo(ink,q['pixel'])
 append(gray,ink,pts,[{'none':0,'tee':1}[x['type']] for x in rr],'junction',repeat=14);manifest.append({'kind':'reviewed-real-graph-points','id':str(ident),'count':len(rr),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
# GraphML port supervision remains weak, unlike manually reviewed points above.
for path in sources('training'):
 im=Image.open(path).convert('RGB');size=im.size;im.thumbnail((2400,2400));w,h=im.size;gray,ink=prepare(im);nodes,edges,adj=load(path.with_suffix('.graphml'),size);scale=np.array([w/size[0],h/size[1]]);objects={k for k,n in nodes.items() if n['label'] in ('valve','instrumentation','equipment','inlet/outlet','pump','tank')};els=[{'id':k,'kind':'unknown','bbox':(nodes[k]['box']/[*size,*size]).tolist(),'properties':{}} for k in sorted(objects)];truth={k:[nodes[q]['point']*scale for q in adj[k] if nodes[q]['label']=='connector'] for k in sorted(objects)};_,_,raw,_,_=route.route(els,[],w,h,im,return_proposals=True);yy=[int(min((np.linalg.norm(np.array(p['pixel'])-q) for q in truth[p['element_id']]),default=999)<=5) for p in raw];chosen=[]
 for label in (0,1):
  ids=np.flatnonzero(np.array(yy)==label);rng.shuffle(ids);chosen.extend(ids[:200])
 append(gray,ink,[raw[i] for i in chosen],[yy[i] for i in chosen],'port');manifest.append({'kind':'weak-GraphML-port','id':str(path.relative_to(ROOT)),'rows':len(chosen),'positives':sum(yy[i] for i in chosen),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 print('graph',path.name,len(chosen),flush=True)
# Exact procedural geometry supplies dots/plain-X and photometric variation.
for seed in range(20281030,20281330):
 gray,prompt,target=scene(seed);ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,7)
 for ch,lab in ((3,1),(4,2),(5,3)):
  count,_,stats,centers=cv2.connectedComponentsWithStats(target[ch])
  for pt in centers[1:]:
   # Include visibly dotted T junctions, not just four-way dots.
   if lab==1 and seed%2==0:
    cv2.circle(gray,tuple(np.round(pt).astype(int)),2 if seed%4 else 3,40,-1);ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,7);lab=2
   p={'pixel':pt.tolist(),'geometry':geo(ink,pt)};append(gray,ink,[p],[lab],'junction',repeat=2)
 # Background / two-arm corner / glyph outline negatives, avoid true junctions.
 forbidden=np.maximum.reduce(target[3:])
 pts=[]
 for _ in range(6):
  x,y=rng.integers(10,246,2)
  if forbidden[max(0,y-8):min(256,y+9),max(0,x-8):min(256,x+9)].any():continue
  pts.append({'pixel':[float(x),float(y)],'geometry':geo(ink,[x,y])})
 append(gray,ink,pts,[0]*len(pts),'junction')

def fit(x,y,name):
 x=np.concatenate(x);y=np.concatenate(y);order=rng.permutation(len(y));x,y=x[order],y[order];net=cv2.ml.RTrees_create();net.setMaxDepth(18);net.setMinSampleCount(3);net.setMaxCategories(4);net.setActiveVarCount(0);net.setCalculateVarImportance(False);net.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER,160,0));net.train(x,cv2.ml.ROW_SAMPLE,y);out=ROOT/'.benchmark-data/v8-models';out.mkdir(exist_ok=True);path=out/(name+'.xml.gz');net.save(str(path));print('fit',name,x.shape,dict(zip(*np.unique(y,return_counts=True))),flush=True);return {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(y),'class_counts':{str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))}}
a=fit(port_x,port_y,'port-context');b=fit(junction_x,junction_y,'junction-context');meta={'model':'hog-rtrees-port-junction-v2','port_file':a['file'],'port_sha256':a['sha256'],'junction_file':b['file'],'junction_sha256':b['sha256'],'port_threshold':.5,'junction_threshold':.2,'junction_display_threshold':.25,'extra_edge_threshold':.65,'routing_mode':'hybrid','score_note':'Uncalibrated tree vote fractions, no automatic engineering approval','seed':20261030};(ROOT/'.benchmark-data/v8-models/topology-context.json').write_text(json.dumps(meta,indent=2)+'\n');(ROOT/'evaluation/v8/TRAINING.json').write_text(json.dumps({'manual_annotations_sha256':hashlib.sha256((ROOT/'evaluation/v8/manual-development.json').read_bytes()).hexdigest(),'reviewed_graph_points_sha256':hashlib.sha256((ROOT/'evaluation/v8/reviewed-graph-points.json').read_bytes()).hexdigest(),'models':{'port':a,'junction':b},'sources':manifest,'seconds':time.perf_counter()-started,'features':'Two32px HOG/ink grids from24/48px point context, oriented for ports, local actual-arm coverage and six candidate geometry values. No project id/page coordinates in feature vector.','cpu_threads':4,'trees':160,'max_depth':18,'procedural_seeds':[20281030,20281329],'training_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
