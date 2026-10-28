"""Select topology configuration on fixed validation only; symbol model frozen."""
import argparse,hashlib,json,sys,time,copy
from pathlib import Path
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v6.data import sources
from evaluation.benchmark import load_extractor,load_gt,score,counts
from evaluation.v5.evaluate import targets
from backend import topology_features as features,topology
ap=argparse.ArgumentParser();ap.add_argument('--epoch',type=int,required=True);args=ap.parse_args();p=ROOT/'.benchmark-data/v7-models'/f'epoch-{args.epoch}.onnx';meta=json.loads(p.with_suffix('.json').read_text());net=cv2.dnn.readNetFromONNX(str(p));features.model=lambda:(net,meta);topology.model=lambda:(net,meta);cv2.setNumThreads(2)
base=load_extractor('844bf3b03ce34de575509bf66b531aca83ffba0d')[0];cache=ROOT/'.benchmark-data/v7-validation';cache.mkdir(exist_ok=True);rows=[]
for group,ident,region in sources('validation'):
 path=ROOT/'.benchmark-data'/group/(ident+'.png')
 if group=='pahala':path=ROOT/'.benchmark-data/v6-highres'/group/(ident+'.png')
 with Image.open(path) as raw:size=raw.size;im=raw.convert('RGB');high=im.copy();high.thumbnail((4800,4800));im.thumbnail((2400,2400))
 gt,truth=(region['nodes'],set(tuple(e) for e in region['edges'])) if region else load_gt(path.with_suffix('.graphml'),size)[:2]
 cached=cache/f'{group}-{ident}.json'
 if cached.exists():data=json.loads(cached.read_text());els,segments,old=data['elements'],data['segments'],data['edges']
 else:
  els,segments=base.raster_candidates(im,[],high_resolution=high);old=base.infer_connections(els,segments,*im.size,image=im);cached.write_text(json.dumps({'elements':els,'segments':segments,'edges':old})+'\n')
 recognized=features.maps(im,els);old_pairs={tuple(sorted((e['from_id'],e['to_id']))) for e in old};print('mapped',group,ident,flush=True)
 for tag,line,port,snap,inferred in [('balanced',.2,.25,6,1.1),('recall',.1,.15,8,1.1),('precision',.35,.4,4,1.1),('learned_cross',.2,.25,6,.8)]:
  meta.update(line_threshold=line,port_threshold=port,port_snap=snap,inferred_junction_threshold=inferred);t=time.perf_counter();pairs,details,ports=topology.route(els,segments,*im.size,im,recognized=recognized);fresh=set(pairs)
  for mode,pred_pairs in [('new',fresh),('union',fresh|old_pairs)]:
   order={e['id']:i for i,e in enumerate(els)};ranked=sorted(pred_pairs,key=lambda pair:(pair not in fresh,tuple(sorted((order[pair[0]],order[pair[1]])))))[:500];edges=[{'from_id':a,'to_id':b} for a,b in ranked];pp,ee=targets(els,edges,region);sc=score(pp,ee,gt,truth,.5);sc.pop('matched_ids');rows.append({'group':group,'id':ident,'configuration':tag+'/'+mode,'scores':sc,'ports':details['port_count'],'junctions':len(details['junctions']),'recognition_seconds':round(time.perf_counter()-t,4)})
  print(tag,'ports',details['port_count'],'segments',details['line_segments'],'edges',len(fresh),flush=True)
 aggregate={}
 for config in sorted({x['configuration'] for x in rows}):
  groups={}
  for g in sorted({x['group'] for x in rows}):
   rr=[x for x in rows if x['configuration']==config and x['group']==g];groups[g]=counts(**{k:sum(x['scores']['connections'][k] for x in rr) for k in ('tp','fp','fn')})
  aggregate[config]={'groups':groups,'objective':float(np.mean([d['f1'] for d in groups.values()]))}
 (ROOT/'evaluation/v7'/f'VALIDATION-{args.epoch}.json').write_text(json.dumps({'epoch':args.epoch,'weights_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':rows,'aggregate':aggregate,'objective':'Mean of connection F1 over3 declared validation source groups. Symbol boxes unchanged. No new-page data used.'},indent=2)+'\n')
print('RANKING',sorted([(d['objective'],k) for k,d in aggregate.items()],reverse=True),flush=True)
