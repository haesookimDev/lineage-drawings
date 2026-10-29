"""Bounded short-dash experiment, historical development labels only."""
import copy,json,sys,time
from pathlib import Path
from PIL import Image
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_gt,score,counts
from evaluation.v5.evaluate import targets
from backend.extract import raster_candidates,infer_connections
from backend.topology_context import model
cv2.setNumThreads(2);meta=model()[2];original=copy.deepcopy(meta);work=[];rows=[]
labels=json.loads((ROOT/'evaluation/v10/expanded-labels.json').read_text())['sheets']
for key,r in labels.items():work.append((key,r,ROOT/'.benchmark-data/v9-held'/r['group']/(r['id']+'-4800.png'),r['nodes'],set(tuple(e) for e in r['edges'])))
for i in range(6,12):
    p=ROOT/'.benchmark-data/open100'/f'{i}.png'
    with Image.open(p) as im:gt,truth,*_=load_gt(p.with_suffix('.graphml'),im.size)
    work.append((f'open100-{i}',None,p,gt,truth))
for key,region,path,gt,truth in work:
    with Image.open(path) as raw:high=raw.convert('RGB');im=high.copy();high.thumbnail((4800,4800));im.thumbnail((2400,2400))
    elements,segments=raster_candidates(im,[],high_resolution=high);row={'key':key,'group':'expanded-development' if region else 'open100-repair-development','variants':{}}
    for flag in (False,True):
        meta.clear();meta.update(original);meta['short_dash_recovery']=flag;els=copy.deepcopy(elements);rec={};started=time.perf_counter();edges=infer_connections(els,segments,*im.size,image=im,diagnostics=rec);elapsed=time.perf_counter()-started;pp,ee=targets(els,edges,region);result=score(pp,ee,gt,truth,.5);result.pop('matched_ids');row['variants']['short' if flag else 'baseline']={'scores':result,'routing_seconds':elapsed}
    rows.append(row);print(key,row['variants'],flush=True)
meta.clear();meta.update(original)
aggregate={}
for group in sorted({r['group'] for r in rows}):
    aggregate[group]={}
    for v in ('baseline','short'):
        rr=[r['variants'][v] for r in rows if r['group']==group];aggregate[group][v]={m:counts(**{k:sum(r['scores'][m][k] for r in rr) for k in ('tp','fp','fn')}) for m in ('localization','connections')};aggregate[group][v]['routing_seconds']=sum(r['routing_seconds'] for r in rr)/len(rr)
eligible=all(a['short']['connections']['recall']>=a['baseline']['connections']['recall'] and a['short']['connections']['f1']>=a['baseline']['connections']['f1'] and a['short']['routing_seconds']<=a['baseline']['routing_seconds']*1.25 for a in aggregate.values()) and any(a['short']['connections']['tp']>a['baseline']['connections']['tp'] for a in aggregate.values())
result={'role':'Development only. Previous OPEN1006-11 and v10 expanded78 labels explicitly reused; no White Rose source or labels read. Single bounded predeclared candidate; no test-driven retuning.','rows':rows,'aggregate':aggregate,'selected_short_dash_recovery':eligible,'selection_policy':'Both development groups recall/F1 nondecreasing, at least1 recovered edge, routing cost <=25% extra. Otherwise retain previous tracker.'}
(ROOT/'evaluation/v11/DEVELOPMENT.json').write_text(json.dumps(result,indent=2)+'\n')
if eligible:
    original['short_dash_recovery']=True;original['routing_revision']='v11-validated-short-dashes';(ROOT/'backend/assets/topology-context.json').write_text(json.dumps(original,indent=2)+'\n')
print('SELECTED',eligible,flush=True)
