#!/usr/bin/env python3
"""Checkpoint / threshold selection reads predeclared validation pages only."""
import argparse,hashlib,json,statistics,sys,time
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v6.data import sources,read
from evaluation.benchmark import load_extractor,load_gt,score,counts
from evaluation.v5.evaluate import targets
from backend import neural_symbols as neural
ap=argparse.ArgumentParser();ap.add_argument('--epoch',type=int,required=True);args=ap.parse_args();directory=ROOT/'.benchmark-data/v6-models';path=directory/f'epoch-{args.epoch}.onnx';meta=json.loads(path.with_suffix('.json').read_text());net=cv2.dnn.readNetFromONNX(str(path));neural.model=lambda:(net,meta);cv2.setNumThreads(2)
base=load_extractor('aef3ca96f8eb6d65ac7c3405b0be6c5da21ffa8b')[0];rows=[]
for group,ident,region in sources('validation'):
 image,bb=read(group,ident,region);high=image.copy();image.thumbnail((2400,2400));original=__import__('PIL').Image.open(ROOT/'.benchmark-data'/group/(ident+'.png')).size
 gt,truth=(region['nodes'],set(tuple(e) for e in region['edges'])) if region else load_gt(ROOT/'.benchmark-data'/group/(ident+'.graphml'),original)[:2]
 elements,lines=base.raster_candidates(image,[]);symbols=[e for e in elements if e['kind']!='text'];texts=[e for e in elements if e['kind']=='text'];fallback=not symbols or all(e.get('properties',{}).get('profile')=='generic_fallback' for e in symbols)
 found=neural.detect(high,threshold=.05);found=[(b,{**p,'profile':'generic_fallback' if fallback else 'shape_assisted'}) for b,p in found];print('inferred',group,ident,'neural',len(found),flush=True)
 for threshold in (.15,.3,.45,.6):
  for mode in ('replace','add','neural_only'):
   combined=neural.merge([(e['bbox'],e['properties']) for e in symbols],[(b,p) for b,p in found if p['detector_score']>=threshold],mode)[:200]
   ee=[base.element('candidate','unknown',b,'neural_symbol_candidate' if p.get('geometry')=='neural_symbol' else 'image_candidate',p) for b,p in combined];edges=base.infer_connections(texts+ee,lines,*image.size,image=image);pred,edge=targets(ee,edges,region);sc=score(pred,edge,gt,truth,.5);sc.pop('matched_ids');rows.append({'group':group,'id':ident,'threshold':threshold,'merge_mode':mode,'scores':sc})
 aggregates={}
 for cut in (.15,.3,.45,.6):
  for mode in ('replace','add','neural_only'):
   key=f'{cut}/{mode}';gg={}
   for g in sorted({r['group'] for r in rows}):
    rr=[r for r in rows if r['group']==g and r['threshold']==cut and r['merge_mode']==mode];gg[g]={m:counts(**{k:sum(r['scores'][m][k] for r in rr) for k in ('tp','fp','fn')}) for m in ('localization','connections')}
   aggregates[key]={'groups':gg,'objective':statistics.mean(v['connections']['f1']+.25*v['localization']['f1'] for v in gg.values())}
 (ROOT/'evaluation/v6'/f'VALIDATION-{args.epoch}.json').write_text(json.dumps({'epoch':args.epoch,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'objective':'Mean across validation source groups: connection F1 + 0.25 localization F1. No training or new-page test scores used.','rows':rows,'aggregate':aggregates},indent=2)+'\n')
print('RANKING',sorted([(v['objective'],k) for k,v in aggregates.items()],reverse=True)[:6],flush=True)
