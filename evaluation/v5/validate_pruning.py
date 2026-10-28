#!/usr/bin/env python3
"""Threshold selection on predeclared validation sheets, not new projects."""
from pathlib import Path
import json,sys,time,copy
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend.learned_symbols import detect,score_boxes
from backend.symbols import iou
from backend.network import axis_segments
from backend.extract import element,infer_connections
from evaluation.benchmark import load_extractor,load_gt,score,counts
from evaluation.v4.evaluate import targets
from collections import Counter,defaultdict

base=load_extractor('ab04ed8e529d8c605995cd5e8fbaa87b2b4ffeaf')[0]
plan=json.loads((ROOT/'evaluation/v5/PLAN.json').read_text());labels=json.loads((ROOT/'evaluation/v5/real-labels.json').read_text())['sheets'];rows=[]
work=[('open100',i,None) for i in plan['threshold_validation']['open100']]+[('pahala',i,labels['pahala-'+i]) for i in plan['threshold_validation']['pahala']]+[('v5-synthetic',Path(i).stem,None) for i in plan['threshold_validation']['synthetic_members']]
for group,ident,region in work:
 path=ROOT/'.benchmark-data'/group/(ident+'.png');im=Image.open(path).convert('RGB');size=im.size;im.thumbnail((2400,2400));gt,truth=(region['nodes'],set(tuple(e) for e in region['edges'])) if region else load_gt(path.with_suffix('.graphml'),size)[:2]
 elements,lines=base.raster_candidates(im,[]);symbols=[e for e in elements if e['kind']!='text'];texts=[e for e in elements if e['kind']=='text'];fallback=not symbols or all(e.get('properties',{}).get('profile')=='generic_fallback' for e in symbols)
 if not fallback:lines=axis_segments(im)
 t=time.perf_counter();candidates=detect(im,[e['bbox'] for e in symbols],.6);detector_time=time.perf_counter()-t
 base_scores=score_boxes(im,[e['bbox'] for e in symbols])
 for cut in [0.,.1,.2,.3,.4]:
  retained=[e for i,e in enumerate(symbols) if base_scores[i]>=cut];new=[]
  if cut is not None:
   for b,p in candidates:
    if p['detector_score']<.85 or any(iou(b,e['bbox'])>.4 for e in retained+new):continue
    new.append(element('model','unknown',b,'learned_symbol_candidate',{**p,'profile':'generic_fallback' if fallback else 'shape_assisted'}))
  combined=(retained+new)[:200];edges=infer_connections(texts+combined,lines,*im.size,image=im);pred,ee=targets(combined,edges,region);scores=score(pred,ee,gt,truth,.5);scores.pop('matched_ids');row={'group':group,'id':ident,'threshold':cut,'added':len(new),'detector_seconds':round(detector_time,4),'scores':scores};rows.append(row);print(group,ident,cut,'add',len(new),'locR/F',round(scores['localization']['recall'],3),round(scores['localization']['f1'],3),'edgeR/F',round(scores['connections']['recall'],3),round(scores['connections']['f1'],3),flush=True)
aggregates={}
for cut in [0.,.1,.2,.3,.4]:
 groups={}
 for group in ['open100','pahala','v5-synthetic']:
  rr=[r for r in rows if r['threshold']==cut and r['group']==group];groups[group]={}
  for metric in ['localization','connections']:
   groups[group][metric]=counts(**{k:sum(r['scores'][metric][k] for r in rr) for k in ['tp','fp','fn']})
 aggregates[str(cut)]=groups
(ROOT/'evaluation/v5/VALIDATION-PRUNING.json').write_text(json.dumps({'rows':rows,'aggregate':aggregates,'method':'Validation router ablation, not immutable v4 deployment baseline. Base candidate pruning thresholds with added candidate threshold .85. Same v4 OCR/base proposals/lines cached per sheet; trained candidates and final connection inference rescored per threshold. New project labels/pixels not loaded.'},indent=2)+'\n');print(json.dumps(aggregates,indent=2))
