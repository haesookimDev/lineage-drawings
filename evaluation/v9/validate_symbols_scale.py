"""Development-only glyph shortlist. No new-project files are opened."""
import json,sys,statistics,time,os
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v9.data import validation
from evaluation.benchmark import load_extractor,score,counts
from evaluation.v5.evaluate import targets
from backend import neural_symbols as neural
cv2.setNumThreads(2);baseline=load_extractor('c9cef5afc6363e30c86e9ab4a5fe17dd68313e11')[0];cache=ROOT/'.benchmark-data/v9-validation-scale';cache.mkdir(exist_ok=True);rows=[]
paths={f'scale{i}':ROOT/f'.benchmark-data/v9b-models/epoch-{i}.onnx' for i in (4,8)}
models={k:(cv2.dnn.readNetFromONNX(str(p)),json.loads(p.with_suffix('.json').read_text())) for k,p in paths.items()}
for group,ident,region,high,gt,truth in validation():
 im=high.copy();im.thumbnail((2400,2400));p=cache/f'{group}-{ident}-base.json'
 if not p.exists():
  els,segs=baseline.raster_candidates(im,[],high_resolution=high);rec={};edges=baseline.infer_connections(els,segs,*im.size,image=im,diagnostics=rec);p.write_text(json.dumps({'elements':els,'segments':segs,'edges':edges,'recognition':rec}))
 d=json.loads(p.read_text());sym=[e for e in d['elements'] if e['kind']!='text'];profile=sym[0].get('properties',{}).get('profile','generic_fallback') if sym else 'generic_fallback'
 for name,(net,meta) in models.items():
  for detail in (0,6):
   cfg={**meta,'detail_tiles':detail,**({'detail_hint_mode':'ports'} if os.getenv('LINEAGE_V9_PORT_HINTS') else {})};neural.model=lambda:(net,cfg);q=cache/f'{group}-{ident}-{name}-d{detail}.json'
   if not q.exists():
    t=time.perf_counter();found=neural.detect(high,threshold=.05);q.write_text(json.dumps({'found':found,'seconds':time.perf_counter()-t}))
   found=json.loads(q.read_text())['found']
   for cut in (.3,.45,.6):
    els=[{'id':str(i),'kind':'unknown','bbox':box,'source':'neural_symbol_candidate','properties':{**props,'profile':profile}} for i,(box,props) in enumerate(found) if props['detector_score']>=cut];pp,_=targets(els,[],region);sc=score(pp,[],gt,truth,.5);sc.pop('matched_ids');rows.append({'group':group,'id':ident,'configuration':f'{name}/{detail}/{cut}','localization':sc['localization']})
 print(group,ident,'done',flush=True)
agg={}
for key in sorted({r['configuration'] for r in rows}):
 groups={g:counts(**{k:sum(r['localization'][k] for r in rows if r['configuration']==key and r['group']==g) for k in ('tp','fp','fn')}) for g in sorted({r['group'] for r in rows})};agg[key]={'groups':groups,'objective':statistics.mean(d['f1'] for d in groups.values())}
rank=sorted(agg,key=lambda k:(agg[k]['objective'],k),reverse=True);chosen=[]
for k in rank:
 if k.split('/')[0] not in [x.split('/')[0] for x in chosen]:chosen.append(k)
 if len(chosen)==2:break
out={'policy':'Glyph shortlist by equal-source macro localization F1, two distinct checkpoint families; original baseline always eligible for final graph selection. Final graph objective is macro(connectionF1+.25localizationF1). Fixed before unseen-project inference.','rows':rows,'aggregate':agg,'shortlist':chosen,'baseline':'base/0/0.6'};(ROOT/'evaluation/v9/SYMBOL-SCALE-VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print('SHORTLIST',chosen,[(k,agg[k]['objective']) for k in rank[:8]],flush=True)
