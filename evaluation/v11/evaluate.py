"""Frozen White Rose project pilot; identical in-memory inputs, cold/warm separate."""
import hashlib,json,statistics,sys,time,copy
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from PIL import Image
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_extractor,score,counts,infer
from evaluation.v5.evaluate import targets,canonical
cv2.setNumThreads(2)
labels=ROOT/'evaluation/v11/white-rose-labels.json';sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'backend').rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.onnx','.npz') and '__pycache__' not in p.parts and p.name!='workflows.py' and p.name!='main.py' and p.name!='db.py'}
freeze={'created_at':datetime.now(timezone.utc).isoformat(),'baseline':'be552914969fe311bae45b9ed7a40e6c850110d1','labels_sha256':hashlib.sha256(labels.read_bytes()).hexdigest(),'sources':sources,'protocol':'Source-only AI annotation frozen before first extraction on White Rose. Two small ROIs, class-agnostic IoU.5. All full-page glyphs extracted; only frozen ROI/ignore used by scoring. No test tuning. Cold clears v11 OCR cache; warm immediately reuses identical image, not new-document latency.3 alternating paired repeats.'};(ROOT/'evaluation/v11/FROZEN-EXTRACTION.json').write_text(json.dumps(freeze,indent=2)+'\n')
variants={'v10':load_extractor(freeze['baseline'])[0],'v11':load_extractor('working')[0]};result={'freeze':freeze,'rows':[]}
for key,region in json.loads(labels.read_text())['sheets'].items():
 path=ROOT/'.benchmark-data/v11-held/white-rose'/(region['id']+'-4800.png');assert hashlib.sha256(path.read_bytes()).hexdigest()==region['image_sha256']
 with Image.open(path) as raw:hi=raw.convert('RGB');im=hi.copy();im.thumbnail((2400,2400))
 row={'key':key,'role':region['role'],'gt_symbols':len(region['nodes']),'gt_edges':len(region['edges']),'variants':{}};first={};times={k:[] for k in variants};warm=[]
 for repeat in range(3):
  for name in (list(variants) if repeat%2==0 else list(variants)[::-1]):
   ex=variants[name];warnings=[]
   if hasattr(ex,'clear_ocr_cache'):ex.clear_ocr_cache()
   t=time.perf_counter();els,segs=ex.raster_candidates(im,warnings,high_resolution=hi);edges=infer(ex,els,segs,im);elapsed=time.perf_counter()-t;times[name].append(elapsed)
   repr=canonical(els,edges)
   if name in first:assert repr==first[name]
   else:
    first[name]=repr;pp,ee=targets(els,edges,region);ss=score(pp,ee,region['nodes'],{tuple(e) for e in region['edges']},.5);mapping=ss.pop('matched_ids');mapped={tuple(sorted((mapping.get(e['from_id'],'!'+e['from_id']),mapping.get(e['to_id'],'!'+e['to_id'])))) for e in ee};matched=set(mapping.values());causes=Counter('recovered' if tuple(e) in mapped else 'missing_symbol' if not set(e)<=matched else 'routing' for e in region['edges']);row['variants'][name]={'scores':ss,'miss_causes':dict(causes),'ocr_words':sum(e['kind']=='text' for e in els),'warnings':warnings}
    out={'elements':[{k:v for k,v in e.items() if k!='label'} for e in els if e['kind']!='text'],'connections':edges};(ROOT/f'evaluation/v11/{key}-{name}-geometry.json').write_text(json.dumps(out,indent=2)+'\n')
   if name=='v11':
    t=time.perf_counter();we,ws=ex.raster_candidates(im,[],high_resolution=hi);wc=infer(ex,we,ws,im);warm.append(time.perf_counter()-t);assert canonical(we,wc)==repr
  print(key,repeat,times,flush=True)
 for name in variants:row['variants'][name]['cold_seconds']=times[name];row['variants'][name]['cold_median']=statistics.median(times[name])
 row['variants']['v11']['identical_image_warm_seconds']=warm;row['variants']['v11']['warm_median']=statistics.median(warm);result['rows'].append(row)
result['aggregate']={}
for name in variants:
 result['aggregate'][name]={m:counts(**{k:sum(r['variants'][name]['scores'][m][k] for r in result['rows']) for k in ('tp','fp','fn')}) for m in ('localization','connections')};result['aggregate'][name]['mean_page_cold_median']=statistics.mean(r['variants'][name]['cold_median'] for r in result['rows'])
result['aggregate']['v11']['mean_identical_image_warm_median']=statistics.mean(r['variants']['v11']['warm_median'] for r in result['rows']);result['limitations']='45 glyphs/56 edges in2 source-only AI-annotated ROIs. No expert certification; no full-page or project-wide quality claim. No gradient training.3 sequential paired repeats, not service SLA or throughput.'
for name,digest in sources.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
(ROOT/'evaluation/v11/HELD-OUT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate'],indent=2),flush=True)
