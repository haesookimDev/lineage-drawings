"""First new-project evaluation after selection; no target-conditioned tuning."""
import hashlib,json,statistics,sys,time
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_extractor,score,counts,infer
from evaluation.v5.evaluate import targets,canonical
from evaluation.v10.data import images
from backend.legend import match
from backend.extract import element
from backend.geometry import _iou
cv2.setNumThreads(2)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
labels=ROOT/'evaluation/v12/new-project-labels.json'
templates=[{'project':'peterhead','page':12,'bbox_pixels':[528,733,584,786],'id':'field','name':'Field mounted instrument','mode':'outline'}, {'project':'peterhead','page':12,'bbox_pixels':[528,864,584,915],'id':'panel','name':'Local panel instrument','mode':'full'}, {'project':'epa-wtp','page':8,'bbox_pixels':[803,347,843,374],'id':'ball','name':'Ball valve','mode':'full'}]
for t in templates:
 p=ROOT/f'.benchmark-data/v12-held/{t["project"]}/p{t["page"]}-4800.png';view,_=images(p);t.update(source_image_sha256=sha(p),viewer_size=list(view.size));t['image']=view.crop(t['bbox_pixels'])
legend_sources=[{k:v for k,v in t.items() if k!='image'} for t in templates];(ROOT/'evaluation/v12/LEGEND-SOURCES.json').write_text(json.dumps({'role':'Separate source legend pages, visually cropped before first extraction. Not target answer crops. Source pixels serve as runtime context only. Fixed threshold .48, max20 candidates; rotation0/90/180/270 and5scales. No test threshold selection.','templates':legend_sources},indent=2)+'\n')
exs={'v11':load_extractor('7e1c349c41c0bf92340d9c53d99629bb8bd4c8a3')[0],'v12':load_extractor('working')[0]}
source_hashes=load_extractor('working')[2]
# UI/storage source may still be polished; freeze extraction modules and assets.
source_hashes={k:v for k,v in source_hashes.items() if Path(k).name not in ('main.py','db.py','workflows.py','assistance.py')}
freeze={'created_at':datetime.now(timezone.utc).isoformat(),'baseline':'7e1c349c41c0bf92340d9c53d99629bb8bd4c8a3','labels_sha256':sha(labels),'legend_sources_sha256':sha(ROOT/'evaluation/v12/LEGEND-SOURCES.json'),'release_selection_sha256':sha(ROOT/'evaluation/v12/RELEASE-SELECTION.json'),'sources':source_hashes,'protocol':'4 new project ROIs,45glyph53edge, single AI annotation. No expert certification.3 paired cold repetitions, alternating order; exact same inputs max4800 source/max2400 OCR/viewer. Each variant clears OCR cache before each repeat. Legend condition separately adds source-legend proposals and re-infers paths without target answers. Code/threshold fixed before first extraction. In-memory OCR+NN+connection extraction only, excluding decode/render/upload/save. Not service SLA.'}
(ROOT/'evaluation/v12/FROZEN-EXTRACTION.json').write_text(json.dumps(freeze,indent=2)+'\n')
result={'freeze':freeze,'rows':[]};folder=ROOT/'.benchmark-data/v12-final';folder.mkdir(exist_ok=True)
def scored(els,edges,r):
 pp,ee=targets(els,edges,r);ss=score(pp,ee,r['nodes'],{tuple(e) for e in r['edges']},.5);mapping=ss.pop('matched_ids');mapped={tuple(sorted((mapping.get(e['from_id'],'!'+e['from_id']),mapping.get(e['to_id'],'!'+e['to_id'])))) for e in ee};matched=set(mapping.values());causes=Counter('recovered' if tuple(sorted(e)) in mapped else 'missing_symbol_or_IoU' if not set(e)<=matched else 'routing' for e in r['edges']);return ss,dict(causes)
for key,r in json.loads(labels.read_text())['sheets'].items():
 path=ROOT/f'.benchmark-data/v12-held/{r["group"]}/{r["id"]}-4800.png';assert sha(path)==r['image_sha256'];view,hi=images(path);row={'sheet':key,'project':r['project'],'gt_glyphs':len(r['nodes']),'gt_edges':len(r['edges']),'variants':{}};times={k:[] for k in exs};first={}
 for repeat in range(3):
  for name in (list(exs) if repeat%2==0 else list(exs)[::-1]):
   ex=exs[name];ex.clear_ocr_cache();warnings=[];timing={};t=time.perf_counter();els,segs=ex.raster_candidates(view,warnings,high_resolution=hi,timings=timing);edges=infer(ex,els,segs,view);elapsed=time.perf_counter()-t;times[name].append(elapsed);identity=canonical(els,edges)
   if name in first:assert first[name]==identity,(key,name,'nondeterministic')
   else:
    first[name]=identity;ss,causes=scored(els,edges,r);row['variants'][name]={'scores':ss,'miss_causes':causes,'ocr_words':sum(e['kind']=='text' for e in els),'timing_components':timing,'warnings':warnings};(folder/f'{key}-{name}.json').write_text(json.dumps({'elements':els,'connections':edges})+'\n')
    if name=='v12':
     start=time.perf_counter();proposals=match(view,[t for t in templates if t['project']==r['project']],.48,20);extra=[p for p in proposals if not any(e['kind']!='text' and _iou(e['bbox'],p['bbox'])>.45 for e in els)];augmented=els+[element(p['name'],'unknown',p['bbox'],'legend_candidate',properties={'template_id':p['template_id']}) for p in extra];new_edges=infer(ex,augmented,segs,view);ls,lc=scored(augmented,new_edges,r);row['legend_assisted']={'scores':ls,'miss_causes':lc,'proposed':len(proposals),'added':len(extra),'proposal_and_routing_seconds':time.perf_counter()-start,'not_automatically_applied':True};(folder/f'{key}-legend.json').write_text(json.dumps({'elements':augmented,'connections':new_edges,'proposals':proposals})+'\n')
  print(key,repeat,times,flush=True)
 for name in exs:row['variants'][name].update(cold_seconds=times[name],cold_median=statistics.median(times[name]))
 result['rows'].append(row)
 result['aggregate']={name:{m:counts(**{k:sum(rr['variants'][name]['scores'][m][k] for rr in result['rows']) for k in ('tp','fp','fn')}) for m in ('localization','connections')} for name in exs}
 for name in exs:result['aggregate'][name]['mean_page_cold_median']=statistics.mean(rr['variants'][name]['cold_median'] for rr in result['rows'])
 result['aggregate']['legend_assisted']={m:counts(**{k:sum(rr['legend_assisted']['scores'][m][k] for rr in result['rows']) for k in ('tp','fp','fn')}) for m in ('localization','connections')}
 (ROOT/'evaluation/v12/HELD-OUT.json').write_text(json.dumps(result,indent=2)+'\n')
for p,digest in source_hashes.items():assert sha(ROOT/p)==digest,p
print(json.dumps(result['aggregate'],indent=2),flush=True)
