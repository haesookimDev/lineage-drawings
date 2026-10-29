"""Exact graph-stage replay with fixed R2 symbols; no post-audit blind-graph claim.
Recomputes sparse paths, verifies ungated R2 replay equals original full inference,
then uses the same production combination helper with the validation-only gate.
Original point detections/recognition are unchanged and retained from full runs.
"""
import json,sys,hashlib,copy
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend.sparse_topology import route
from backend.topology_context import model
from backend.geometry import combine_context_pairs
from backend.extract import ocr
from evaluation.benchmark import score,counts,load_gt
from evaluation.v5.evaluate import targets
from evaluation.v7.evaluate import recognition_metrics
from collections import Counter
original=json.loads((ROOT/'evaluation/v8r2/results/report.json').read_text());assert len(original['sheets'])==28
region_sets={g:json.loads((ROOT/p).read_text())['sheets'] for g,p in [('new-page','evaluation/v8r2/new-page-labels.json'),('r1-test-now-training','evaluation/v8/new-page-labels.json'),('former-test-now-training','evaluation/v7/new-page-labels.json'),('historical-real-roi','evaluation/v6/new-page-labels.json')]};old_real=json.loads((ROOT/'evaluation/v5/real-labels.json').read_text())['sheets'];out=ROOT/'evaluation/v8r2/guard-results';out.mkdir(exist_ok=True);result={**original,'sheets':[],'aggregate':{},'release_guard':json.loads((ROOT/'evaluation/v8r2/STRICT-RELEASE-SELECTION.json').read_text())['selected_configuration'],'graph_evaluation_role':'Audited regression after R2 historical precision issue; gate chosen only on fixed validation. Original point-model new-test results remain independent because neither point models nor outputs change. Graph replay reuses fixed detections, omits OCR/symbol stage, and is not timed as full inference.'};meta=model()[2]
for row in original['sheets']:
 group,ident=row['group'],row['id'];region=region_sets.get(group,{}).get(ident)
 if group=='pahala-p21-regression':region=old_real['pahala-p21']
 if region:
  g,i=region['group'],region['id'];folder='v8r2-pages' if group=='new-page' else 'v8-pages' if group=='r1-test-now-training' else 'v7-highres' if group=='former-test-now-training' else 'v6-highres';filename=i+'-4800.png' if folder in ('v8r2-pages','v8-pages') else i+'.png';path=ROOT/'.benchmark-data'/folder/g/filename
 else:path=ROOT/'.benchmark-data'/('open100' if group=='open100-regression' else 'pid-confirmation')/(ident+'.png')
 with Image.open(path) as im0:size=im0.size;im=im0.convert('RGB');im.thumbnail((2400,2400))
 gt,truth=(region['nodes'],set(tuple(e) for e in region['edges'])) if region else load_gt(path.with_suffix('.graphml'),size)[:2]
 b=json.loads((ROOT/'evaluation/v8r2/results'/f'{group}-{ident}-v7-predictions.json').read_text());c=json.loads((ROOT/'evaluation/v8r2/results'/f'{group}-{ident}-v8-predictions.json').read_text());bs=[e for e in b['elements'] if e['kind']!='text'];cs=[e for e in c['elements'] if e['kind']!='text'];assert [e['bbox'] for e in bs]==[e['bbox'] for e in cs];alias={a['id']:z['id'] for a,z in zip(bs,cs)};basepairs=[tuple(sorted((alias[e['from_id']],alias[e['to_id']]))) for e in b['connections']];els=c['elements']
 if region:
  cache=ROOT/'.benchmark-data/v8r2-release-ocr'/f'{group}-{ident}.json';cache.parent.mkdir(exist_ok=True)
  if not cache.exists():cache.write_text(json.dumps(ocr(im,[])[:500-len(cs)]))
  els=json.loads(cache.read_text())+cs
 _,raw,_=route(els,[],*im.size,im,config={'strict_contacts':False,'junction_threshold':.25});ungated,_=combine_context_pairs(c['elements'],basepairs,raw['paths'],{**meta,'extra_symbol_threshold':0.,'extra_max_points':128});assert set(ungated)=={tuple(sorted((e['from_id'],e['to_id']))) for e in c['connections']},(ident,'graph replay differs from full R2 inference')
 _,raw,_=route(els,[],*im.size,im);pairs,composition=combine_context_pairs(c['elements'],basepairs,raw['paths'],meta);allowed=set(pairs);known={tuple(sorted((e['from_id'],e['to_id']))):e for e in c['connections']};edges=[known.get(k,{'id':hashlib.sha256(str(k).encode()).hexdigest()[:32],'from_id':k[0],'to_id':k[1],'source':'recognized_path_candidate','kind':'connectedTo','status':'needs_review'}) for k in pairs];boxes={e['id']:e['bbox'] for e in cs};paths=[{**p,'from_bbox':boxes[p['from_id']],'to_bbox':boxes[p['to_id']]} for p in raw['paths'] if tuple(sorted((p['from_id'],p['to_id']))) in allowed];rec={**c['recognition'],**composition,'routing_contacts':raw.get('routing_contacts',[]),'paths':paths};(out/f'{group}-{ident}-release-predictions.json').write_text(json.dumps({'elements':c['elements'],'connections':edges,'recognition':rec},ensure_ascii=False,indent=2)+'\n')
 rr=copy.deepcopy(row);rr['variants']={k:v for k,v in rr['variants'].items() if k in ('v7','v8')};d=rr['variants']['v8'];pp,ee=targets(c['elements'],edges,region);sc=score(pp,ee,gt,truth,.5);mapping=sc.pop('matched_ids');matched=set(mapping.values());mapped={tuple(sorted((mapping.get(e['from_id'],'unmatched:'+e['from_id']),mapping.get(e['to_id'],'unmatched:'+e['to_id'])))) for e in ee};causes=Counter('recovered' if e in mapped else 'missing_symbol' if not set(e)<=matched else 'routing' for e in truth);d['scores']=sc;
 if region and 'recognition' in d:d['association_junctions']=recognition_metrics(c['elements'],{**rec,'junctions':rec.get('routing_contacts',[])},region,im.size,mapping)['junctions']
 d['scores']=sc;d['full_page_connections']=len(edges);d['supplemental_connections']=composition['context_supplemental_connections'];d['conditional_endpoints']['recovered']=sc['connections']['tp'];d['conditional_endpoints']['recall']=sc['connections']['tp']/max(1,d['conditional_endpoints']['eligible']);d['miss_attribution']=dict(causes);d['timing']['scope']='Original R2 before endpoint guard; final paired timing measured separately';rr['ungated_replay_exact']=True;rr['added_true_roi_edges']=sc['connections']['tp']-rr['variants']['v7']['scores']['connections']['tp'];rr['added_false_roi_edges']=sc['connections']['fp']-rr['variants']['v7']['scores']['connections']['fp'];result['sheets'].append(rr);print(group,ident,sc['connections'],flush=True)
for g in sorted({r['group'] for r in result['sheets']}):
 ss=[r for r in result['sheets'] if r['group']==g];result['aggregate'][g]={}
 for v in ('v7','v8'):
  agg={m:counts(**{k:sum(r['variants'][v]['scores'][m][k] for r in ss) for k in ('tp','fp','fn')}) for m in ('localization','connections')};agg['conditional_endpoints']={k:sum(r['variants'][v]['conditional_endpoints'][k] for r in ss) for k in ('eligible','recovered')};agg['conditional_endpoints']['recall']=agg['conditional_endpoints']['recovered']/max(1,agg['conditional_endpoints']['eligible']);agg['pre_guard_mean_sheet_median_seconds']=sum(r['variants'][v]['timing']['median'] for r in ss)/len(ss)
  if 'recognition' in ss[0]['variants'][v]:agg['recognition']={m:counts(**{k:sum(r['variants'][v]['recognition'][m][k] for r in ss) for k in ('tp','fp','fn')}) for m in ('ports','junctions')};agg['association_junctions']=counts(**{k:sum(r['variants'][v]['association_junctions'][k] for r in ss) for k in ('tp','fp','fn')})
  result['aggregate'][g][v]=agg
(out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('AGGREGATE',json.dumps(result['aggregate']),flush=True)
