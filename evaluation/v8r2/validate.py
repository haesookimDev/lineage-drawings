"""Fixed validation only. End-to-end graph objective; manual point labels separate."""
import json,sys,time,copy,itertools
from pathlib import Path
from PIL import Image
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_extractor,load_gt,score,counts,match_boxes
from evaluation.v5.evaluate import targets
from evaluation.v6.data import sources
from evaluation.v7.evaluate import recognition_metrics
from backend import sparse_topology as sparse,topology_context as context
cv2.setNumThreads(2)
base=load_extractor('43041df641c860e064815845df54ea9be12e6270')[0];manual=json.loads((ROOT/'evaluation/v8r2/manual-validation-reviewed.json').read_text())['sheets'];oldlabels=json.loads((ROOT/'evaluation/v5/real-labels.json').read_text())['sheets'];cache=ROOT/'.benchmark-data/v8-validation';cache.mkdir(exist_ok=True);rows=[];baselines=[]
work=list(sources('validation'))+[('virden','p7',oldlabels['virden-p7']),('ruttan','p62',oldlabels['ruttan-p62'])]
for group,ident,region in work:
 path=ROOT/'.benchmark-data'/group/(ident+'.png')
 if group in ('pahala','virden','ruttan'):path=ROOT/'.benchmark-data/v6-highres'/group/(ident+'.png')
 with Image.open(path) as raw:size=raw.size;im=raw.convert('RGB');high=im.copy();high.thumbnail((4800,4800));im.thumbnail((2400,2400))
 gt,truth=(region['nodes'],set(tuple(e) for e in region['edges'])) if region else load_gt(path.with_suffix('.graphml'),size)[:2];cached=cache/f'{group}-{ident}.json'
 if cached.exists():data=json.loads(cached.read_text())
 else:
  els,segs=base.raster_candidates(im,[],high_resolution=high);details={};edges=base.infer_connections(els,segs,*im.size,image=im,diagnostics=details);data={'elements':els,'segments':segs,'edges':edges,'details':details};cached.write_text(json.dumps(data)+'\n')
 els,segs,old=data['elements'],data['segments'],data['edges'];oldpairs={tuple(sorted((e['from_id'],e['to_id']))) for e in old};pp,ee=targets(els,old,region);baseline=score(pp,ee,gt,truth,.5);baseline.pop('matched_ids');br={'group':group,'id':ident,'scores':baseline}
 key=group+'-'+ident
 if key in manual:
  mr=manual[key];mapping=match_boxes([e for e in els if e['kind']!='text'],mr['nodes'],.5);br['recognition']=recognition_metrics([e for e in els if e['id'] in mapping],data['details'],mr,im.size,mapping)
 baselines.append(br);gray,ink,raw,junctions,lines=sparse.route(els,segs,*im.size,im,return_proposals=True);pm,jm,meta=context.model();_,intervals=sparse.proposals(els,lines,*im.size);pred={'gray':gray,'ink':ink,'lines':lines,'dashed':0,'raw':raw,'intervals':intervals,'junctions':junctions,'port_scores':context.votes(pm,context.features(gray,ink,raw),2)[:,1],'junction_scores':context.votes(jm,context.features(gray,ink,junctions),4)};print('mapped',key,'ports',len(raw),'junctions',len(junctions),flush=True)
 for port,junction,extra in itertools.product((.35,.5,.65),(.1,.25),(.55,.75,.9)):
  config={'port_threshold':port,'junction_threshold':junction,'junction_display_threshold':max(.25,junction),'extra_edge_threshold':extra};pairs,details,ports=sparse.route(els,segs,*im.size,im,config=config,recognized=pred);fresh={tuple(sorted((p['from_id'],p['to_id']))) for p in details['paths'] if p['score']>=extra};order={e['id']:i for i,e in enumerate(els)}
  for mode,display in itertools.product(('hybrid','new'),(.25,.35,.45)):
   selected=(oldpairs|fresh) if mode=='hybrid' else fresh;selected=sorted(selected,key=lambda p:(p not in oldpairs if mode=='hybrid' else False,tuple(sorted((order[p[0]],order[p[1]])))))[:500];edges=[{'from_id':a,'to_id':b} for a,b in selected];pp,ee=targets(els,edges,region);sc=score(pp,ee,gt,truth,.5);sc.pop('matched_ids');row={'group':group,'id':ident,'configuration':f'{port}/{junction}/{extra}/{mode}/{display}','scores':sc,'ports':details['port_count'],'junctions':len(details['junctions'])}
   if key in manual:
    mr=manual[key];mapping=match_boxes([e for e in els if e['kind']!='text'],mr['nodes'],.5);pe=copy.deepcopy(els)
    for e in pe:e.setdefault('properties',{})['ports']=[{a:b for a,b in p.items() if a!='pixel'} for p in ports.get(e['id'],[])]
    # Score only annotated symbols' ports for development. End-to-end new-test
    # ROIs enumerate all symbols and count every ROI candidate, including FPs.
    matched=set(mapping);pe=[e for e in pe if e['id'] in matched];display_details={**details,'junctions':[p for p in details['junctions'] if p['score']>=display]};row['recognition']=recognition_metrics(pe,display_details,mr,im.size,mapping)
   rows.append(row)
 aggregate={};groups=sorted({r['group'] for r in baselines})
 for config in sorted({r['configuration'] for r in rows}):
  metrics={g:counts(**{k:sum(r['scores']['connections'][k] for r in rows if r['configuration']==config and r['group']==g) for k in ('tp','fp','fn')}) for g in groups};rp=[r['recognition'] for r in rows if r['configuration']==config and 'recognition' in r];points={m:counts(**{k:sum(r[m][k] for r in rp) for k in ('tp','fp','fn')}) for m in ('ports','junctions')};aggregate[config]={'groups':metrics,'objective':float(np.mean([m['f1'] for m in metrics.values()])),'recognition':points}
 baselinegroups={g:counts(**{k:sum(r['scores']['connections'][k] for r in baselines if r['group']==g) for k in ('tp','fp','fn')}) for g in groups}
 (ROOT/'evaluation/v8r2/VALIDATION.json').write_text(json.dumps({'rows':rows,'baseline_rows':baselines,'aggregate':aggregate,'baseline':{'groups':baselinegroups,'objective':float(np.mean([m['f1'] for m in baselinegroups.values()]))},'policy':'Select by macro connection F1 over declared validation source groups. No new-page test inference. Ports development metric only annotated matched symbols; end-to-end final test includes unmatched ROI FPs. Original v5 edge labels preserved for val graph scoring.'},indent=2)+'\n')
print('BASELINE',json.dumps(baselinegroups),flush=True)
print('RANKING',sorted([(d['objective'],k,d['recognition']) for k,d in aggregate.items()],reverse=True)[:8],flush=True)
