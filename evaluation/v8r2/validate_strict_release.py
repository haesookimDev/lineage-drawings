"""Validate conservative new-edge contacts; never alter point models or labels."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_gt,score,counts
from evaluation.v5.evaluate import targets
from evaluation.v6.data import sources
from backend.sparse_topology import route
from backend.topology_context import model
from backend.geometry import combine_context_pairs
real=json.load(open(ROOT/'evaluation/v5/real-labels.json'))['sheets'];work=list(sources('validation'))+[('virden','p7',real['virden-p7']),('ruttan','p62',real['ruttan-p62'])];rows=[];meta=model()[2]
for group,ident,region in work:
 p=ROOT/'.benchmark-data'/group/(ident+'.png')
 if group in ('pahala','virden','ruttan'):p=ROOT/'.benchmark-data/v6-highres'/group/(ident+'.png')
 with Image.open(p) as raw:size=raw.size;im=raw.convert('RGB');im.thumbnail((2400,2400))
 gt,truth=(region['nodes'],set(tuple(e) for e in region['edges'])) if region else load_gt(p.with_suffix('.graphml'),size)[:2];d=json.load(open(ROOT/'.benchmark-data/v8-validation'/f'{group}-{ident}.json'));els=d['elements'];base=[tuple(sorted((e['from_id'],e['to_id']))) for e in d['edges']]
 for junction in (.45,):
  _,r,_=route(els,d['segments'],*im.size,im,config={'strict_contacts':True,'junction_threshold':junction})
  for limit in (3,4):
   symbol=.8
   pairs,comp=combine_context_pairs(els,base,r['paths'],{**meta,'extra_symbol_threshold':symbol,'extra_max_points':limit});pp,ee=targets(els,[{'from_id':a,'to_id':b} for a,b in pairs],region);s=score(pp,ee,gt,truth,.5);s.pop('matched_ids');rows.append({'group':group,'id':ident,'configuration':f'{junction}/{symbol}/{limit}','scores':s})
 print(group,ident,flush=True)
base=json.load(open(ROOT/'evaluation/v8r2/VALIDATION.json'))['baseline'];agg={}
for key in sorted({r['configuration'] for r in rows}):
 groups={g:counts(**{k:sum(r['scores']['connections'][k] for r in rows if r['configuration']==key and r['group']==g) for k in ('tp','fp','fn')}) for g in sorted({r['group'] for r in rows})};safe=all(groups[g]['precision']>=base['groups'][g]['precision']-1e-12 for g in ('open100','pahala','virden'));agg[key]={'groups':groups,'objective':float(np.mean([d['f1'] for d in groups.values()])),'real_validation_precision_preserved':safe}
eligible=[k for k,v in agg.items() if v['real_validation_precision_preserved'] and v['objective']>base['objective']];assert eligible,'No new-edge variant improves validated baseline under precision constraint';chosen=max(eligible,key=lambda k:(agg[k]['objective'],*[float(x) for x in k.split('/')]))
out={'policy':'New edges must use the same actual arms, glyph/OCR clearance and confidence .45 as displayed contacts; plain X stays separate. Supplemental paths are limited to one/two bends (3/4 points); the bound prevents error amplification along long inferred chains. These constraints were developed after auditing historical regression, including exploring path limits2/3/4/5; final graph results are development regressions. Select only variants whose precision does not fall in any predeclared real validation source group, then maximize source-macro F1. Point models unchanged. Final graph evaluations are post-audit regressions, not untouched blind graph tests.','rows':rows,'aggregate':agg,'baseline':base,'selected_configuration':chosen,'selected':agg[chosen]};(ROOT/'evaluation/v8r2/STRICT-RELEASE-SELECTION.json').write_text(json.dumps(out,indent=2)+'\n');print(chosen,agg[chosen],flush=True)
