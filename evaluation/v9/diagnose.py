"""Post-test diagnosis only; never rewrite labels or tune the release."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import score,iou
from evaluation.v5.evaluate import targets
labels=json.loads((ROOT/'evaluation/v9/new-project-labels.json').read_text());out={}
for key,s in labels['sheets'].items():
 d=json.loads((ROOT/'evaluation/v9/results'/f'{key}-v9-predictions.json').read_text());pp,ee=targets(d['elements'],d['connections'],s);truth=set(tuple(e) for e in s['edges']);sc=score(pp,ee,s['nodes'],truth,.5);m=sc.pop('matched_ids');matched=set(m.values());mapped={tuple(sorted((m[e['from_id']],m[e['to_id']]))) for e in ee if e['from_id'] in m and e['to_id'] in m};miss=[]
 for g in s['nodes']:
  if g['id'] not in matched:miss.append({'gt_id':g['id'],'kind':g['kind'],'best_box_iou':max((iou(g['bbox'],p['bbox']) for p in pp),default=0.)})
 out[key]={'missing_symbols':miss,'missing_connections':[{'pair':e,'cause':'missing_symbol' if not set(e)<=matched else 'line_or_port_routing','subsets':[k for k,v in s['edge_subsets'].items() if e in set(tuple(x) for x in v)]} for e in sorted(truth-mapped)],'false_connections':sc['connections']['fp']};print(key,miss,'missing edges',len(out[key]['missing_connections']))
(ROOT/'evaluation/v9/ERRORS.json').write_text(json.dumps({'role':'Post-test diagnostic, no edits to labels,weights,thresholds or routing; not an extra validation step','sheets':out},indent=2)+'\n')
