"""Combine already conforming real/validation rows with corrected large synthetic inputs."""
import json,statistics,shutil
from collections import Counter
from pathlib import Path
H=Path(__file__).parent
old=json.loads((H/'regression/report.json').read_text());new=json.loads((H/'regression-synthetic-production/report.json').read_text());assert len(old['sheets'])==17 and len(new['sheets'])==8
old['sheets']=[r for r in old['sheets'] if r['group']!='synthetic']+new['sheets'];old['role']='Historical regression / validation diagnostic with production-matching independent original-to-viewer and original-to-symbol resizing. Real9 rows retained unchanged; synthetic8 re-executed, see REGRESSION-PROTOCOL.json. Single repeats are not a robust speed benchmark.'
for row in old['sheets']:
 for d in row['variants'].values():d['deterministic']=None
old['aggregate']={}
from evaluation.benchmark import counts
for field in ('all','group','project'):
 values=['all'] if field=='all' else sorted({r[field] for r in old['sheets']});old['aggregate'][field]={}
 for key in values:
  rr=[r for r in old['sheets'] if field=='all' or r[field]==key];old['aggregate'][field][key]={}
  for v in ('v8','v9'):
   sums={m:counts(**{k:sum(r['variants'][v]['scores'][m][k] for r in rr) for k in ('tp','fp','fn')}) for m in ('localization','connections')};sums['mean_sheet_median_seconds']=statistics.mean(r['variants'][v]['timing']['median'] for r in rr);sums['miss_attribution']=dict(sum((Counter(r['variants'][v]['miss_attribution']) for r in rr),Counter()));sums['edge_subsets']={k:{'recovered':0,'total':0} for k in ('dashed','branch')};old['aggregate'][field][key][v]=sums
for p in (H/'regression-synthetic-production').glob('*predictions.json'):shutil.copyfile(p,H/'regression'/p.name)
(H/'regression/report.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n');print('Merged17 production-normalized historical/validation pages')
