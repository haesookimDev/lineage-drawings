"""Frozen unseen-project evaluation with full-pipeline symbol-only ablation."""
import argparse,json,hashlib,sys,time,statistics,copy
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_extractor,score,counts,load_gt
from evaluation.v5.evaluate import targets,canonical

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',type=int,default=3);ap.add_argument('--scope',choices=['new','regression'],default='new');args=ap.parse_args();freeze=json.loads((ROOT/'evaluation/v9/FROZEN-RELEASE.json').read_text())
 for n,h in freeze['sources'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
 base=load_extractor(freeze['baseline'])[0];current=load_extractor('working')[0];variants={'v8':base,'glyph-only':SimpleNamespace(raster_candidates=current.raster_candidates,infer_connections=base.infer_connections,MAX_ELEMENTS=current.MAX_ELEMENTS),'v9':current};folder=ROOT/'evaluation/v9'/('results' if args.scope=='new' else 'regression-synthetic-production');folder.mkdir(exist_ok=True);labels=json.loads((ROOT/'evaluation/v9/new-project-labels.json').read_text());audit=json.loads((ROOT/'evaluation/v9/SPLIT-AUDIT.json').read_text());work=[]
 if args.scope=='new':
  for k,s in labels['sheets'].items():work.append((s['group'],s['id'],s,ROOT/'.benchmark-data/v9-held'/s['group']/(s['id']+'-4800.png'),audit['test_sheet_projects'][k]['facility_project']))
 else:
  variants={k:v for k,v in variants.items() if k!='glyph-only'}
  for i in range(6,12):work.append(('open100',str(i),None,ROOT/'.benchmark-data/open100'/f'{i}.png','OPEN100 historical regression'))
  for k,s in json.loads((ROOT/'evaluation/v8r2/new-page-labels.json').read_text())['sheets'].items():work.append((s['group'],s['id'],s,ROOT/'.benchmark-data/v8r2-pages'/s['group']/(s['id']+'-4800.png'),'Validation diagnostic, not unseen'))
  for i in [9,34,385,285,317,281,82,382]:work.append(('synthetic',str(i),None,ROOT/'.benchmark-data/pid-confirmation'/f'{i}.png','Synthetic historical regression'))
 assert args.scope=='regression';work=[row for row in work if row[0]=='synthetic']
 result={'baseline':freeze['baseline'],'frozen_sha256':hashlib.sha256((ROOT/'evaluation/v9/FROZEN-RELEASE.json').read_bytes()).hexdigest(),'role':'Corrected validation-conforming new-project small-ROI evaluation. No project in training/selection; prior nonconforming merge run disclosed in VALIDATION-CONTRACT-FIX.json. Weights/thresholds/GT unchanged.' if args.scope=='new' else 'Historical regression and validation diagnostic, not new-project evidence','runs':args.runs,'timing_scope':'Full in-memory OCR + same max4800 symbol input +2400 line/port/junction inference. Sequential rotated3-variant order,3 repeats; mean of per-sheet medians. Decode/render/network/save excluded; CPU OpenCV2/OMP1, not Pod SLA.','sheets':[]}
 for group,ident,region,path,project in work:
  with Image.open(path) as raw:original=raw.size;high=raw.convert('RGB');im=high.copy();high.thumbnail((4800,4800));im.thumbnail((2400,2400))
  gt,truth=(region['nodes'],set(tuple(e) for e in region['edges'])) if region else load_gt(path.with_suffix('.graphml'),original)[:2];row={'group':group,'id':ident,'project':project,'roi_only':region is not None,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'gt_symbols':len(gt),'gt_edges':len(truth),'variants':{}};times={k:[] for k in variants};first={};point_evidence={}
  for run in range(args.runs):
   order=list(variants);order=order[run%len(order):]+order[:run%len(order)]
   for name in order:
    ex=variants[name];rec={};t=time.perf_counter();els,segs=ex.raster_candidates(im,[],high_resolution=high);els=els[:ex.MAX_ELEMENTS];edges=ex.infer_connections(els,segs,*im.size,image=im,diagnostics=rec);times[name].append(time.perf_counter()-t);assert any(e['kind']=='text' for e in els),(ident,name,'no OCR');rep=canonical(els,edges);sym=[e for e in els if e['kind']!='text'];index={e['id']:j for j,e in enumerate(sym)};ev={'ports':[e.get('properties',{}).get('ports',[]) for e in sym],'junctions':rec.get('junctions',[]),'paths':sorted((tuple(sorted((index[p['from_id']],index[p['to_id']]))),p['points'] if index[p['from_id']]<index[p['to_id']] else list(reversed(p['points']))) for p in rec.get('paths',[]))};assert len(sym)<=200 and len(edges)<=500
    if name in first:assert rep==first[name] and ev==point_evidence[name],(group,ident,name,'nondeterministic')
    else:
     first[name]=rep;point_evidence[name]=ev;pp,ee=targets(els,edges,region);sc=score(pp,ee,gt,truth,.5);mapping=sc.pop('matched_ids');matched=set(mapping.values());mapped={tuple(sorted((mapping.get(e['from_id'],'unmatched:'+e['from_id']),mapping.get(e['to_id'],'unmatched:'+e['to_id'])))) for e in ee};causes=Counter('recovered' if k in mapped else 'missing_symbol' if not set(k)<=matched else 'routing' for k in truth);subsets={k:{'recovered':len(mapped&set(tuple(e) for e in vv)),'total':len(vv),'recall':len(mapped&set(tuple(e) for e in vv))/max(1,len(vv))} for k,vv in region.get('edge_subsets',{}).items()} if region else {};row['variants'][name]={'scores':sc,'full_page_symbols':len(sym),'full_page_connections':len(edges),'conditional_endpoints':{'eligible':sum(set(e)<=matched for e in truth),'recovered':sc['connections']['tp']},'miss_attribution':dict(causes),'edge_subsets':subsets,'deterministic':True if args.runs>1 else None,'neural_tiles':rec.get('neural_tile_count'),'detail_tiles':next((e['properties'].get('detail_tile_count',0) for e in sym if e['source']=='neural_symbol_candidate'),0),'ports':rec.get('port_count',0),'junctions':len(rec.get('junctions',[]))}
     export=[{k:v for k,v in e.items() if k!='label'} for e in els if e['kind']!='text'] if region else els;(folder/f'{group}-{ident}-{name}-predictions.json').write_text(json.dumps({'elements':export,'connections':edges,'recognition':rec},ensure_ascii=False,separators=(',',':'))+'\n')
  if 'glyph-only' in first:assert first['glyph-only']['boxes']==first['v9']['boxes']
  for name in variants:row['variants'][name]['timing']={'seconds':times[name],'median':statistics.median(times[name])}
  result['sheets'].append(row);result['aggregate']={}
  for field in ('all','group','project'):
   values=['all'] if field=='all' else sorted({r[field] for r in result['sheets']});result['aggregate'][field]={}
   for key in values:
    rr=[r for r in result['sheets'] if field=='all' or r[field]==key];result['aggregate'][field][key]={}
    for v in variants:
     sums={m:counts(**{k:sum(r['variants'][v]['scores'][m][k] for r in rr) for k in ('tp','fp','fn')}) for m in ('localization','connections')};sums['mean_sheet_median_seconds']=statistics.mean(r['variants'][v]['timing']['median'] for r in rr);sums['miss_attribution']=dict(sum((Counter(r['variants'][v]['miss_attribution']) for r in rr),Counter()));sums['edge_subsets']={k:{'recovered':sum(r['variants'][v]['edge_subsets'].get(k,{}).get('recovered',0) for r in rr),'total':sum(r['variants'][v]['edge_subsets'].get(k,{}).get('total',0) for r in rr)} for k in ('dashed','branch')};result['aggregate'][field][key][v]=sums
  (folder/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(group,ident,{k:{'symbols':d['scores']['localization'],'edges':d['scores']['connections'],'seconds':d['timing']['median'],'subsets':d['edge_subsets']} for k,d in row['variants'].items()},flush=True)
 print('FINAL',json.dumps(result['aggregate']['all'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
