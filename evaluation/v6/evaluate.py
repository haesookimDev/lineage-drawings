#!/usr/bin/env python3
"""Frozen neural high-resolution vs immutable deployed v5, paired CPU runs."""
import argparse,hashlib,json,statistics,sys,time
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_extractor,load_gt,score,counts,infer
from evaluation.v5.evaluate import canonical,targets

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',type=int,default=3);ap.add_argument('--group',choices=['all','new','regression'],default='all');args=ap.parse_args();folder=ROOT/'evaluation/v6/results';folder.mkdir(exist_ok=True)
 freeze=json.loads((ROOT/'evaluation/v6/FROZEN.json').read_text());assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in freeze['sources'].items());assert hashlib.sha256((ROOT/'evaluation/v6/new-page-labels.json').read_bytes()).hexdigest()==freeze['labels_sha256']
 base=load_extractor(freeze['baseline'])[0];current=load_extractor('working')[0];labels=json.loads((ROOT/'evaluation/v6/new-page-labels.json').read_text())['sheets'];work=[]
 if args.group in ('all','new'):
  for key,s in labels.items():work.append(('new-page',key,s,ROOT/'.benchmark-data/v6-highres'/s['group']/(s['id']+'.png')))
 if args.group in ('all','regression'):
  for i in range(6,12):work.append(('open100-regression',str(i),None,ROOT/'.benchmark-data/open100'/f'{i}.png'))
  s=json.loads((ROOT/'evaluation/v5/real-labels.json').read_text())['sheets']['pahala-p21'];work.append(('pahala-p21-regression','p21',s,ROOT/'.benchmark-data/v6-highres/pahala/p21.png'))
  for i in [9,34,385,285,317,281,82,382]:work.append(('synthetic-regression',str(i),None,ROOT/'.benchmark-data/pid-confirmation'/f'{i}.png'))
 result={'baseline':freeze['baseline'],'evaluated_at':datetime.now(timezone.utc).isoformat(),'runs':args.runs,'frozen_sha256':hashlib.sha256((ROOT/'evaluation/v6/FROZEN.json').read_bytes()).hexdigest(),'timing_scope':'RGB decode/thumbnail and PDF render excluded. Timed OCR, symbol/line/edge inference. Same 2400px preview, v6 receives at most4800px original. Per-sheet three sequential paired repeats with reversed order. v6-preview ablation receives only2400px. Not throughput or SLA.','sheets':[]}
 for group,ident,region,path in work:
  with Image.open(path) as raw:
   size=raw.size;image=raw.convert('RGB');high=image.copy();high.thumbnail((4800,4800));image.thumbnail((2400,2400))
  if region:gt,truth=region['nodes'],set(tuple(e) for e in region['edges'])
  else:gt,truth,_=load_gt(path.with_suffix('.graphml'),size)
  variants={'v5':base,'v6':current}
  if group=='new-page':variants={'v5':base,'v6-preview':current,'v6':current}
  row={'group':group,'id':ident,'role':'new-page-known-project' if group=='new-page' else 'historical-regression','roi_only':region is not None,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_size':size,'rendered_size':image.size,'gt_symbols':len(gt),'gt_connections':len(truth),'variants':{}};first={};timings={v:[] for v in variants}
  for repeat in range(args.runs):
   for name in (list(variants) if repeat%2==0 else list(reversed(variants))):
    extractor=variants[name];warning=[];start=time.perf_counter();els,segments=extractor.raster_candidates(image,warning,high_resolution=high if name=='v6' else image) if name!='v5' else extractor.raster_candidates(image,warning);edges=infer(extractor,els,segments,image);seconds=time.perf_counter()-start;timings[name].append(round(seconds,4));assert any(e['kind']=='text' for e in els),(group,ident,name,'empty OCR');rep=canonical(els,edges)
    if name in first:assert first[name]==rep,(group,ident,name,'nondeterministic')
    else:
     first[name]=rep;pred,ee=targets(els,edges,region);sc=score(pred,ee,gt,truth,.5);mapping=sc.pop('matched_ids');matched=set(mapping.values());mapped={tuple(sorted((mapping.get(e['from_id'],'unmatched:'+e['from_id']),mapping.get(e['to_id'],'unmatched:'+e['to_id'])))) for e in ee};causes=Counter('recovered' if e in mapped else 'missing_symbol' if not set(e)<=matched else 'line_or_port_routing' for e in truth)
     row['variants'][name]={'scores':sc,'pred_symbols':len(pred),'pred_connections':len(ee),'full_page_symbols':len(rep['boxes']),'full_page_connections':len(edges),'miss_attribution':dict(causes),'warnings':warning,'deterministic':True}
     export=els if group!='new-page' and not group.startswith('pahala') else [{k:v for k,v in e.items() if k!='label'} for e in els if e['kind']!='text']
     (folder/f'{group}-{ident}-{name}-predictions.json').write_text(json.dumps({'elements':export,'connections':edges},ensure_ascii=False,indent=2)+'\n')
  for name in variants:row['variants'][name]['timing']={'seconds':timings[name],'median':statistics.median(timings[name]),'min':min(timings[name]),'max':max(timings[name])}
  result['sheets'].append(row);result['aggregate']={}
  for g in sorted({s['group'] for s in result['sheets']}):
   ss=[s for s in result['sheets'] if s['group']==g];result['aggregate'][g]={}
   for v in ss[0]['variants']:
    result['aggregate'][g][v]={m:counts(**{k:sum(s['variants'][v]['scores'][m][k] for s in ss) for k in ('tp','fp','fn')}) for m in ('localization','connections')};result['aggregate'][g][v]['mean_sheet_median_seconds']=statistics.mean(s['variants'][v]['timing']['median'] for s in ss);result['aggregate'][g][v]['miss_attribution']=dict(sum((Counter(s['variants'][v]['miss_attribution']) for s in ss),Counter()))
  (folder/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(group,ident,{v:{'symbol_R':round(d['scores']['localization']['recall'],3),'edge_R':round(d['scores']['connections']['recall'],3),'edge_F1':round(d['scores']['connections']['f1'],3),'seconds':d['timing']['median']} for v,d in row['variants'].items()},flush=True)
 print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
