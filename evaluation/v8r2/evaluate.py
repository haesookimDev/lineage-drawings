"""Frozen v7/v8 paired test and role-separated historical regression."""
import argparse,hashlib,json,statistics,sys,time,copy
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_extractor,load_gt,score,counts
from evaluation.v5.evaluate import canonical,targets
from evaluation.v7.evaluate import recognition_metrics


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',type=int,default=3);args=ap.parse_args();freeze=json.loads((ROOT/'evaluation/v8r2/FROZEN.json').read_text())
 for p,h in freeze['sources'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 folder=ROOT/'evaluation/v8r2/results';folder.mkdir(exist_ok=True);work=[]
 for key,s in json.loads((ROOT/'evaluation/v8r2/new-page-labels.json').read_text())['sheets'].items():work.append(('new-page',key,s,ROOT/'.benchmark-data/v8r2-pages'/s['group']/(s['id']+'-4800.png')))
 for key,s in json.loads((ROOT/'evaluation/v8/new-page-labels.json').read_text())['sheets'].items():work.append(('r1-test-now-training',key,s,ROOT/'.benchmark-data/v8-pages'/s['group']/(s['id']+'-4800.png')))
 for key,s in json.loads((ROOT/'evaluation/v7/new-page-labels.json').read_text())['sheets'].items():work.append(('former-test-now-training',key,s,ROOT/'.benchmark-data/v7-highres'/s['group']/(s['id']+'.png')))
 for key,s in json.loads((ROOT/'evaluation/v6/new-page-labels.json').read_text())['sheets'].items():work.append(('historical-real-roi',key,s,ROOT/'.benchmark-data/v6-highres'/s['group']/(s['id']+'.png')))
 for i in range(6,12):work.append(('open100-regression',str(i),None,ROOT/'.benchmark-data/open100'/f'{i}.png'))
 s=json.loads((ROOT/'evaluation/v5/real-labels.json').read_text())['sheets']['pahala-p21'];work.append(('pahala-p21-regression','p21',s,ROOT/'.benchmark-data/v6-highres/pahala/p21.png'))
 for i in [9,34,385,285,317,281,82,382]:work.append(('synthetic-regression',str(i),None,ROOT/'.benchmark-data/pid-confirmation'/f'{i}.png'))
 all_variants={v:load_extractor(ref)[0] for v,ref in [('v7',freeze['baseline']),('v8','working'),('v8-r1','6836d0ab900fef970de90bee2be04345f7be084c')]};result={'baseline':freeze['baseline'],'evaluated_at':datetime.now(timezone.utc).isoformat(),'frozen_sha256':hashlib.sha256((ROOT/'evaluation/v8r2/FROZEN.json').read_bytes()).hexdigest(),'runs':args.runs,'timing_scope':'In-memory OCR + same max4800 symbol detector + max2400 adjacency/recognition. RGB decode/resize, PDF render, network and save excluded. Sequential paired repeats, reversed order each repeat. CPU OpenCV2threads/OMP1. Mean of per-sheet3-run medians; not Pod SLA.','sheets':[]}
 for group,ident,region,path in work:
  variants={k:v for k,v in all_variants.items() if group=='new-page' or k!='v8-r1'}
  with Image.open(path) as raw:size=raw.size;im=raw.convert('RGB');high=im.copy();high.thumbnail((4800,4800));im.thumbnail((2400,2400))
  gt,truth=(region['nodes'],set(tuple(e) for e in region['edges'])) if region else load_gt(path.with_suffix('.graphml'),size)[:2]
  row={'group':group,'id':ident,'role':'new-page-known-project' if group=='new-page' else 'training-diagnostic' if group in ('former-test-now-training','r1-test-now-training') else 'historical-regression','roi_only':region is not None,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rendered_size':im.size,'gt_symbols':len(gt),'gt_edges':len(truth),'variants':{}};first={};evidence={};timings={v:[] for v in variants}
  for repeat in range(args.runs):
   for name in (list(variants) if repeat%2==0 else list(reversed(variants))):
    ex=variants[name];details={};warn=[];t=time.perf_counter();els,segs=ex.raster_candidates(im,warn,high_resolution=high);els=els[:ex.MAX_ELEMENTS];edges=ex.infer_connections(els,segs,*im.size,image=im,diagnostics=details);timings[name].append(round(time.perf_counter()-t,4));assert any(e['kind']=='text' for e in els),(ident,'empty OCR');rep=canonical(els,edges);index={e['id']:i for i,e in enumerate(els) if e['kind']!='text'};ev={'ports':{index[e['id']]:e.get('properties',{}).get('ports',[]) for e in els if e['kind']!='text'},'junctions':details.get('junctions',[]),'paths':sorted((tuple(sorted((index[p['from_id']],index[p['to_id']]))),p['points'] if index[p['from_id']]<index[p['to_id']] else list(reversed(p['points']))) for p in details.get('paths',[]))}
    if name in first:assert first[name]==rep and evidence[name]==ev,(ident,name,'nondeterministic')
    else:
     first[name]=rep;evidence[name]=ev;pp,ee=targets(els,edges,region);sc=score(pp,ee,gt,truth,.5);mapping=sc.pop('matched_ids');matched=set(mapping.values());mapped={tuple(sorted((mapping.get(e['from_id'],'unmatched:'+e['from_id']),mapping.get(e['to_id'],'unmatched:'+e['to_id'])))) for e in ee};causes=Counter('recovered' if e in mapped else 'missing_symbol' if not set(e)<=matched else 'routing' for e in truth);eligible=sum(set(e)<=matched for e in truth)
     row['variants'][name]={'scores':sc,'full_page_symbols':len(rep['boxes']),'full_page_connections':len(edges),'ports':details.get('port_count',0),'junctions':len(details.get('junctions',[])),'observed_paths':len(details.get('paths',[])),'supplemental_connections':details.get('context_supplemental_connections',0),'conditional_endpoints':{'eligible':eligible,'recovered':sc['connections']['tp'],'recall':sc['connections']['tp']/eligible if eligible else 0.},'miss_attribution':dict(causes),'deterministic':True}
     if group in ('new-page','former-test-now-training','r1-test-now-training'):
      row['variants'][name]['recognition']=recognition_metrics(els,details,region,im.size,mapping)
      associated={**details,'junctions':details.get('routing_contacts',[p for p in details.get('junctions',[]) if p.get('connected')])};row['variants'][name]['association_junctions']=recognition_metrics(els,associated,region,im.size,mapping)['junctions']
     export=[{k:v for k,v in e.items() if k!='label'} for e in els if e['kind']!='text'] if region else els
     (folder/f'{group}-{ident}-{name}-predictions.json').write_text(json.dumps({'elements':export,'connections':edges,'recognition':details},ensure_ascii=False,indent=2)+'\n')
  assert first['v7']['boxes']==first['v8']['boxes'],(ident,'symbol detector changed')
  assert set(first['v7']['edges'])<=set(first['v8']['edges']),(ident,'validated candidate removed')
  row['symbols_unchanged']=True;row['old_candidates_retained']=True
  for name in variants:row['variants'][name]['timing']={'seconds':timings[name],'median':statistics.median(timings[name])}
  row['added_true_roi_edges']=row['variants']['v8']['scores']['connections']['tp']-row['variants']['v7']['scores']['connections']['tp'];row['added_false_roi_edges']=row['variants']['v8']['scores']['connections']['fp']-row['variants']['v7']['scores']['connections']['fp'];result['sheets'].append(row);result['aggregate']={}
  for g in sorted({r['group'] for r in result['sheets']}):
   ss=[r for r in result['sheets'] if r['group']==g];result['aggregate'][g]={}
   for v in ss[0]['variants']:
    agg={m:counts(**{k:sum(r['variants'][v]['scores'][m][k] for r in ss) for k in ('tp','fp','fn')}) for m in ('localization','connections')};agg['mean_sheet_median_seconds']=statistics.mean(r['variants'][v]['timing']['median'] for r in ss);agg['conditional_endpoints']={k:sum(r['variants'][v]['conditional_endpoints'][k] for r in ss) for k in ('eligible','recovered')};agg['conditional_endpoints']['recall']=agg['conditional_endpoints']['recovered']/max(1,agg['conditional_endpoints']['eligible']);agg['observed_paths']=sum(r['variants'][v]['observed_paths'] for r in ss);agg['miss_attribution']=dict(sum((Counter(r['variants'][v]['miss_attribution']) for r in ss),Counter()))
    if g in ('new-page','former-test-now-training','r1-test-now-training'):
     agg['recognition']={m:counts(**{k:sum(r['variants'][v]['recognition'][m][k] for r in ss) for k in ('tp','fp','fn')}) for m in ('ports','junctions')};agg['association_junctions']=counts(**{k:sum(r['variants'][v]['association_junctions'][k] for r in ss) for k in ('tp','fp','fn')})
    result['aggregate'][g][v]=agg
  (folder/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(group,ident,{v:{'R':d['scores']['connections']['recall'],'F1':d['scores']['connections']['f1'],'seconds':d['timing']['median'],'ports':d.get('recognition',{}).get('ports'),'junctions':d.get('recognition',{}).get('junctions')} for v,d in row['variants'].items()},flush=True)
 print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
