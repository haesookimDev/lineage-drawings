#!/usr/bin/env python3
"""Paired v6/v7 end-to-end evaluation; fixed symbols and validated adjacency."""
import argparse,hashlib,json,statistics,sys,time,math
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_extractor,load_gt,score,counts
from evaluation.v5.evaluate import canonical,targets,within

def point_metrics(pred,truth,size,element_mapping=None):
    candidates=[]
    for i,p in enumerate(pred):
        for j,t in enumerate(truth):
            if 'type' in t and p.get('type')!=t['type']:continue
            if 'element_id' in t and (element_mapping or {}).get(p.get('element_id'))!=t['element_id']:continue
            dist=math.hypot(*[(p['point'][k]-t['point'][k])*size[k] for k in (0,1)])
            if dist<=6:candidates.append((dist,i,j))
    used_p,used_t=set(),set()
    for _,i,j in sorted(candidates):
        if i not in used_p and j not in used_t:used_p.add(i);used_t.add(j)
    return counts(tp=len(used_p),fp=len(pred)-len(used_p),fn=len(truth)-len(used_t))

def recognition_metrics(els,details,region,size,mapping):
    inside=lambda p:within(p['point']*2,region['roi']) and not any(within(p['point']*2,b) for b in region['ignore'])
    ports=[{**p,'element_id':e['id']} for e in els if e['kind']!='text' for p in e.get('properties',{}).get('ports',[]) if inside(p)]
    junctions=[j for j in details.get('junctions',[]) if inside(j)]
    return {'ports':point_metrics(ports,region['ports'],size,mapping),'junctions':point_metrics(junctions,region['junctions'],size)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=int,default=3);args=ap.parse_args()
    freeze=json.loads((ROOT/'evaluation/v7/FROZEN.json').read_text())
    for p,h in freeze['sources'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    folder=ROOT/'evaluation/v7/results';folder.mkdir(exist_ok=True)
    labels=json.loads((ROOT/'evaluation/v7/new-page-labels.json').read_text())['sheets'];work=[]
    for key,s in labels.items():work.append(('new-page',key,s,ROOT/'.benchmark-data/v7-highres'/s['group']/(s['id']+'.png')))
    for key,s in json.loads((ROOT/'evaluation/v6/new-page-labels.json').read_text())['sheets'].items():work.append(('v6-roi-regression',key,s,ROOT/'.benchmark-data/v6-highres'/s['group']/(s['id']+'.png')))
    for i in range(6,12):work.append(('open100-regression',str(i),None,ROOT/'.benchmark-data/open100'/f'{i}.png'))
    s=json.loads((ROOT/'evaluation/v5/real-labels.json').read_text())['sheets']['pahala-p21'];work.append(('pahala-p21-regression','p21',s,ROOT/'.benchmark-data/v6-highres/pahala/p21.png'))
    for i in [9,34,385,285,317,281,82,382]:work.append(('synthetic-regression',str(i),None,ROOT/'.benchmark-data/pid-confirmation'/f'{i}.png'))
    variants={v:load_extractor(ref)[0] for v,ref in [('v6',freeze['baseline']),('v7','working')]}
    report={'baseline':freeze['baseline'],'frozen_sha256':hashlib.sha256((ROOT/'evaluation/v7/FROZEN.json').read_bytes()).hexdigest(),'evaluated_at':datetime.now(timezone.utc).isoformat(),'runs':args.runs,'timing_scope':'Paired in-memory OCR + high-res symbol inference + line/edge recognition, max4800 symbol/max2400 view, three repeats reversed variant order; RGB decode/render and upload excluded. Sequential local CPU, not Pod SLA or throughput. Recognition v7 is advisory, so adjacency equality is a required invariant.','sheets':[]}
    for group,ident,region,path in work:
        with Image.open(path) as raw:size=raw.size;im=raw.convert('RGB');high=im.copy();high.thumbnail((4800,4800));im.thumbnail((2400,2400))
        gt,truth=(region['nodes'],set(tuple(e) for e in region['edges'])) if region else load_gt(path.with_suffix('.graphml'),size)[:2]
        row={'group':group,'id':ident,'roi_only':region is not None,'role':'new-pages-known-projects' if group=='new-page' else 'historical-regression','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_size':size,'rendered_size':im.size,'gt_symbols':len(gt),'gt_edges':len(truth),'variants':{}};first={};timings={v:[] for v in variants};evidence={}
        for repeat in range(args.runs):
            for name in (list(variants) if repeat%2==0 else list(reversed(variants))):
                ex=variants[name];warn=[];details={};t=time.perf_counter();els,segs=ex.raster_candidates(im,warn,high_resolution=high);els=els[:ex.MAX_ELEMENTS];edges=ex.infer_connections(els,segs,*im.size,image=im,diagnostics=details);timings[name].append(round(time.perf_counter()-t,4));assert any(e['kind']=='text' for e in els),'empty OCR';rep=canonical(els,edges)
                idx={e['id']:i for i,e in enumerate(els) if e['kind']!='text'};ev={'ports':{idx[e['id']]:e['properties']['ports'] for e in els if e['kind']!='text' and 'ports' in e.get('properties',{})},'junctions':details.get('junctions',[]),'paths':sorted((tuple(sorted((idx[p['from_id']],idx[p['to_id']]))),p['points'] if idx[p['from_id']]<idx[p['to_id']] else list(reversed(p['points']))) for p in details.get('paths',[]))}
                if name in first:assert first[name]==rep and evidence[name]==ev,(ident,name,'nondeterministic')
                else:
                    first[name]=rep;evidence[name]=ev;pp,ee=targets(els,edges,region);sc=score(pp,ee,gt,truth,.5);mapping=sc.pop('matched_ids');matched=set(mapping.values());mapped={tuple(sorted((mapping.get(e['from_id'],'unmatched:'+e['from_id']),mapping.get(e['to_id'],'unmatched:'+e['to_id'])))) for e in ee};causes=Counter('recovered' if e in mapped else 'missing_symbol' if not set(e)<=matched else 'routing' for e in truth)
                    row['variants'][name]={'scores':sc,'miss_attribution':dict(causes),'full_page_symbols':len(rep['boxes']),'full_page_connections':len(edges),'ports':details.get('port_count',0),'junctions':len(details.get('junctions',[])),'observed_paths':len(details.get('paths',[])),'recognition_tiles':details.get('recognition_tiles',0),'deterministic':True}
                    if group=='new-page':row['variants'][name]['recognition']=recognition_metrics(els,details,region,im.size,mapping)
                    # Engineering source drawings, full OCR and source crops are not redistributed.
                    real=region is not None;export=[{k:v for k,v in e.items() if k!='label'} for e in els if e['kind']!='text'] if real else els
                    (folder/f'{group}-{ident}-{name}-predictions.json').write_text(json.dumps({'elements':export,'connections':edges,'recognition':details if name=='v7' else {}},ensure_ascii=False,indent=2)+'\n')
        assert first['v6']==first['v7'],(ident,'validated adjacency or symbols regressed')
        for name in variants:row['variants'][name]['timing']={'seconds':timings[name],'median':statistics.median(timings[name])}
        row['symbols_and_adjacency_unchanged']=True;report['sheets'].append(row);report['aggregate']={}
        for g in sorted({r['group'] for r in report['sheets']}):
            ss=[r for r in report['sheets'] if r['group']==g];report['aggregate'][g]={}
            for v in variants:
                agg={m:counts(**{k:sum(r['variants'][v]['scores'][m][k] for r in ss) for k in ('tp','fp','fn')}) for m in ('localization','connections')};agg['mean_sheet_median_seconds']=statistics.mean(r['variants'][v]['timing']['median'] for r in ss);agg['observed_paths']=sum(r['variants'][v]['observed_paths'] for r in ss);agg['miss_attribution']=dict(sum((Counter(r['variants'][v]['miss_attribution']) for r in ss),Counter()))
                if g=='new-page':agg['recognition']={m:counts(**{k:sum(r['variants'][v]['recognition'][m][k] for r in ss) for k in ('tp','fp','fn')}) for m in ('ports','junctions')}
                report['aggregate'][g][v]=agg
        (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(group,ident,{v:{'R':r['scores']['connections']['recall'],'seconds':r['timing']['median'],'ports':r['ports'],'paths':r['observed_paths']} for v,r in row['variants'].items()},flush=True)
    print(json.dumps(report['aggregate'],indent=2))
if __name__=='__main__':main()
