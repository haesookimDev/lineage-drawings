#!/usr/bin/env python3
"""Frozen v4 / learned v5 comparison with explicit training, validation and test roles."""
import argparse,hashlib,json,sys,time,statistics
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_extractor,load_gt,score,counts,infer


def canonical(elements,edges):
    symbols=[e for e in elements if e['kind']!='text'];idx={e['id']:i for i,e in enumerate(symbols)}
    return {'boxes':[e['bbox'] for e in symbols], 'edges':sorted(tuple(sorted((idx[e['from_id']],idx[e['to_id']]))) for e in edges)}


def within(box,roi):
    x,y=(box[0]+box[2])/2,(box[1]+box[3])/2
    return roi[0]<=x<=roi[2] and roi[1]<=y<=roi[3]


def targets(elements,edges,region):
    symbols=[e for e in elements if e['kind']!='text']
    if region:
        symbols=[e for e in symbols if within(e['bbox'],region['roi']) and not any(within(e['bbox'],b) for b in region['ignore'])]
    ids={e['id'] for e in symbols}
    return symbols,[e for e in edges if e['from_id'] in ids and e['to_id'] in ids]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=int,default=3);ap.add_argument('--group',choices=['open100','real','synthetic','all'],default='all');ap.add_argument('--output',type=Path,default=ROOT/'evaluation/v5/results');args=ap.parse_args()
    freeze=json.loads((ROOT/'evaluation/v5/FROZEN.json').read_text())
    for name,digest in freeze['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    labels=json.loads((ROOT/'evaluation/v5/real-labels.json').read_text())
    assert hashlib.sha256((ROOT/'evaluation/v5/real-labels.json').read_bytes()).hexdigest()==freeze['labels_sha256']
    variants={k:load_extractor(ref)[0] for k,ref in [('v4',freeze['baseline']),('v5','working')]}
    work=[]
    plan=json.loads((ROOT/'evaluation/v5/PLAN.json').read_text())
    if args.group in ('open100','all'):
        for i in range(4,12):work.append(('open100',str(i),None,ROOT/'.benchmark-data/open100'/f'{i}.png','threshold-validation' if i in (4,5) else 'historical-regression'))
    if args.group in ('real','all'):
        old_labels=json.loads((ROOT/'evaluation/v4/pahala-labels.json').read_text())['sheets']
        for ident,region in old_labels.items():work.append(('pahala-training',ident,region,ROOT/'.benchmark-data/pahala'/f'{ident}.png','training-diagnostic'))
        for key,region in labels['sheets'].items():
            group='pahala-validation' if region['role']=='threshold-validation' else 'pahala-unseen-page' if region['group']=='pahala' else region['group']
            work.append((group,region['id'],region,ROOT/'.benchmark-data'/region['group']/(region['id']+'.png'),region['role']))
    if args.group in ('synthetic','all'):
        for name in plan['threshold_validation']['synthetic_members']:
            ident=Path(name).stem;work.append(('synthetic-validation',ident,None,ROOT/'.benchmark-data/v5-synthetic'/(ident+'.png'),'threshold-validation'))
        for i in [9,34,385,285,317,281,82,382]:work.append(('synthetic-regression',str(i),None,ROOT/'.benchmark-data/pid-confirmation'/f'{i}.png','historical-regression'))
    result={'baseline':freeze['baseline'],'frozen_sha256':hashlib.sha256((ROOT/'evaluation/v5/FROZEN.json').read_bytes()).hexdigest(),'evaluated_at':datetime.now(timezone.utc).isoformat(),'runs':args.runs,'timing_scope':'in-memory RGB input at max side 2400; OCR+shape+line+connection inference, excluding decode/export/oracle diagnostic; sequential paired runs, reversed order on alternate repeats. Not throughput or service SLA.','sheets':[]}
    for group,ident,region,path,role in work:
        with Image.open(path) as raw:
            original=raw.size;image=raw.convert('RGB');image.thumbnail((2400,2400))
        if region:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==region['image_sha256']
            gt=region['nodes'];truth={tuple(e) for e in region['edges']}
        else:gt,truth,_=load_gt(path.with_suffix('.graphml'),original)
        row={'group':group,'role':role,'roi_only':region is not None,'id':ident,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rendered_size':list(image.size),'gt_symbols':len(gt),'gt_connections':len(truth),'variants':{}}
        first={};timings={k:[] for k in variants}
        for repeat in range(args.runs):
            order=list(variants) if repeat%2==0 else list(reversed(variants))
            for name in order:
                extractor=variants[name];warnings=[];start=time.perf_counter();elements,segments=extractor.raster_candidates(image,warnings);elements=elements[:extractor.MAX_ELEMENTS];edges=infer(extractor,elements,segments,image);elapsed=time.perf_counter()-start;timings[name].append(round(elapsed,4))
                assert any(e['kind']=='text' for e in elements),(group,ident,name,'empty OCR')
                representation=canonical(elements,edges)
                if name in first:assert representation==first[name],(group,ident,name,'non-deterministic predictions')
                else:
                    first[name]=representation;pred,selected_edges=targets(elements,edges,region);scores=score(pred,selected_edges,gt,truth,.5);mapping=scores['matched_ids'];mapped={tuple(sorted((mapping.get(e['from_id'],'unmatched:'+e['from_id']),mapping.get(e['to_id'],'unmatched:'+e['to_id'])))) for e in selected_edges}
                    matched=set(mapping.values());eligible={mapping[e['id']] for e in pred if e['id'] in mapping and e.get('properties',{}).get('connection_eligible',True)};causes=Counter()
                    for edge in truth:
                        key='recovered' if edge in mapped else 'missing_symbol' if not set(edge)<=matched else 'disabled_symbol_port' if not set(edge)<=eligible else 'line_or_port_routing';causes[key]+=1
                    fallback=all(e.get('properties',{}).get('profile')=='generic_fallback' for e in elements if e['kind']!='text')
                    oracle_gt=[{**g,'properties':{'profile':'generic_fallback'}} for g in gt] if fallback else gt
                    oracle=infer(extractor,oracle_gt,segments,image);oe={tuple(sorted((e['from_id'],e['to_id']))) for e in oracle}
                    scores.pop('matched_ids');scores['oracle_boxes']=counts(len(oe&truth),len(oe-truth),len(truth-oe))
                    row['variants'][name]={'scores':scores,'miss_attribution':dict(causes),'pred_symbols':len(pred),'pred_connections':len(selected_edges),'full_page_symbols':len(representation['boxes']),'full_page_connections':len(edges),'ocr_words':sum(e['kind']=='text' for e in elements),'warnings':warnings,'profile':'generic' if fallback else 'shape_assisted','deterministic':True}
                    args.output.mkdir(parents=True,exist_ok=True)
                    # No full Pahala OCR text or image is republished. Geometry only.
                    prediction={'elements':elements if group=='open100' or group.startswith('synthetic') else [{k:v for k,v in e.items() if k!='label'} for e in elements if e['kind']!='text'],'connections':edges}
                    (args.output/f'{group}-{ident}-{name}-predictions.json').write_text(json.dumps(prediction,ensure_ascii=False,indent=2)+'\n')
        for name in variants:row['variants'][name]['timing']={'seconds':timings[name],'median':statistics.median(timings[name]),'min':min(timings[name]),'max':max(timings[name])}
        result['sheets'].append(row)
        print(group,ident,{k:{'f1':round(v['scores']['connections']['f1'],4),'recall':round(v['scores']['connections']['recall'],4),'seconds':v['timing']['median']} for k,v in row['variants'].items()},flush=True)
        result['aggregate']={}
        for group_name in sorted({r['group'] for r in result['sheets']}):
            selected=[r for r in result['sheets'] if r['group']==group_name];result['aggregate'][group_name]={}
            for name in variants:
                aggregate={}
                for metric in ['localization','connections','oracle_boxes']:
                    total={k:sum(r['variants'][name]['scores'][metric][k] for r in selected) for k in ['tp','fp','fn']};aggregate[metric]=counts(**total)
                aggregate['mean_sheet_median_seconds']=statistics.mean(r['variants'][name]['timing']['median'] for r in selected)
                aggregate['miss_attribution']=dict(sum((Counter(r['variants'][name]['miss_attribution']) for r in selected),Counter()))
                result['aggregate'][group_name][name]=aggregate
        (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['aggregate'],indent=2))

if __name__=='__main__':main()
