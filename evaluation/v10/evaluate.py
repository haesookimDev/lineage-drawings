"""Frozen production-scale full inference, paired latency and expanded labels."""
import argparse,hashlib,json,statistics,sys,time
from collections import Counter
from pathlib import Path
from PIL import Image
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_extractor,load_gt,score,counts
from evaluation.v5.evaluate import targets,canonical,within
from evaluation.v10.data import images

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['expanded','repair','synthetic'],required=True);parser.add_argument('--runs',type=int,default=3);args=parser.parse_args()
    freeze=json.loads((ROOT/'evaluation/v10/FROZEN.json').read_text())
    for name,value in freeze['sources'].items():assert digest(ROOT/name)==value,name
    label_path=ROOT/'evaluation/v10/expanded-labels.json';assert digest(label_path)==freeze['expanded_labels_sha256']
    cv2.setNumThreads(2);variants={'v9':load_extractor(freeze['baseline'])[0],'v10':load_extractor('working')[0]}
    # Initialize network execution once, equally, outside the measured page runs.
    import importlib,numpy as np
    for ex in variants.values():
        nn=importlib.import_module(ex.__package__+'.neural_symbols');net,meta=nn.model();blob,_=nn.preprocess(np.full((640,640,3),255,np.uint8));net.setInput(blob);net.forward()
    work=[]
    if args.scope=='expanded':
        labels=json.loads(label_path.read_text())
        for key,s in labels['sheets'].items():work.append((key,s,ROOT/'.benchmark-data/v9-held'/s['group']/(s['id']+'-4800.png'),s['nodes'],{tuple(e) for e in s['edges']}))
    else:
        group='open100' if args.scope=='repair' else 'pid-confirmation'
        for ident in (range(6,12) if args.scope=='repair' else [9,34,385,285,317,281,82,382]):
            path=ROOT/'.benchmark-data'/group/f'{ident}.png'
            with Image.open(path) as raw:gt,truth,*_=load_gt(path.with_suffix('.graphml'),raw.size)
            work.append((f'{group}-{ident}',None,path,gt,truth))
    folder=ROOT/'evaluation/v10'/('results' if args.scope=='expanded' else args.scope);folder.mkdir(exist_ok=True)
    result={'role':'Expanded held-out annotations on untrained pages with prior whole-image/project exposure; old35 labels used for preservation selection, extension labels not selected on.' if args.scope=='expanded' else 'Development-exposed historical repair confirmation' if args.scope=='repair' else 'Historical synthetic regression, not unseen project evidence','runs':args.runs,'baseline':freeze['baseline'],'frozen_sha256':digest(ROOT/'evaluation/v10/FROZEN.json'),'timing_scope':'Same-host paired rotated2-variant order, in-memory OCR+max4800 neural+max2400 geometry/topology; independent original resizes, 2 OpenCV threads/OMP1. Mean of per-sheet3-run medians, decode/PDF/network/storage excluded, not Pod SLA.','sheets':[]}
    for key,region,path,gt,truth in work:
        im,high=images(path);first={};times={n:[] for n in variants};row={'key':key,'source_sha256':digest(path),'rendered_size':list(im.size),'symbol_size':list(high.size),'gt_symbols':len(gt),'gt_edges':len(truth),'roi_only':region is not None,'variants':{}}
        for repeat in range(args.runs):
            order=list(variants) if repeat%2==0 else list(reversed(variants))
            for name in order:
                ex=variants[name];rec={};warnings=[];start=time.perf_counter();els,segs=ex.raster_candidates(im,warnings,high_resolution=high);els=els[:ex.MAX_ELEMENTS];edges=ex.infer_connections(els,segs,*im.size,image=im,diagnostics=rec);times[name].append(time.perf_counter()-start)
                assert any(e['kind']=='text' for e in els),(key,name,'invalid OCR');assert sum(e['kind']!='text' for e in els)<=200 and len(edges)<=500
                symbols=[e for e in els if e['kind']!='text'];indices={e['id']:i for i,e in enumerate(symbols)}
                representation={'geometry':canonical(els,edges),'ports':[e.get('properties',{}).get('ports',[]) for e in symbols],'junctions':rec.get('junctions'),'paths':sorted((tuple(sorted((indices[p['from_id']],indices[p['to_id']]))),p['points'] if indices[p['from_id']]<indices[p['to_id']] else list(reversed(p['points']))) for p in rec.get('paths',[]))}
                if name in first:assert first[name]==representation,(key,name,'nondeterminism')
                else:
                    first[name]=representation;pp,ee=targets(els,edges,region);sc=score(pp,ee,gt,truth,.5);mapping=sc.pop('matched_ids');matched=set(mapping.values());mapped={tuple(sorted((mapping.get(e['from_id'],'unmatched:'+e['from_id']),mapping.get(e['to_id'],'unmatched:'+e['to_id'])))) for e in ee};miss=Counter('recovered' if edge in mapped else 'missing_symbol' if not set(edge)<=matched else 'routing' for edge in truth)
                    data={'scores':sc,'miss_attribution':dict(miss),'symbols':len(symbols),'connections':len(edges),'ocr_words':sum(e['kind']=='text' for e in els),'warnings':warnings,'deterministic':True if args.runs>1 else None,'profile':sorted({e.get('properties',{}).get('profile') for e in symbols}),'ports':rec.get('port_count'),'junctions':len(rec.get('junctions',[]))}
                    if region:
                        subsets={k:{'tp':len(mapped&{tuple(e) for e in values}),'total':len(values)} for k,values in region['edge_subsets'].items()};data['edge_subsets']=subsets
                        old_ids=set(region['old_node_ids']);old_truth={tuple(e) for e in region['old_edges']};new_truth=truth-old_truth;new_pred={e for e in mapped if not set(e)<=old_ids}
                        data['extension_connections']=counts(len(new_pred&new_truth),len(new_pred-new_truth),len(new_truth-new_pred))
                        old_region={**region,'roi':region['old_roi'],'nodes':[n for n in gt if n['id'] in old_ids],'edges':region['old_edges']};op,oe=targets(els,edges,old_region);oldsc=score(op,oe,old_region['nodes'],old_truth,.5);oldsc.pop('matched_ids');data['original_roi_scores']=oldsc
                    row['variants'][name]=data
                    # Government source text stays private. Reproducible output
                    # contains generic candidate geometry/evidence only.
                    public={'elements':[{k:v for k,v in e.items() if k!='label'} for e in symbols],'connections':edges,'recognition':rec}
                    (folder/(key+'-'+name+'-predictions.json')).write_text(json.dumps(public,separators=(',',':'))+'\n')
        for name in variants:row['variants'][name]['timing']={'seconds':times[name],'median':statistics.median(times[name])}
        result['sheets'].append(row);aggregate={}
        for name in variants:
            rr=[r['variants'][name] for r in result['sheets']];d={m:counts(**{k:sum(r['scores'][m][k] for r in rr) for k in ('tp','fp','fn')}) for m in ('localization','connections')};d['mean_sheet_median_seconds']=statistics.mean(r['timing']['median'] for r in rr);d['miss_attribution']=dict(sum((Counter(r['miss_attribution']) for r in rr),Counter()))
            if region:
                d['extension_connections']=counts(**{k:sum(r['extension_connections'][k] for r in rr) for k in ('tp','fp','fn')});d['original_roi_scores']={m:counts(**{k:sum(r['original_roi_scores'][m][k] for r in rr) for k in ('tp','fp','fn')}) for m in ('localization','connections')};d['edge_subsets']={s:{'tp':sum(r['edge_subsets'][s]['tp'] for r in rr),'total':sum(r['edge_subsets'][s]['total'] for r in rr)} for s in ('dashed','branch')}
            aggregate[name]=d
        result['aggregate']=aggregate;(folder/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(key,{name:row['variants'][name] for name in variants},flush=True)
    print('FINAL',json.dumps(result['aggregate']),flush=True)

if __name__=='__main__':main()
