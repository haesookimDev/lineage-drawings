#!/usr/bin/env python3
"""Class-agnostic symbol localization and equipment-adjacency evaluation.

No evaluation-image annotations or graph labels are passed to extraction.
The production model may load development-only shape prototypes.
The separately named oracle_boxes diagnostic is not an end-to-end score.
"""
import argparse
from datetime import datetime, timezone
import cv2
import PIL
from collections import Counter, defaultdict
import hashlib
import importlib.util
import inspect
import re
import types
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PROTOCOL = json.loads((ROOT/'evaluation/protocol.json').read_text())
NS = {'g':'http://graphml.graphdrawing.org/xmlns'}


def load_gt(path, size):
    root = ET.parse(path).getroot()
    keys = {k.get('id'):k.get('attr.name') for k in root.findall('g:key', NS)}
    nodes = {}
    for node in root.findall('.//g:node', NS):
        d = {keys[v.get('key')]:v.text for v in node.findall('g:data',NS)}
        box = [float(d[k])/size[i%2] for i,k in enumerate(('xmin','ymin','xmax','ymax'))]
        nodes[node.get('id')] = {'id':node.get('id'),'kind':d['label'],'bbox':box}
    adjacent = defaultdict(set)
    for edge in root.findall('.//g:edge',NS):
        a,b = edge.get('source'),edge.get('target')
        adjacent[a].add(b); adjacent[b].add(a)
    semantic = {k:v for k,v in nodes.items() if v['kind'] in PROTOCOL['semantic_labels']}
    # Traverse topology primitives, stopping at the first equipment on each path.
    # This uses the annotated graph, never spatial guesses about crossings.
    edges = set()
    for start in semantic:
        seen = {start}; todo = list(adjacent[start])
        while todo:
            end = todo.pop()
            if end in seen: continue
            seen.add(end)
            if end in semantic:
                edges.add(tuple(sorted((start,end))))
            elif nodes[end]['kind'] != 'background':
                todo.extend(adjacent[end]-seen)
    return list(semantic.values()),edges,Counter(n['kind'] for n in nodes.values())


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def match_boxes(pred,gt,threshold):
    neighbors={p['id']:[g['id'] for score,g in sorted(((iou(p['bbox'],g['bbox']),g) for g in gt),key=lambda x:(-x[0],x[1]['id'])) if score>=threshold] for p in pred}
    # Kuhn augmenting paths maximize match cardinality (not confidence/AP).
    assigned={}
    def augment(pid,seen):
        for gid in neighbors[pid]:
            if gid in seen:continue
            seen.add(gid)
            if gid not in assigned or augment(assigned[gid],seen):
                assigned[gid]=pid;return True
        return False
    for p in pred:augment(p['id'],set())
    return {pid:gid for gid,pid in assigned.items()}


def counts(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def score(pred,edges,gt,gt_edges,threshold):
    mapping=match_boxes(pred,gt,threshold)
    predicted_edges=set()
    for edge in edges:
        a,b=edge['from_id'],edge['to_id']
        if a==b:continue
        # Unmatched endpoints stay distinct and count as false positive edges.
        predicted_edges.add(tuple(sorted((mapping.get(a,'unmatched:'+a),mapping.get(b,'unmatched:'+b)))))
    overlap=predicted_edges & gt_edges
    pby={p['id']:p for p in pred};gby={g['id']:g for g in gt}
    typed=sum(pby[p]['kind']==gby[g]['kind'] for p,g in mapping.items())
    return {'localization':counts(len(mapping),len(pred)-len(mapping),len(gt)-len(mapping)),
            'connections':counts(len(overlap),len(predicted_edges)-len(overlap),len(gt_edges)-len(overlap)),
            'classified_correct':typed,'matched_ids':mapping}


def load_extractor(ref):
    """Isolate the entire historical backend, including geometry and model assets.

    Loading only extract.py would import today's backend.geometry and silently
    contaminate the v2 baseline. Relocate package imports, not extraction logic.
    Hashes below always describe the original immutable Git bytes.
    """
    if ref=='working':
        from backend import extract
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((ROOT/'backend').rglob('*')) if p.is_file()
                 and p.suffix in ('.py','.npz','.json') and '__pycache__' not in p.parts}
        return extract,sources['backend/extract.py'],sources
    revision=subprocess.check_output(['git','rev-parse','--verify',ref+'^{commit}'],cwd=ROOT,text=True).strip()
    package_name='lineage_frozen_'+revision
    snapshot=ROOT/'.benchmark-data'/'extractor-snapshots'/revision
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'backend'],cwd=ROOT,text=True).splitlines()
    sources={}
    for name in names:
        if Path(name).suffix not in ('.py','.npz','.json'):continue
        data=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
        sources[name]=hashlib.sha256(data).hexdigest()
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.suffix=='.py':
            data=re.sub(rb'from backend(?=[. ])',('from '+package_name).encode(),data)
        target.write_bytes(data)
    spec=importlib.util.spec_from_file_location(package_name,snapshot/'backend/__init__.py',submodule_search_locations=[str(snapshot/'backend')])
    package=importlib.util.module_from_spec(spec);sys.modules[package_name]=package;spec.loader.exec_module(package)
    spec=importlib.util.spec_from_file_location(package_name+'.extract',snapshot/'backend/extract.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module,sources['backend/extract.py'],sources


def infer(extractor,elements,segments,image):
    kwargs={'image':image} if 'image' in inspect.signature(extractor.infer_connections).parameters else {}
    return extractor.infer_connections(elements,segments,*image.size,**kwargs)


def aggregate(sheets):
    result={}
    for threshold in ('0.5','0.25'):
        result[threshold]={}
        for metric in ('localization','connections','oracle_boxes'):
            total={k:sum(s['scores'][threshold][metric][k] for s in sheets) for k in ('tp','fp','fn')}
            result[threshold][metric]=counts(**total)
        result[threshold]['classified_correct']=sum(s['scores'][threshold]['classified_correct'] for s in sheets)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'.benchmark-data/open100')
    p.add_argument('--extractor-ref',default='working')
    p.add_argument('--split',choices=['development','holdout','all'],default='development')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',default=None,help='Evaluation role, separate from legacy split names')
    p.add_argument('--ids',help='Explicit comma-separated sheet IDs; used for the predeclared synthetic confirmation set')
    args=p.parse_args();extractor,digest,sources=load_extractor(args.extractor_ref)
    ids=PROTOCOL['development_ids'] if args.split=='development' else PROTOCOL['holdout_ids'] if args.split=='holdout' else PROTOCOL['development_ids']+PROTOCOL['holdout_ids']
    if args.ids:ids=args.ids.split(',')
    sheets=[];args.output.mkdir(parents=True,exist_ok=True)
    for ident in ids:
        image_path=args.data/f'{ident}.png'
        with Image.open(image_path) as raw:
            original=raw.size;im=raw.convert('RGB');im.thumbnail((extractor.MAX_SIDE,extractor.MAX_SIDE))
        gt,gt_edges,labels=load_gt(args.data/f'{ident}.graphml',original)
        warnings=[];start=time.perf_counter()
        elements,segments=extractor.raster_candidates(im,warnings)
        elements=elements[:extractor.MAX_ELEMENTS]
        # Every OPEN100 sheet has text: an empty OCR result here indicates an
        # invalid evaluation environment (e.g. a missing tessdata/configs/tsv).
        if not any(e['kind']=='text' for e in elements):
            raise RuntimeError(f'OPEN100 sheet {ident}: zero OCR words. Check Tesseract, eng/kor traineddata and configs/tsv; refusing an invalid OCR comparison.')
        edges=infer(extractor,elements,segments,im)
        elapsed=time.perf_counter()-start
        symbols=[e for e in elements if e['kind']!='text']
        fallback=symbols and all(e.get('properties',{}).get('profile')=='generic_fallback' for e in symbols)
        oracle_gt=[{**g,'properties':{'profile':'generic_fallback'}} for g in gt] if fallback else gt
        oracle=infer(extractor,oracle_gt,segments,im)
        scores={}
        for threshold in (0.5,0.25):
            result=score(symbols,edges,gt,gt_edges,threshold)
            oe={tuple(sorted((e['from_id'],e['to_id']))) for e in oracle}
            result['oracle_boxes']=counts(len(oe & gt_edges),len(oe-gt_edges),len(gt_edges-oe))
            scores[str(threshold)]=result
        sheet={'id':ident,'image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'original_size':original,'rendered_size':im.size,'gt_labels':dict(labels),'gt_symbols':len(gt),'gt_connections':len(gt_edges),'pred_symbols':len(symbols),'pred_connections':len(edges),'ocr_words':sum(e['kind']=='text' for e in elements),'elapsed_seconds':round(elapsed,3),'warnings':warnings,'scores':scores}
        sheets.append(sheet)
        (args.output/f'{ident}-predictions.json').write_text(json.dumps({'elements':elements,'connections':edges},ensure_ascii=False,indent=2)+'\n')
        print(ident,json.dumps({k:sheet[k] for k in ('gt_symbols','pred_symbols','gt_connections','pred_connections','elapsed_seconds')}),flush=True)
    report={'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'evaluated_at':datetime.now(timezone.utc).isoformat(),'opencv':cv2.__version__,'pillow':PIL.__version__,'extractor_sources':sources,'protocol':PROTOCOL,'extractor_ref':args.extractor_ref,'extractor_sha256':digest,'split':args.split,'explicit_ids':args.ids,'evaluation_role':args.role or args.split,'sheets':sheets,'aggregate':aggregate(sheets),'python':sys.version,'ocr_version':subprocess.run(['tesseract','--version'],capture_output=True,text=True).stdout.splitlines()[0]}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['aggregate'],indent=2))

if __name__=='__main__':main()
