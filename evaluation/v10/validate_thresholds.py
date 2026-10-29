"""Last bounded development trial: two blends at confidence0.45, no test labels."""
import copy,json,sys,time
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v10.data import development,images
from evaluation.benchmark import score,counts
from evaluation.v5.evaluate import targets
from backend import neural_symbols as nn
from backend.symbols import neural_style_support
from backend.geometry import _trace_validated,combine_context_pairs
from backend.sparse_topology import route
from backend.topology_context import model
cv2.setNumThreads(2);meta=model()[2];rows=[];candidates=[]
configs=[{'name':f'edge{edge}-points{points}','extra_edge_threshold':edge,'extra_max_points':points} for edge in (.5,.65,.8) for points in (6,8)]
for fraction in (.25,.5):
    path=ROOT/'.benchmark-data/v10/models'/f'blend-{fraction}.onnx';net=cv2.dnn.readNetFromONNX(str(path));nmeta={**json.loads(path.with_suffix('.json').read_text()),'threshold':.45};nn.model=lambda:(net,nmeta);name=f'blend-{fraction}-cut0.45';candidates.append({'name':name,'manifest':nmeta})
    for group,ident,region,source,gt,truth in development():
        im,high=images(source);ctx=json.loads((ROOT/'.benchmark-data/v10/route-screen'/f'{group}-{ident}-input.json').read_text());cache=ROOT/'.benchmark-data/v10/route-screen'/f'{group}-{ident}-{name}-routes-v2.json'
        if cache.exists():d=json.loads(cache.read_text());elements=d['elements'];base=[tuple(p) for p in d['base']];details=d['details']
        else:
            found=nn.detect(high);profile='shape_assisted' if neural_style_support(im,ctx['legacy'],found) else 'generic_fallback'
            elements=[{'id':f's{j}','kind':'unknown','bbox':box,'source':'neural_symbol_candidate','properties':{**prop,'profile':profile}} for j,(box,prop) in enumerate(found)]+copy.deepcopy(ctx['texts']);segments=ctx['axis'] if profile=='shape_assisted' else ctx['legacy_lines'];base=_trace_validated(elements,segments,*im.size,image=im);_,details,ports=route(elements,segments,*im.size,im);d={'elements':elements,'base':base,'details':details,'ports':ports};cache.write_text(json.dumps(d,separators=(',',':'))+'\n')
        for config in configs:
            pairs,_=combine_context_pairs(elements,base,details['paths'],{**meta,**config});pp,ee=targets(elements,[{'from_id':a,'to_id':b} for a,b in pairs],region);sc=score(pp,ee,gt,truth,.5);sc.pop('matched_ids');rows.append({'model':name,'config':config['name'],'group':group,'id':ident,'scores':sc})
        print(name,group,ident,flush=True)
aggregate={}
for candidate in candidates:
    name=candidate['name'];aggregate[name]={}
    for config in configs:
        selected=[r for r in rows if r['model']==name and r['config']==config['name']];bygroup={}
        for group in sorted({r['group'] for r in selected}):
            rr=[r for r in selected if r['group']==group];bygroup[group]={m:counts(**{k:sum(r['scores'][m][k] for r in rr) for k in ('tp','fp','fn')}) for m in ('localization','connections')}
        aggregate[name][config['name']]=bygroup
(ROOT/'evaluation/v10/THRESHOLD-VALIDATION.json').write_text(json.dumps({'role':'Final bounded development fourth trial, prior20 development pages only. Lower confidence0.45 on blend0.25/0.5; no enlarged labels inspected. No further search after final scoring.','candidates':candidates,'configs':configs,'rows':rows,'aggregate':aggregate},indent=2)+'\n')
