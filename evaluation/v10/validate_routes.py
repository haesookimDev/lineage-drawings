"""Cached model boxes, real OCR, image routes; no expanded test-label access."""
import copy,json,sys,time
from pathlib import Path
from collections import defaultdict
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v10.data import development,images
from evaluation.benchmark import score,counts
from evaluation.v5.evaluate import targets
from backend.extract import ocr
from backend.geometry import legacy_candidates,_trace_validated,combine_context_pairs
from backend.network import axis_segments
from backend.symbols import neural_style_support,refine_symbols
from backend.sparse_topology import route
from backend.topology_context import model

cv2.setNumThreads(2)
out=ROOT/'.benchmark-data/v10/route-screen';out.mkdir(parents=True,exist_ok=True)
meta=model()[2];rows=[];gates=[]
names=['v8','v9','blend-0.25','blend-0.5','blend-0.75']
configs=[{'name':f'edge{edge}-points{points}-junction{junction}','extra_edge_threshold':edge,'extra_max_points':points,'extra_junction_threshold':junction} for edge in (.5,.65,.8) for points in (4,6,8) for junction in (0.,.4,.55,.7)]
for group,ident,region,path,gt,truth in development():
    im,high=images(path);private=out/f'{group}-{ident}-input.json'
    if private.exists():ctx=json.loads(private.read_text());texts=ctx['texts'];legacy=ctx['legacy'];legacy_lines=ctx['legacy_lines'];axis=ctx['axis'];old=ctx['old_profile']
    else:
        texts=ocr(im,[]);assert texts
        legacy,legacy_lines,_=legacy_candidates(im,texts);axis=axis_segments(im);refined=refine_symbols(im,legacy);old='generic_fallback' if not refined or all(p.get('profile')=='generic_fallback' for _,p in refined) else 'shape_assisted'
        private.write_text(json.dumps({'texts':texts,'legacy':legacy,'legacy_lines':legacy_lines,'axis':axis,'old_profile':old},separators=(',',':'))+'\n')
    for name in names:
        elements=json.loads((ROOT/'.benchmark-data/v10/symbol-screen'/f'{group}-{ident}-{name}.json').read_text())
        fast='shape_assisted' if neural_style_support(im,legacy,[(e['bbox'],e['properties']) for e in elements]) else 'generic_fallback'
        gates.append({'group':group,'id':ident,'candidate':name,'old_profile':old,'local_profile':fast})
        for e in elements:e['properties']['profile']=fast
        elements+=copy.deepcopy(texts);segments=axis if fast=='shape_assisted' else legacy_lines
        cache=out/f'{group}-{ident}-{name}-routes-v2.json'
        if cache.exists():cached=json.loads(cache.read_text());base=[tuple(p) for p in cached['base']];details=cached['details']
        else:
            start=time.perf_counter();base=_trace_validated(elements,segments,*im.size,image=im);fresh,details,ports=route(elements,segments,*im.size,im);cached={'base':base,'details':details,'ports':ports,'elements':elements,'seconds':time.perf_counter()-start};cache.write_text(json.dumps(cached,separators=(',',':'))+'\n')
        for config in configs:
            pairs,composition=combine_context_pairs(elements,base,details['paths'],{**meta,**config});edges=[{'from_id':a,'to_id':b} for a,b in pairs];pp,ee=targets(elements,edges,region);sc=score(pp,ee,gt,truth,.5);sc.pop('matched_ids');rows.append({'model':name,'config':config['name'],'group':group,'id':ident,'scores':sc})
        print(group,ident,name,old,fast,flush=True)
    aggregate={}
    for name in names:
        aggregate[name]={}
        for config in configs:
            selected=[r for r in rows if r['model']==name and r['config']==config['name']];groups=sorted({r['group'] for r in selected});bygroup={}
            for g in groups:
                rr=[r for r in selected if r['group']==g];bygroup[g]={m:counts(**{k:sum(r['scores'][m][k] for r in rr) for k in ('tp','fp','fn')}) for m in ('localization','connections')}
            aggregate[name][config['name']]=bygroup
    (ROOT/'evaluation/v10/ROUTE-VALIDATION.json').write_text(json.dumps({'role':'Explicit development only. OPEN1006-11 and v9 previous35 labels now repair/preservation development. No enlarged test labels.','configs':configs,'rows':rows,'aggregate':aggregate,'style_gates':gates},indent=2)+'\n')
