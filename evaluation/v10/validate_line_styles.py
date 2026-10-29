"""Development-only style-gate ablation, reusing actual fixed graph paths."""
import copy,json,sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v10.data import development,images
from evaluation.benchmark import score,counts
from evaluation.v5.evaluate import targets
from backend.geometry import combine_context_pairs
from backend.topology_context import model,prepare
from backend.line_recovery import path_has_periodic_dashes

cv2.setNumThreads(2);meta=model()[2];rows=[]
configs=[{'name':f'solid{solid}-junction{junction}','extra_solid_threshold':solid,'extra_junction_threshold':junction} for solid in (.5,.65,.8) for junction in (0.,.4)]
for group,ident,region,path,gt,truth in development():
    im,_=images(path);_,ink=prepare(im)
    for name in ['v8','v9','blend-0.25','blend-0.5','blend-0.75']:
        cached=json.loads((ROOT/'.benchmark-data/v10/route-screen'/f'{group}-{ident}-{name}-routes-v2.json').read_text());paths=cached['details']['paths'];elements=cached['elements'];base=[tuple(p) for p in cached['base']]
        for p in paths:p['periodic_dashed']=path_has_periodic_dashes(ink,p['points'],*im.size) if len(p['points'])<=8 else False
        for config in configs:
            pairs,_=combine_context_pairs(elements,base,paths,{**meta,**config});pp,ee=targets(elements,[{'from_id':a,'to_id':b} for a,b in pairs],region);sc=score(pp,ee,gt,truth,.5);sc.pop('matched_ids');rows.append({'model':name,'config':config['name'],'group':group,'id':ident,'scores':sc})
        print(group,ident,name,flush=True)
aggregate={}
for name in sorted({r['model'] for r in rows}):
    aggregate[name]={}
    for config in configs:
        selected=[r for r in rows if r['model']==name and r['config']==config['name']];bygroup={}
        for group in sorted({r['group'] for r in selected}):
            rr=[r for r in selected if r['group']==group];bygroup[group]={m:counts(**{k:sum(r['scores'][m][k] for r in rr) for k in ('tp','fp','fn')}) for m in ('localization','connections')}
        aggregate[name][config['name']]=bygroup
(ROOT/'evaluation/v10/STYLE-VALIDATION.json').write_text(json.dumps({'role':'Development third iteration, old labels only; no enlarged label access. Same paths/boxes with visible periodic pixel classification; no full routing rerun or fabricated alternative paths.','configs':configs,'rows':rows,'aggregate':aggregate},indent=2)+'\n')
