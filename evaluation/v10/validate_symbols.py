"""Development-only localization screen; expanded held-out labels never loaded."""
import hashlib,json,sys,time,importlib
from pathlib import Path
from collections import defaultdict
from PIL import Image
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_extractor,load_gt,score,counts
from evaluation.v9.data import validation
from evaluation.v5.evaluate import targets
from backend import neural_symbols as nn

cv2.setNumThreads(2)
out=ROOT/'.benchmark-data/v10/symbol-screen';out.mkdir(parents=True,exist_ok=True)
work=[]
for group,ident,region,high,gt,truth in validation():
    work.append((group,ident,region,high,gt,truth))
for ident in range(6,12):
    path=ROOT/'.benchmark-data/open100'/f'{ident}.png'
    with Image.open(path) as raw:
        gt,truth,*_=load_gt(path.with_suffix('.graphml'),raw.size);high=raw.convert('RGB');high.thumbnail((4800,4800))
    work.append(('open100-repair',str(ident),None,high,gt,truth))
for s in json.loads((ROOT/'evaluation/v9/new-project-labels.json').read_text())['sheets'].values():
    high=Image.open(ROOT/'.benchmark-data/v9-held'/s['group']/(s['id']+'-4800.png')).convert('RGB')
    work.append((s['group']+'-preservation',s['id'],s,high,s['nodes'],set(tuple(e) for e in s['edges'])))
rows=[]
base=load_extractor('c9cef5a')[0]
baseline_nn=importlib.import_module(base.__package__+'.neural_symbols')
endpoints=[('v8',baseline_nn.model()),('v9',nn.model())]
for fraction in (.25,.5,.75):
    p=ROOT/'.benchmark-data/v10/models'/f'blend-{fraction}.onnx'
    net=cv2.dnn.readNetFromONNX(str(p));meta=json.loads(p.with_suffix('.json').read_text());endpoints.append((f'blend-{fraction}',(net,meta)))
for name,(net,meta) in endpoints:
    nn.model=lambda: (net,meta)
    for group,ident,region,high,gt,truth in work:
        start=time.perf_counter();found=nn.detect(high);els=[{'id':f's{j}','kind':'unknown','bbox':box,'source':'neural_symbol_candidate','properties':prop} for j,(box,prop) in enumerate(found)];pp,_=targets(els,[],region);sc=score(pp,[],gt,truth,.5)['localization'];row={'candidate':name,'group':group,'id':ident,'scores':sc,'seconds':time.perf_counter()-start};rows.append(row)
        (out/f'{group}-{ident}-{name}.json').write_text(json.dumps(els,separators=(',',':'))+'\n')
        print(name,group,ident,sc,flush=True)
    aggregate={}
    for group in sorted({r['group'] for r in rows}):
        aggregate[group]={n:counts(**{k:sum(r['scores'][k] for r in rows if r['candidate']==n and r['group']==group) for k in ('tp','fp','fn')}) for n,_ in endpoints if any(r['candidate']==n and r['group']==group for r in rows)}
    (ROOT/'evaluation/v10/SYMBOL-VALIDATION.json').write_text(json.dumps({'role':'Development repair/validation only; v9 old ROIs explicitly reused for preservation; no new enlarged region labels read.','rows':rows,'aggregate':aggregate},indent=2)+'\n')
