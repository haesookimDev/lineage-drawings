"""A fixed confidence/detail calibration grid on development only."""
import copy,json,sys,time
from pathlib import Path
from collections import defaultdict
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend import neural_symbols as nn
from backend.extract import ocr
from backend.geometry import legacy_candidates
from evaluation.v10.data import development,images
from evaluation.benchmark import score,counts
from evaluation.v5.evaluate import targets
cv2.setNumThreads(2);original=nn.model;rows=[];folder=ROOT/'.benchmark-data/v12-screen';work=list(development());grid=[(name,cut,detail) for name in ('epoch-4','epoch-8') for cut in (.45,.55) for detail in (0,)]
for group,ident,region,path,gt,truth in work:
 view,high=images(path)
 for name,cut,detail in grid:
  p=ROOT/'.benchmark-data/v12-models'/f'{name}.onnx';net=cv2.dnn.readNetFromONNX(str(p));meta={**json.loads(p.with_suffix('.json').read_text()),'threshold':cut,'detail_tiles':detail};nn.model=lambda:(net,meta);t=time.perf_counter();found=nn.detect(high);els=[{'id':f's{i}','kind':'unknown','bbox':b,'source':'neural_symbol_candidate','properties':props} for i,(b,props) in enumerate(found)];pp,_=targets(els,[],region);ss=score(pp,[],gt,truth,.5)['localization'];candidate=f'{name}-score{cut}';rows.append({'group':group,'id':ident,'candidate':candidate,'localization':ss,'seconds':time.perf_counter()-t});(folder/f'{group}-{ident}-{candidate}.json').write_text(json.dumps(els)+'\n');print(rows[-1],flush=True)
 aggregate={g:{n:counts(**{k:sum(r['localization'][k] for r in rows if r['group']==g and r['candidate']==n) for k in ('tp','fp','fn')}) for n in sorted({r['candidate'] for r in rows})} for g in sorted({r['group'] for r in rows})};(ROOT/'evaluation/v12/CALIBRATION-SCREEN.json').write_text(json.dumps({'role':'Development only. Fixed confidence grid .45/.55 after detecting score drift; no new-project inputs.','rows':rows,'aggregate':aggregate},indent=2)+'\n')
nn.model=original
