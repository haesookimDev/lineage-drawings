"""Development-only detector screen. New projects never opened here."""
import hashlib,json,sys,time,copy
from pathlib import Path
from collections import defaultdict
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend import neural_symbols as nn
from backend.extract import ocr
from backend.geometry import legacy_candidates
from evaluation.v10.data import development,images
from evaluation.benchmark import score,counts
from evaluation.v5.evaluate import targets
cv2.setNumThreads(2);rows=[];original=nn.model;folder=ROOT/'.benchmark-data/v12-screen';folder.mkdir(exist_ok=True)
work=list(development())
for group,ident,region,path,gt,truth in work:
 view,high=images(path);text=ocr(view,[]);hints=[b for b,_ in legacy_candidates(view,text)[0]]
 for name in ('initial','epoch-4','epoch-8'):
  p=ROOT/'.benchmark-data/v12-models'/f'{name}.onnx';net=cv2.dnn.readNetFromONNX(str(p));meta=json.loads(p.with_suffix('.json').read_text());nn.model=lambda:(net,meta);start=time.perf_counter();found=nn.detect(high,proposal_hints=hints);els=[{'id':f's{i}','kind':'unknown','bbox':b,'source':'neural_symbol_candidate','properties':props} for i,(b,props) in enumerate(found)];pp,_=targets(els,[],region);ss=score(pp,[],gt,truth,.5)['localization'];row={'group':group,'id':ident,'candidate':name,'localization':ss,'seconds':time.perf_counter()-start};rows.append(row);(folder/f'{group}-{ident}-{name}.json').write_text(json.dumps(els)+'\n');print(row,flush=True)
 aggregate={}
 for g in sorted({r['group'] for r in rows}):aggregate[g]={n:counts(**{k:sum(r['localization'][k] for r in rows if r['group']==g and r['candidate']==n) for k in ('tp','fp','fn')}) for n in ('initial','epoch-4','epoch-8')}
 (ROOT/'evaluation/v12/SYMBOL-SCREEN.json').write_text(json.dumps({'role':'Development only, all previous development strata reused; White Rose training not selection and no new-project source opened','rows':rows,'aggregate':aggregate},indent=2)+'\n')
nn.model=original
