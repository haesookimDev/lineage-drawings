"""Paired development ablation; no new-project source or annotations read."""
import copy,json,sys,time
from pathlib import Path
from collections import defaultdict
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend import neural_symbols as nn
from backend import topology_context
from backend.extract import raster_candidates,infer_connections
from evaluation.v10.data import development,images
from evaluation.benchmark import score,counts
from evaluation.v5.evaluate import targets
cv2.setNumThreads(2);point_meta=topology_context.model()[2];original_meta=copy.deepcopy(point_meta);original_nn=nn.model;rows=[];folder=ROOT/'.benchmark-data/v12-pipeline';folder.mkdir(exist_ok=True)
variants=[('baseline','initial',0,False,0),('retrained','frozen-bn-epoch-4',0,True,0),('selective','frozen-bn-epoch-4',4,True,0),('joint025','frozen-bn-epoch-4',4,True,.25),('joint05','frozen-bn-epoch-4',4,True,.5)]
for group,ident,region,path,gt,truth in development():
 view,high=images(path)
 for name,checkpoint,detail,parallel,mix in variants:
  p=ROOT/'.benchmark-data/v12-models'/f'{checkpoint}.onnx';net=cv2.dnn.readNetFromONNX(str(p));meta={**json.loads(p.with_suffix('.json').read_text()),'threshold':.6,'detail_tiles':detail,'detail_hint_mode':'hybrid','detail_tile_size':384,'parallel_ocr':parallel};nn.model=lambda:(net,meta);point_meta.clear();point_meta.update(original_meta,joint_context=bool(mix),joint_mix=mix);warnings=[];timings={};start=time.perf_counter();els,segs=raster_candidates(view,warnings,high_resolution=high,timings=timings);diagnostics={};edges=infer_connections(els,segs,*view.size,image=view,diagnostics=diagnostics);elapsed=time.perf_counter()-start;pp,ee=targets(els,edges,region);ss=score(pp,ee,gt,truth,.5);ss.pop('matched_ids');row={'group':group,'id':ident,'variant':name,'scores':ss,'diagnostic_warm_seconds':elapsed};rows.append(row);(folder/f'{group}-{ident}-{name}.json').write_text(json.dumps({'elements':els,'connections':edges,'recognition':diagnostics,'timings':timings})+'\n');print(row,flush=True)
 aggregate={g:{v:{m:counts(**{k:sum(r['scores'][m][k] for r in rows if r['group']==g and r['variant']==v) for k in ('tp','fp','fn')}) for m in ('localization','connections')} for v,*_ in variants} for g in sorted({r['group'] for r in rows})};(ROOT/'evaluation/v12/PIPELINE-VALIDATION.json').write_text(json.dumps({'role':'Development only, 5 fixed ablations. Timing diagnostics share OCR cache and are not cold latency. No new-project input used.','variants':variants,'rows':rows,'aggregate':aggregate},indent=2)+'\n')
point_meta.clear();point_meta.update(original_meta);nn.model=original_nn
