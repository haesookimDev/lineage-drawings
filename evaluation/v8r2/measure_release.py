"""Final full-pipeline paired measurements; graph-test role is post-audit regression."""
import hashlib,json,statistics,sys,time
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_extractor,score
from evaluation.v5.evaluate import canonical,targets
from evaluation.v7.evaluate import recognition_metrics
freeze=json.loads((ROOT/'evaluation/v8r2/FROZEN-RELEASE.json').read_text())
for p,h in freeze['sources'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
exs={k:load_extractor(v)[0] for k,v in [('v7',freeze['baseline']),('release','working')]};rows=[]
for ident,s in json.loads((ROOT/'evaluation/v8r2/new-page-labels.json').read_text())['sheets'].items():
 with Image.open(ROOT/'.benchmark-data/v8r2-pages'/s['group']/(s['id']+'-4800.png')) as raw:im=raw.convert('RGB');high=im.copy();high.thumbnail((4800,4800));im.thumbnail((2400,2400))
 reference=json.loads((ROOT/'evaluation/v8r2/guard-results'/f'new-page-{ident}-release-predictions.json').read_text());ref=canonical(reference['elements'],reference['connections']);first={};evidence={};times={k:[] for k in exs};row={'id':ident,'variants':{}}
 for run in range(3):
  for k in (list(exs) if run%2==0 else list(reversed(exs))):
   ex=exs[k];rec={};t=time.perf_counter();els,segs=ex.raster_candidates(im,[],high_resolution=high);els=els[:ex.MAX_ELEMENTS];edges=ex.infer_connections(els,segs,*im.size,image=im,diagnostics=rec);times[k].append(time.perf_counter()-t);rep=canonical(els,edges);sym=[e for e in els if e['kind']!='text'];ev={'ports':[e.get('properties',{}).get('ports',[]) for e in sym],'junctions':rec['junctions']};assert any(e['kind']=='text' for e in els)
   if k in first:assert first[k]==rep and evidence[k]==ev,(ident,k,'determinism')
   else:
    first[k]=rep;evidence[k]=ev;pp,ee=targets(els,edges,s);sc=score(pp,ee,s['nodes'],set(tuple(e) for e in s['edges']),.5);mapping=sc.pop('matched_ids');row['variants'][k]={'scores':sc,'recognition':recognition_metrics(els,rec,s,im.size,mapping),'deterministic':True}
    if k=='release':
     assert rep==ref,(ident,'graph replay differs from full release pipeline')
     expected_ev={'ports':[e.get('properties',{}).get('ports',[]) for e in reference['elements'] if e['kind']!='text'],'junctions':reference['recognition']['junctions']}
     if ev!=expected_ev:
      Path('/tmp/lineage-v8-point-diff.json').write_text(json.dumps({'actual':ev,'expected':expected_ev},indent=2))
      raise AssertionError((ident,'point outputs changed; diagnostic saved'))
 assert first['v7']['boxes']==first['release']['boxes'] and set(first['v7']['edges'])<=set(first['release']['edges'])
 for k in exs:row['variants'][k]['timing']={'seconds':times[k],'median':statistics.median(times[k])}
 rows.append(row);print(ident,{k:d['timing'] for k,d in row['variants'].items()},flush=True)
out={'scope':'18 actual full inference runs, final release and immutable production v7,3 repeats each on3 new pages. Includes OCR, same4800px symbol detector and2400px line/recognition; excludes decode/PDF/network/save. CPU OpenCV2 threads/OMP1. Sequential alternating variant order. Source known projects; graph comparisons are post-audit regression, point models/outputs unchanged from original R2 blind point test.','release_freeze_sha256':hashlib.sha256((ROOT/'evaluation/v8r2/FROZEN-RELEASE.json').read_bytes()).hexdigest(),'sheets':rows,'mean_sheet_median_seconds':{k:statistics.mean(r['variants'][k]['timing']['median'] for r in rows) for k in exs}}
(ROOT/'evaluation/v8r2/RELEASE-TIMING.json').write_text(json.dumps(out,indent=2)+'\n');print(out['mean_sheet_median_seconds'])
