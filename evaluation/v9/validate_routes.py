"""Small staged routing grid, evaluated only on predeclared development pages."""
import json,sys,statistics,copy,os
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v9.data import validation
from evaluation.benchmark import score,counts
from evaluation.v5.evaluate import targets
from backend import sparse_topology as st
from backend.line_recovery import short_links
from backend.geometry import _trace_validated,combine_context_pairs
from backend.topology_context import model,prepare,features,votes
cv2.setNumThreads(2);cache=ROOT/'.benchmark-data/v9-validation-scale';svpath=ROOT/'evaluation/v9/SYMBOL-SCALE-VALIDATION.json';sv=json.loads((svpath if svpath.exists() else ROOT/'evaluation/v9/SYMBOL-VALIDATION.json').read_text());assert all(k.startswith('scale') for k in sv['shortlist']);keys=list(dict.fromkeys(['base/0/0.6','4/6/0.45']+sv['shortlist']));meta={**json.loads((ROOT/'.benchmark-data/extractor-snapshots/c9cef5afc6363e30c86e9ab4a5fe17dd68313e11/backend/assets/topology-context.json').read_text()),'line_recovery':False,'short_dash_recovery':False}
configs={'v8':{},'recover-conservative':{'line_recovery':True},'recover-balanced':{'line_recovery':True,'port_threshold':.5,'junction_threshold':.35,'junction_display_threshold':.35,'extra_edge_threshold':.65,'extra_symbol_threshold':.45,'extra_max_points':8},'recover-recall':{'line_recovery':True,'port_threshold':.45,'junction_threshold':.35,'junction_display_threshold':.35,'extra_edge_threshold':.5,'extra_symbol_threshold':.45,'extra_max_points':8},'sparse-balanced':{'line_recovery':True,'port_threshold':.5,'junction_threshold':.35,'junction_display_threshold':.35,'extra_edge_threshold':.65,'extra_symbol_threshold':.45,'extra_max_points':8,'routing_mode':'sparse'}};configs['recover-short']={**configs['recover-recall'],'short_dash_recovery':True};rows=[]
for group,ident,region,high,gt,truth in validation():
 im=high.copy();im.thumbnail((2400,2400));w,h=im.size;d=json.loads((cache/f'{group}-{ident}-base.json').read_text());bs=[e for e in d['elements'] if e['kind']!='text'];texts=[e for e in d['elements'] if e['kind']=='text'];profile=bs[0].get('properties',{}).get('profile','generic_fallback') if bs else 'generic_fallback';gray,ink=prepare(im)
 basepred,baseedge=targets(d['elements'],d['edges'],region);b=score(basepred,baseedge,gt,truth,.5);b.pop('matched_ids');rows.append({'group':group,'id':ident,'configuration':'production','scores':b})
 for key in keys:
  name,detail,cut=key.split('/');found=json.loads((cache/f'{group}-{ident}-{name}-d{detail}.json').read_text())['found'];symbols=[{'id':str(i),'kind':'unknown','bbox':box,'source':'neural_symbol_candidate','properties':{**props,'profile':profile}} for i,(box,props) in enumerate(found) if props['detector_score']>=float(cut)];els=texts[:500-len(symbols)]+symbols;basepairs=_trace_validated(els,d['segments'],w,h,im);recognized={}
  for recover,short in ((False,False),(True,False),(True,True)):
   lines,dashed=st.segments(im,ink,d['segments'],recover=recover);
   if short:
    extra=short_links(els,ink,w,h);lines=st.merge_segments(lines+extra);dashed+=len(extra)
   raw,intervals=st.proposals(els,lines,w,h);junctions=st.intersections(lines,intervals,ink,recover=recover);pm,jm,_=model();recognized[recover,short]={'gray':gray,'ink':ink,'lines':lines,'dashed':dashed,'raw':raw,'intervals':intervals,'junctions':junctions,'port_scores':votes(pm,features(gray,ink,raw),2)[:,1],'junction_scores':votes(jm,features(gray,ink,junctions),4)}
  for name,cfg in configs.items():
   cfg={**meta,**cfg};_,rec,_=st.route(els,d['segments'],w,h,im,config=cfg,recognized=recognized[bool(cfg.get('line_recovery')),bool(cfg.get('short_dash_recovery'))]);pairs,comp=combine_context_pairs(els,basepairs,rec['paths'],cfg);pp,ee=targets(els,[{'from_id':a,'to_id':b} for a,b in pairs],region);s=score(pp,ee,gt,truth,.5);s.pop('matched_ids');rows.append({'group':group,'id':ident,'configuration':key+'/'+name,'scores':s})
 print(group,ident,'routes done',flush=True)
agg={}
for key in sorted({r['configuration'] for r in rows}):
 groups={g:{m:counts(**{k:sum(r['scores'][m][k] for r in rows if r['configuration']==key and r['group']==g) for k in ('tp','fp','fn')}) for m in ('localization','connections')} for g in sorted({r['group'] for r in rows})};agg[key]={'groups':groups,'objective':statistics.mean(d['connections']['f1']+.25*d['localization']['f1'] for d in groups.values())}
chosen=max(agg,key=lambda k:(agg[k]['objective'],k=='production',k));out={'policy':'Equal-source macro(connectionF1+.25symbolF1), baseline eligible, fixed staged grid; no unseen-project inference/labels used. This is development validation, not final test.','rows':rows,'aggregate':agg,'configs':configs,'selected':chosen,'baseline':'production'};(ROOT/'evaluation/v9/ROUTE-VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print('SELECTED',chosen,agg[chosen],flush=True)
