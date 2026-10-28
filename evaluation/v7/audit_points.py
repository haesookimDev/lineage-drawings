"""Describe fixed-test recognition errors, without revising labels or thresholds."""
import sys,json,math
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import match_boxes
from evaluation.v5.evaluate import targets,within
out=[]
for key,region in json.loads((ROOT/'evaluation/v7/new-page-labels.json').read_text())['sheets'].items():
    d=json.loads((ROOT/'evaluation/v7/results'/('new-page-'+key+'-v7-predictions.json')).read_text());im=Image.open(ROOT/'.benchmark-data'/region['group']/(region['id']+'.png')).convert('RGB');w,h=im.size;symbols,_=targets(d['elements'],d['connections'],region);mapping=match_boxes(symbols,region['nodes'],.5)
    for kind in ('ports','junctions'):
        pred=[{**p,'element_id':e['id']} for e in d['elements'] if e['kind']!='text' for p in e.get('properties',{}).get('ports',[])] if kind=='ports' else d['recognition']['junctions'];pred=[p for p in pred if within(p['point']*2,region['roi']) and not any(within(p['point']*2,b) for b in region['ignore'])];truth=region[kind];candidates=[]
        for i,p in enumerate(pred):
            for j,t in enumerate(truth):
                dist=math.hypot((p['point'][0]-t['point'][0])*w,(p['point'][1]-t['point'][1])*h)
                if dist<=6 and (mapping.get(p.get('element_id'))==t.get('element_id') if kind=='ports' else p['type']==t['type']):candidates.append((dist,i,j))
        usedp,usedt=set(),set()
        for _,i,j in sorted(candidates):
            if i not in usedp and j not in usedt:usedp.add(i);usedt.add(j)
        row={'id':key,'kind':kind,'tp':len(usedp),'false_positives':[p for i,p in enumerate(pred) if i not in usedp],'false_negatives':[t for j,t in enumerate(truth) if j not in usedt]};out.append(row)
        if kind=='ports':draw=ImageDraw.Draw(im)
        for i,p in enumerate(pred):
            x,y=p['point'][0]*w,p['point'][1]*h;draw.ellipse((x-3,y-3,x+3,y+3),outline='green' if i in usedp else 'red',width=2)
        for j,t in enumerate(truth):
            if j not in usedt:
                x,y=t['point'][0]*w,t['point'][1]*h;draw.rectangle((x-4,y-4,x+4,y+4),outline='blue',width=2)
    im.crop(tuple(round(v*[w,h][i%2]) for i,v in enumerate(region['roi']))).resize((round((region['roi'][2]-region['roi'][0])*w*2),round((region['roi'][3]-region['roi'][1])*h*2))).save('/tmp/v7-audit-'+key+'.png')
(ROOT/'evaluation/v7/POINT-ERRORS.json').write_text(json.dumps({'note':'Post-test audit only; no label or inference changes. TP=green, FP=red,FN=blue in ignored local visualization. Raw engineering crop images never committed.','rows':out},indent=2)+'\n')
print([(r['id'],r['kind'],r['tp'],len(r['false_positives']),len(r['false_negatives'])) for r in out])
