#!/usr/bin/env python3
"""Fixed visual transcription. No detector predictions used in these labels."""
import hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sheets={
'pahala-p20':{'group':'pahala','id':'p20','role':'threshold-validation','canvas':[1978,1279],'roi':[380,668,1740,780],'nodes':[
 ['in','inlet/outlet',[385,725,515,758]],['pump','pump',[711,722,770,757]],['tsh','instrumentation',[737,668,771,703]],['v1','valve',[824,707,843,725]],['v2','valve',[880,706,901,726]],['out','inlet/outlet',[1604,700,1736,734]]],
 'nets':[['in','pump'],['pump','v1'],['v1','v2'],['v2','out'],['pump','tsh']]},
'pahala-p21':{'group':'pahala','id':'p21','role':'unseen-page-known-project','canvas':[1978,1279],'roi':[200,730,1550,1100],'nodes':[
 ['in1','inlet/outlet',[214,964,282,983]],['in2','inlet/outlet',[214,1043,282,1065]],['basin','equipment',[438,966,530,993]],['tank','tank',[646,795,1026,1088]],['le','instrumentation',[721,984,757,1020]],['lsh','instrumentation',[898,739,934,777]],['fit','instrumentation',[1270,869,1307,905]],['fe','instrumentation',[1270,899,1307,935]],['out','inlet/outlet',[1473,970,1544,989]]],
 'nets':[['in1','in2','basin'],['basin','tank'],['tank','le'],['tank','lsh'],['tank','fe','out'],['fe','fit']]},
'virden-p5':{'group':'virden','id':'p5','role':'unseen-project','canvas':[2048,1244],'roi':[620,770,940,945],'nodes':[
 ['bav','valve',[661,785,692,814]],['snv','valve',[733,777,765,813]],['lit','instrumentation',[690,822,724,857]],['le','instrumentation',[690,852,724,887]],['hs','instrumentation',[792,797,825,836]],['screen','equipment',[692,836,830,934]],['kgv','valve',[843,887,873,914]]],
 'nets':[['bav','snv'],['snv','screen'],['screen','kgv'],['lit','le'],['le','screen']]},
'virden-p7':{'group':'virden','id':'p7','role':'unseen-project','canvas':[2048,1244],'roi':[820,725,1110,955],'nodes':[
 ['a1','valve',[846,779,867,804]],['p1','pump',[900,765,946,815]],['pi1','instrumentation',[982,730,1018,769]],['c1','valve',[1031,776,1058,799]],['a2','valve',[1081,776,1104,800]],
 ['b1','valve',[846,913,868,939]],['p2','pump',[900,902,947,947]],['pi2','instrumentation',[981,866,1018,904]],['c2','valve',[1031,913,1058,936]],['b2','valve',[1081,913,1105,938]]],
 'nets':[['a1','p1'],['p1','pi1','c1'],['c1','a2'],['b1','p2'],['p2','pi2','c2'],['c2','b2'],['a2','b2']]},
'ruttan-p50':{'group':'ruttan','id':'p50','role':'unseen-project-revision-pair','canvas':[1978,1279],'roi':[630,420,835,505],'nodes':[
 ['hs','instrumentation',[640,429,678,468]],['pump','pump',[677,457,777,491]],['v1','valve',[724,431,751,450]],['v2','valve',[787,461,817,484]]],
 'nets':[['hs','pump'],['v1','pump'],['v2','pump']]},
'ruttan-p62':{'group':'ruttan','id':'p62','role':'unseen-project-revision-pair','canvas':[2400,1855],'roi':[710,575,1005,810],'nodes':[
 ['v1','valve',[724,581,741,600]],['v2','valve',[730,616,747,642]],['sample','valve',[819,654,838,682]],['pump','pump',[848,630,972,676]],['ae','instrumentation',[895,691,937,737]],['fe','instrumentation',[946,707,990,752]],['fit','instrumentation',[903,749,950,794]]],
 'nets':[['v1','v2','pump'],['sample','ae','fe'],['fe','fit']]}
}
# Tighten graphical extents on native-resolution crops, excluding printed tags
# outside a symbol. This visual QA precedes all test-project model inference.
refined={
'pahala-p20':{'in':[8,71,161,107],'pump':[404,72,471,105],'tsh':[436,3,471,41],'v1':[542,50,560,68],'v2':[609,50,630,68],'out':[1488,41,1642,79]},
'virden-p5':{'bav':[55,25,75,44],'snv':[139,10,159,45],'lit':[85,67,118,100],'le':[85,104,118,137],'hs':[202,44,235,78],'screen':[86,80,236,186],'kgv':[267,144,284,162]},
'virden-p7':{'a1':[35,68,52,87],'p1':[99,50,147,96],'pi1':[194,8,228,44],'c1':[257,65,272,81],'a2':[313,64,330,84],'b1':[35,227,53,246],'p2':[100,211,148,254],'pi2':[193,169,228,205],'c2':[257,223,273,241],'b2':[314,224,331,243]},
'ruttan-p50':{'hs':[18,12,55,49],'pump':[59,51,176,83],'v1':[126,18,145,35],'v2':[201,58,221,75]},
'ruttan-p62':{'v1':[17,10,30,24],'v2':[23,43,35,65],'sample':[112,80,125,104],'pump':[143,59,251,96],'ae':[189,120,227,159],'fe':[239,135,279,176],'fit':[199,181,238,216]}
}
for key,by_id in refined.items():
 s=sheets[key];im=Image.open(ROOT/'.benchmark-data'/s['group']/(s['id']+'.png'));w,h=s['canvas'];roi=[round(v*im.size[i%2]/[w,h][i%2]) for i,v in enumerate(s['roi'])]
 s['nodes']=[[ident,kind,[roi[i%2]+v for i,v in enumerate(by_id[ident])]] for ident,kind,box in s['nodes']];s['canvas']=list(im.size);s['roi']=roi
for key,s in sheets.items():
 w,h=s['canvas'];norm=lambda b:[v/[w,h][i%2] for i,v in enumerate(b)];s['roi']=norm(s['roi']);s['ignore']=[];s['nodes']=[{'id':i,'kind':k,'bbox':norm(b)} for i,k,b in s['nodes']];edges=set()
 for net in s.pop('nets'):
  for a in net:
   for b in net:
    if a!=b:edges.add(tuple(sorted((a,b))))
 s['edges']=sorted(edges);path=ROOT/'.benchmark-data'/s['group']/(s['id']+'.png');s['image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 im=Image.open(path).convert('RGB');draw=ImageDraw.Draw(im)
 for n in s['nodes']:
  box=[round(v*im.size[j%2]) for j,v in enumerate(n['bbox'])];draw.rectangle(box,outline='red',width=2);draw.text((box[0],box[1]-12),n['id'],fill='red')
 roi=[round(v*im.size[j%2]) for j,v in enumerate(s['roi'])];im.crop(roi).save('/tmp/lineage-v5-'+key+'-gt.png')
labels={'annotation':'Visual single-annotator transcription before detector inference. Fixed process ROIs on full-page raster input; class-agnostic localization and undirected first-equipment adjacency. Unlabelled electronic control buses outside ROIs excluded. No latent/hypothetical control relationship added. Not independently certified engineering GT. Ruttan p50 and p62 are revisions of the same plant, not independent projects.','sheets':sheets}
(ROOT/'evaluation/v5/real-labels.json').write_text(json.dumps(labels,indent=2)+'\n');print({k:(len(v['nodes']),len(v['edges'])) for k,v in sheets.items()})
