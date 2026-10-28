"""Manual new-page labels fixed from source pixels before neural prediction."""
import hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
S={
'virden-p6':{'roi':[1260,875,1750,1085],'ignore':[[330,0,375,123],[112,140,140,209],[337,139,365,209]],'nodes':[
 ['v1','valve',[51,125,68,142]],['psl','instrumentation',[88,82,124,116]],['sep','equipment',[142,101,206,145]],['ts','instrumentation',[158,68,194,104]],['psh','instrumentation',[207,82,242,116]],['check','valve',[244,124,262,143]],['v2','valve',[278,124,296,143]],['fe','instrumentation',[404,81,440,117]],['fit','instrumentation',[404,10,439,45]]],
 'nets':[['v1','psl','sep'],['sep','ts'],['sep','psh','check'],['check','v2'],['v2','fe'],['fe','fit']]},
'virden-p8':{'roi':[965,695,1440,860],'ignore':[[99,116,127,164],[371,116,398,164],[182,20,205,78]],'nodes':[
 ['v1','valve',[34,93,53,112]],['pump','pump',[184,77,232,122]],['pi','instrumentation',[272,6,308,42]],['pe','instrumentation',[272,48,308,84]],['check','valve',[329,91,348,110]],['v2','valve',[410,91,431,110]]],
 'nets':[['v1','pump'],['pump','pe','check'],['pe','pi'],['check','v2']]},
'pahala-p22':{'roi':[290,750,1650,1300],'ignore':[[477,34,564,82],[606,177,695,221],[744,319,839,363],[925,460,1012,503]],'nodes':[
 ['in','inlet/outlet',[27,283,108,304]],['v1','valve',[582,49,602,94]],['v2','valve',[713,192,733,235]],['v3','valve',[854,332,874,376]],['v4','valve',[1030,472,1052,517]],['out1','inlet/outlet',[1178,70,1259,91]],['out2','inlet/outlet',[1178,211,1259,234]],['out3','inlet/outlet',[1178,330,1259,351]],['out4','inlet/outlet',[1178,471,1259,492]]],
 'nets':[['in','v1','v2','v3','v4'],['v1','out1'],['v2','out2'],['v3','out3'],['v4','out4']]},
'ruttan-p51':{'roi':[1420,30,1800,430],'ignore':[],'nodes':[
 ['foot3','valve',[139,52,154,73]],['foot4','valve',[234,52,250,73]],['p3','pump',[154,239,205,310]],['p4','pump',[240,222,292,283]],['hs3','instrumentation',[28,253,65,291]],['hs4','instrumentation',[333,236,371,274]],['v3','valve',[153,312,170,335]],['v4','valve',[240,341,256,361]]],
 'nets':[['foot3','p3'],['p3','v3'],['hs3','p3'],['foot4','p4'],['p4','v4'],['hs4','p4']]}
}
for key,s in S.items():
 group,ident=key.split('-');im=Image.open(ROOT/'.benchmark-data'/group/(ident+'.png')).convert('RGB');w,h=im.size;origin=s['roi'][:2];norm=lambda b:[float(v/[w,h][i%2]) for i,v in enumerate(b)];absbox=lambda b:[v+origin[i%2] for i,v in enumerate(b)]
 s['group']=group;s['id']=ident;s['role']='new-page-known-project';s['nodes']=[{'id':i,'kind':k,'bbox':norm(absbox(b))} for i,k,b in s['nodes']];s['ignore']=[norm(absbox(b)) for b in s['ignore']];roi=s['roi'];s['roi']=norm(roi);edges=set()
 for net in s.pop('nets'):
  for a in net:
   for b in net:
    if a!=b:edges.add(tuple(sorted((a,b))))
 s['edges']=sorted(edges);s['image_sha256']=hashlib.sha256((ROOT/'.benchmark-data'/group/(ident+'.png')).read_bytes()).hexdigest();draw=ImageDraw.Draw(im)
 for n in s['nodes']:
  b=[round(v*[w,h][i%2]) for i,v in enumerate(n['bbox'])];draw.rectangle(b,outline='red',width=1);draw.text((b[0],max(0,b[1]-12)),n['id'],fill='red')
 for ig in s['ignore']:draw.rectangle([round(v*[w,h][i%2]) for i,v in enumerate(ig)],outline='blue')
 im.crop(roi).save('/tmp/v6-'+key+'-annotated.png')
labels={'annotation':'Single-annotator visual transcription from original page rasters, before neural extraction on these pages. Known projects, new pages only. Specified small ROIs; Pahala motor valve assemblies include actuator and valve body, position indicator circles ignored. Auxiliary Virden branch valves/drains/probes explicitly ignored. Not independent engineering-certified GT or whole-sheet coverage. Ruttan51 is a pumping-station sheet, distinct from training treatment-system50/62.','sheets':S}
(ROOT/'evaluation/v6/new-page-labels.json').write_text(json.dumps(labels,indent=2)+'\n');print({k:(len(s['nodes']),len(s['edges'])) for k,s in S.items()})
