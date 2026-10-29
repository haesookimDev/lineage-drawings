"""Reviewed source-coordinate annotations. No v8 prediction used for new test."""
from pathlib import Path
from itertools import combinations
import json,hashlib,copy
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
old=json.loads((ROOT/'evaluation/v5/real-labels.json').read_text())['sheets'];manual={}
# Former v7 test cases are now explicit development training examples.
for key,s in json.loads((ROOT/'evaluation/v7/new-page-labels.json').read_text())['sheets'].items():
 s=copy.deepcopy(s);s['role']='training';manual[key]=s
ports={
 'virden-p5':{'bav':[(55,34),(75,34)],'snv':[(149,10),(149,45)],'lit':[(102,100)],'le':[(102,104),(102,137)],'hs':[(219,78)],'kgv':[(267,153),(284,153)]},
 'ruttan-p50':{'hs':[(36,49)],'pump':[(59,67),(176,67),(135,51)],'v1':[(135,18),(135,35)],'v2':[(201,67),(221,67)]},
 'virden-p7':{'a1':[(35,77),(52,77)],'p1':[(99,76),(147,76)],'pi1':[(211,44)],'c1':[(257,74),(272,74)],'a2':[(313,74),(330,74)],'b1':[(35,236),(53,236)],'p2':[(100,235),(148,235)],'pi2':[(211,205)],'c2':[(257,232),(273,232)],'b2':[(314,233),(331,233)]},
 'ruttan-p62':{'v1':[(17,17),(30,17)],'v2':[(29,43),(29,65)],'sample':[(119,80),(119,104)],'ae':[(208,120),(208,159)],'fe':[(259,176)],'fit':[(238,198)]},
 'pahala-p20':{'in':[(161,89)],'pump':[(404,89),(471,89),(437,72)],'tsh':[(454,41)],'v1':[(542,59),(560,59)],'v2':[(609,59),(630,59)],'out':[(1488,59)]}
}
for key,pp in ports.items():
 s=copy.deepcopy(old[key]);w,h=s['canvas'];a,b=[round(v*[w,h][i]) for i,v in enumerate(s['roi'][:2])];s['nodes']=[n for n in s['nodes'] if n['id'] in pp];s['ports']=[{'element_id':ident,'point':[(a+x)/w,(b+y)/h]} for ident,points in pp.items() for x,y in points];s['junctions']=[];s['point_samples']=[];s['role']='validation' if key in ('virden-p7','ruttan-p62','pahala-p20') else 'training';manual[key]=s
 if key=='virden-p7':s['junctions']=[{'point':[(a+166)/w,(b+y)/h],'type':'tee'} for y in (74,233)]
 if key=='ruttan-p62':s['junctions']=[{'point':[(a+119)/w,(b+110)/h],'type':'tee'}]
# Explicit hard negatives inspected in v7 error audit; not implicit blanket labels.
for row in json.loads((ROOT/'evaluation/v7/POINT-ERRORS.json').read_text())['rows']:
 if row['kind']=='junctions':manual[row['id']].setdefault('point_samples',[]).extend({'point':p['point'],'type':'none','review':'visible frame corner / bend or symbol contour, reviewed from source'} for p in row['false_positives'])
# 64 topology-node source crops were visually inspected; ambiguous taps excluded.
reviewed=[]
for row in json.loads((ROOT/'evaluation/v8/graph-point-draft.json').read_text()):
 if row['index'] in (3,4,5,7):continue
 reviewed.append({**row,'type':'tee' if row['degree']==3 else 'none','review':'source crop visually reviewed; degree2 corners excluded from junction display; ambiguous samples3,4,5,7 skipped'})
(ROOT/'evaluation/v8/reviewed-graph-points.json').write_text(json.dumps(reviewed,indent=2)+'\n')
(ROOT/'evaluation/v8/manual-development.json').write_text(json.dumps({'annotation':'Source-visible ports of enumerated symbols, no implied latent physical flow. Selected symbol list is complete for port training; other symbols omitted from candidate generation. Single annotator, no engineering certification. Junction point_samples are explicit reviewed negatives, not blanket unknown-as-negative. Former v7 tests are now training.','sheets':manual},indent=2)+'\n')
new={}
def add(key,roi,nodes,nets,pp,jj):
 group,ident=key.split('-');im=Image.open(ROOT/'.benchmark-data/v8-pages'/group/(ident+'-2400.png'));w,h=im.size
 norm=lambda p:[p[0]/w,p[1]/h]
 new[key]={'group':group,'id':ident,'role':'new-page-test','canvas':[w,h],'roi':[v/[w,h][i%2] for i,v in enumerate(roi)],'ignore':[],'nodes':[{'id':k,'kind':kind,'bbox':[v/[w,h][i%2] for i,v in enumerate(box)]} for k,kind,box in nodes],'ports':[{'element_id':k,'point':norm(p)} for k,p in pp],'junctions':[{'point':norm(p),'type':typ} for p,typ in jj],'edges':sorted({tuple(sorted(e)) for net in nets for e in combinations(net,2)}),'image_sha256':hashlib.sha256((ROOT/'.benchmark-data/v8-pages'/group/(ident+'-2400.png')).read_bytes()).hexdigest()}
nodes=[];pp=[];nets=[];jj=[]
for tag,x in [('a',707),('b',958)]:
 for suffix,box in [('pi',[x-16,1103,x+16,1137]),('psh',[x+22,1103,x+55,1137]),('bav',[x-7,1169,x+7,1183])]:nodes.append((tag+suffix,'instrumentation' if suffix!='bav' else 'valve',box))
 pp += [(tag+'pi',[x,1137]),(tag+'psh',[x+39,1137]),(tag+'bav',[x,1169]),(tag+'bav',[x,1183])];nets.append([tag+s for s in ('pi','psh','bav')]);jj.append(([x,1156],'tee'))
add('virden-p10',[680,1096,1025,1195],nodes,nets,pp,jj)
nodes=[];pp=[];nets=[]
for tag,xs,yt,yb,xlsh in [('a',[568,616,665],855,898,522),('b',[1170,1227,1276],853,896,1124)]:
 for j,x in enumerate(xs):
  top=tag+str(j)+'top';bot=tag+str(j)+'bottom';nodes += [(top,'instrumentation',[x-17,yt-17,x+17,yt+17]),(bot,'instrumentation',[x-17,yb-17,x+17,yb+17])];pp += [(top,[x,yt+17]),(bot,[x,yb-17]),(bot,[x,yb+17])];nets.append([top,bot])
 nodes.append((tag+'lsh','instrumentation',[xlsh-17,yb-17,xlsh+17,yb+17]));pp.append((tag+'lsh',[xlsh,yb+17]))
add('virden-p12',[500,825,1320,920],nodes,nets,pp,[])
xs=[351,402,453,504,555,606,657,708,758,809,860];nodes=[(f'i{k}','instrumentation',[x-17,592,x+17,627]) for k,x in enumerate(xs)];pp=[(f'i{k}',[x,592]) for k,x in enumerate(xs)];jj=[([x,583],'dot') for x in xs[:-1]]
add('pahala-p18',[330,575,882,635],nodes,[[n[0] for n in nodes]],pp,jj)
(ROOT/'evaluation/v8/new-page-labels.json').write_text(json.dumps({'annotation':'Single source annotator, geometry-assisted circle centers visually reviewed; not detector predictions. Visible shared instrument bus adjacency, no semantic flow approval. New pages from known projects; ROIs selected without any v8 inference. Junction labels distinguish actual black dots from T and two-arm bends. Very small selected regions, not full-sheet GT or independent-project test.','sheets':new},indent=2)+'\n')
for key,s in {**manual,**new}.items():
 p=ROOT/'.benchmark-data/v8-pages'/s['group']/(s['id']+'-2400.png') if key in new else ROOT/'.benchmark-data'/s['group']/(s['id']+'.png');im=Image.open(p).convert('RGB');w,h=im.size;d=ImageDraw.Draw(im)
 for n in s['nodes']:
  box=[v*[w,h][i%2] for i,v in enumerate(n['bbox'])];d.rectangle(box,outline='red');d.text((box[0],box[1]-10),n['id'],fill='red')
 for p in s['ports']:
  x,y=p['point'][0]*w,p['point'][1]*h;d.ellipse((x-3,y-3,x+3,y+3),outline='blue')
 for p in s['junctions']:
  x,y=p['point'][0]*w,p['point'][1]*h;d.rectangle((x-3,y-3,x+3,y+3),outline='green')
 box=[round(v*[w,h][i%2]) for i,v in enumerate(s['roi'])];im.crop(box).save('/tmp/v8-labels-'+key+'.png')
print('MANUAL',[(k,s['role'],len(s['ports']),len(s['junctions'])) for k,s in manual.items()]);print('NEW',[(k,len(s['nodes']),len(s['edges']),len(s['ports']),len(s['junctions'])) for k,s in new.items()])
