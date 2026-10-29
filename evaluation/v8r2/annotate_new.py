"""Second-round new source labels, fixed before any R2 test inference."""
from pathlib import Path
from itertools import combinations
import json,hashlib
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];out={}
def add(key,roi,nodes,edges,ports,junctions):
 g,i=key.split('-');path=ROOT/'.benchmark-data/v8r2-pages'/g/(i+'-2400.png');im=Image.open(path).convert('RGB');w,h=im.size;norm=lambda p:[p[0]/w,p[1]/h]
 s={'group':g,'id':i,'role':'new-page-test','canvas':[w,h],'roi':[v/[w,h][k%2] for k,v in enumerate(roi)],'ignore':[],'nodes':[{'id':k,'kind':kind,'bbox':[v/[w,h][j%2] for j,v in enumerate(box)]} for k,kind,box in nodes],'edges':sorted(set(tuple(sorted(e)) for e in edges)),'ports':[{'element_id':k,'point':norm(p)} for k,p in ports],'junctions':[{'point':norm(p),'type':t} for p,t in junctions],'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};out[key]=s;d=ImageDraw.Draw(im)
 for k,kind,box in nodes:d.rectangle(box,outline='red');d.text((box[0],box[1]-10),k,fill='red')
 for k,(x,y) in ports:d.ellipse((x-3,y-3,x+3,y+3),outline='blue')
 for (x,y),t in junctions:d.rectangle((x-3,y-3,x+3,y+3),outline='green')
 im.crop(roi).resize(((roi[2]-roi[0])*2,(roi[3]-roi[1])*2)).save('/tmp/v8r2-labels-'+key+'.png')
xs=[351,402,453,504,555,606,657,707,758,809,860];nodes=[(f'i{k}','instrumentation',[x-17,592,x+17,627]) for k,x in enumerate(xs)];ports=[(f'i{k}',[x,592]) for k,x in enumerate(xs)];jj=[([x,583],'dot') for x in xs[:-1]]
add('pahala-p14',[330,575,882,635],nodes,list(combinations([n[0] for n in nodes],2)),ports,jj)
add('virden-p13',[685,744,815,808],[('check','valve',[697,753,712,775]),('globe','valve',[739,752,756,775]),('drain','valve',[797,772,812,789])],[('check','globe'),('globe','drain')],[('check',[697,764]),('check',[712,764]),('globe',[739,764]),('globe',[756,764]),('drain',[804,772]),('drain',[804,789])],[([796,764],'tee'),([804,764],'tee')])
add('virden-p14',[1482,825,1590,911],[('loop','valve',[1489,846,1506,866]),('pi','instrumentation',[1511,829,1545,863]),('tap','valve',[1519,867,1537,885]),('out','valve',[1556,870,1577,903])],[('pi','tap'),('loop','tap'),('loop','out'),('tap','out')],[('loop',[1497,846]),('loop',[1497,866]),('pi',[1528,863]),('tap',[1528,867]),('tap',[1528,885]),('out',[1556,892]),('out',[1577,892])],[([1497,892],'tee'),([1528,892],'tee')])
(ROOT/'evaluation/v8r2/new-page-labels.json').write_text(json.dumps({'annotation':'New R2 pages only. One source annotator, no engineering certification. Image geometry assisted circle positions, checked visually on original before R2 inference. Known projects, selected small ROIs, not full-sheet or independent-project GT. Pahala14 bus55 edges are correlated pairs over one shared bus, not55 independent networks. Pixel port and junction tolerance6 at2400 view. Pahala17 touching dual-instrument assemblies were excluded before inference because component boundaries/adjacency were ambiguous under this protocol; no model output used in exclusion.','sheets':out},indent=2)+'\n');print([(k,len(s['nodes']),len(s['edges']),len(s['ports']),len(s['junctions'])) for k,s in out.items()])
