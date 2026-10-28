"""Manual recognition-test ports/junctions fixed before new-page inference."""
from pathlib import Path
import hashlib,json
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sheets={
'virden-p11':{'roi':[990,565,1210,675],'nodes':[
 ['pi','instrumentation',[27,19,60,52]],['ti','instrumentation',[74,19,108,52]],['ps','instrumentation',[121,19,155,52]],['ts','instrumentation',[169,19,203,52]]],
 'nets':[['pi','ti','ps','ts']],'ports':[['pi',[43,52]],['ti',[91,52]],['ps',[138,52]],['ts',[185,52]]],'junctions':[[[43,85],'tee'],[[91,85],'tee'],[[138,85],'tee'],[[185,85],'tee']],'ignore':[]},
'virden-p9':{'roi':[1560,1010,1610,1095],'nodes':[
 ['fit','instrumentation',[7,7,43,43]],['fe','instrumentation',[7,45,43,81]]],
 'nets':[['fit','fe']],'ports':[['fit',[25,7]],['fit',[25,43]],['fe',[25,45]],['fe',[25,81]]],'junctions':[],'ignore':[]},
'pahala-p19':{'roi':[110,820,1160,1040],'nodes':[
 ['air','inlet/outlet',[29,12,183,49]],['water','inlet/outlet',[29,87,183,123]],['valve','valve',[601,158,619,177]],['lt','instrumentation',[732,152,768,189]]],
 'nets':[['water','valve']],'ports':[['air',[183,30]],['water',[183,105]],['valve',[609,158]],['valve',[609,177]],['lt',[750,152]],['lt',[750,189]]],'junctions':[[[750,30],'plain_cross']],'ignore':[[995,48,1030,90]]}
}
for key,s in sheets.items():
 g,i=key.split('-');im=Image.open(ROOT/'.benchmark-data'/g/(i+'.png')).convert('RGB');w,h=im.size;a,b=s['roi'][:2];norm=lambda q:[float(q[0]/w),float(q[1]/h)];absolute=lambda q:[q[0]+a,q[1]+b];roi=s['roi'];s['nodes']=[{'id':k,'kind':kind,'bbox':[float((v+[a,b][j%2])/[w,h][j%2]) for j,v in enumerate(box)]} for k,kind,box in s['nodes']];s['ports']=[{'element_id':k,'point':norm(absolute(q))} for k,q in s['ports']];s['junctions']=[{'point':norm(absolute(q)),'type':t} for q,t in s['junctions']];s['ignore']=[[float((v+[a,b][j%2])/[w,h][j%2]) for j,v in enumerate(box)] for box in s['ignore']];s['roi']=[float(v/[w,h][j%2]) for j,v in enumerate(roi)];edges=set()
 for net in s.pop('nets'):
  for x in net:
   for y in net:
    if x!=y:edges.add(tuple(sorted((x,y))))
 s.update(group=g,id=i,role='new-page-known-project',edges=sorted(edges),image_sha256=hashlib.sha256((ROOT/'.benchmark-data'/g/(i+'.png')).read_bytes()).hexdigest())
 draw=ImageDraw.Draw(im)
 for n in s['nodes']:
  box=[round(v*[w,h][j%2]) for j,v in enumerate(n['bbox'])];draw.rectangle(box,outline='red');draw.text((box[0],box[1]-12),n['id'],fill='red')
 for p in s['ports']:
  x,y=[round(v*[w,h][j]) for j,v in enumerate(p['point'])];draw.ellipse((x-3,y-3,x+3,y+3),outline='blue')
 for p in s['junctions']:
  x,y=[round(v*[w,h][j]) for j,v in enumerate(p['point'])];draw.rectangle((x-3,y-3,x+3,y+3),outline='green')
 im.crop(roi).save('/tmp/v7-'+key+'-labels.png')
(ROOT/'evaluation/v7/new-page-labels.json').write_text(json.dumps({'annotation':'Single-annotator visual transcription before v7 inference; very small diagnostic ROIs from known training projects. Symbols10, edges8, ports14, junctions5. Inlet marker includes tag/letter assembly. Pahala air/level crossing is visibly unmarked; visible continuous water path is one scored edge. Not independent engineering-certified or full-sheet GT. Ports evaluated at6px / junctions6px in2400 view.','sheets':sheets},indent=2)+'\n')
