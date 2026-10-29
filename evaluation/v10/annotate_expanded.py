"""Source-only extension of v9 labels; never read detector output here.

The same pages have prior whole-image inference exposure. Only the extension
labels are held out from development. This is not a new-project blind test.
"""
import copy,hashlib,json
from itertools import combinations
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
old=json.loads((ROOT/'evaluation/v9/new-project-labels.json').read_text())
sheets=copy.deepcopy(old['sheets'])
clique=lambda *ids:list(combinations(ids,2))

def extend(key,roi,nodes,edges,ignore=(),dashed=(),branch=()):
    s=sheets[key];w,h=s['canvas'];norm=lambda b:[v/[w,h][i%2] for i,v in enumerate(b)]
    s['old_roi']=s['roi'];s['old_node_ids']=[n['id'] for n in s['nodes']]
    s['old_edges']=copy.deepcopy(s['edges']);s['roi']=norm(roi)
    s['role']='expanded-label-holdout-untrained-page-with-prior-image-exposure'
    s['nodes'] += [{'id':ident,'kind':kind,'bbox':norm(box)} for ident,kind,box in nodes]
    s['ignore'] += [norm(b) for b in ignore]
    s['edges']=sorted(set(tuple(e) for e in s['edges'])|{tuple(sorted(e)) for e in edges})
    for kind,values in [('dashed',dashed),('branch',branch)]:s['edge_subsets'][kind]=sorted(set(tuple(e) for e in s['edge_subsets'][kind])|{tuple(sorted(e)) for e in values})
    ids={n['id'] for n in s['nodes']};assert len(ids)==len(s['nodes'])
    assert all(set(e)<=ids and len(set(e))==2 for e in s['edges'])
    assert {tuple(e) for e in s['old_edges']}=={tuple(e) for e in s['edges'] if set(e)<=set(s['old_node_ids'])}
    source=ROOT/'.benchmark-data/v9-held'/s['group']/(s['id']+'-2400.png')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==s['image_sha256']
    im=Image.open(source).convert('RGB');draw=ImageDraw.Draw(im)
    for n in s['nodes']:
        box=[v*[w,h][i%2] for i,v in enumerate(n['bbox'])]
        draw.rectangle(box,outline='red',width=1);draw.text((box[0],box[1]-11),n['id'],fill='red')
    for b in s['ignore']:draw.rectangle([v*[w,h][i%2] for i,v in enumerate(b)],outline='blue')
    draw.rectangle(roi,outline='green',width=2)
    im.save(ROOT/'.benchmark-data/v10'/f'{key}-expanded-label-overlay.png')

extend('winkler-p142',[80,300,2070,1150],[
    ('water_in','inlet/outlet',[111,365,209,383]),
    ('water_valve','valve',[478,366,495,383]),
    ('hauled_out','inlet/outlet',[1964,519,2064,537]),
    ('drain_out','inlet/outlet',[1964,1107,2064,1125]),
], [('water_in','water_valve'),('meter','hauled_out'),('drain','drain_out')],
ignore=[[300,495,520,585],[520,345,560,385]])

train1_bus=clique('ae1','ae2','v1','mixed1','ferric1','waste_v1')
train2_bus=clique('ae3','ae4','v4','mixed2','ferric2','waste_v2')
train2_mid=clique('v4','v5','drain','tap2')
effluent=[('clarifier1','collection'),('clarifier2','collection')]
# The two effluent pipes enter the collection tank separately. There is no
# direct clarifier-to-clarifier adjacency through this intermediate tank.
extend('winkler-p150',[80,100,2070,1470],[
    ('water','inlet/outlet',[111,651,209,670]),
    ('ferric1','inlet/outlet',[111,765,209,784]),
    ('mixed1','inlet/outlet',[111,973,209,992]),
    ('ferric2','inlet/outlet',[111,1057,209,1076]),
    ('mixed2','inlet/outlet',[111,1268,209,1287]),
    ('waste_out1','inlet/outlet',[1964,137,2064,156]),
    ('scum_out1','inlet/outlet',[1964,386,2064,405]),
    ('ras_out1','inlet/outlet',[1964,545,2064,564]),
    ('effluent_out','inlet/outlet',[1964,732,2064,751]),
    ('scum_out2','inlet/outlet',[1964,970,2064,989]),
    ('ras_out2','inlet/outlet',[1964,1268,2064,1287]),
    ('waste_out2','inlet/outlet',[1964,1435,2064,1454]),
    ('m1','instrumentation',[1005,276,1029,300]),
    ('m2','instrumentation',[1254,858,1278,882]),
    ('clarifier1','tank',[816,307,1218,485]),
    ('clarifier2','tank',[1066,891,1468,1072]),
    ('collection','tank',[1669,687,1726,788]),
    ('spray1','inlet/outlet',[1179,272,1190,292]),
    ('spray2','inlet/outlet',[1428,859,1440,879]),
    ('waste_v1','valve',[757,139,776,154]),
    ('waste_v2','valve',[757,1437,776,1452]),
    ('scum_v1','valve',[1714,388,1733,403]),
    ('scum_v2','valve',[1714,972,1733,987]),
    ('v4','valve',[756,1269,780,1286]),
    ('v5','valve',[884,1269,908,1286]),
    ('tap2','valve',[851,1345,867,1363]),
    ('sample2','inlet/outlet',[845,1390,873,1413]),
    ('ait3','instrumentation',[252,1145,286,1179]),
    ('ae3','instrumentation',[252,1220,286,1254]),
    ('ait4','instrumentation',[371,1147,405,1181]),
    ('ae4','instrumentation',[371,1222,405,1256]),
    ('ait_eff','instrumentation',[1752,611,1786,645]),
    ('ae_eff','instrumentation',[1752,687,1786,721]),
],train1_bus+train2_bus+train2_mid+effluent+clique('water','spray1','spray2')+
clique('collection','ae_eff','effluent_out')+[
    ('waste_v1','waste_out1'),('waste_v2','waste_out2'),
    ('ait3','ae3'),('ait4','ae4'),('ait_eff','ae_eff'),
    ('tap2','sample2'),('v2','clarifier1'),('v5','clarifier2'),
    ('m1','clarifier1'),('m2','clarifier2'),
    ('scum_v1','clarifier1'),('scum_v1','scum_out1'),('ras_out1','clarifier1'),
    ('scum_v2','clarifier2'),('scum_v2','scum_out2'),('ras_out2','clarifier2')],
ignore=[[880,332,954,380],[1130,920,1204,973],[670,894,692,921],[670,1335,692,1362]],
dashed=[('ait3','ae3'),('ait4','ae4'),('ait_eff','ae_eff')],
branch=train1_bus+train2_bus+train2_mid+effluent+clique('water','spray1','spray2')+clique('collection','ae_eff','effluent_out'))

extend('winnipeg-p2',[1250,100,2150,1230],[
    ('fit750','instrumentation',[1363,814,1410,861]),
    ('fe750','instrumentation',[1363,950,1410,998]),
    ('v740h5','valve',[1448,962,1478,986]),
],[('p750','fit750'),('fit750','fe750'),('fe750','v740h5'),('v740h5','v710b'),('v740h5','v720b')],
ignore=[[1638,961,1661,988]],dashed=[('p750','fit750'),('fit750','fe750')],branch=[('v740h5','v710b'),('v740h5','v720b')])

extend('winnipeg-p7',[550,190,660,1110],[
    ('panel2','instrumentation',[610,755,653,797]),
    ('fit2','instrumentation',[611,898,653,940]),
    ('fe2','instrumentation',[611,1007,653,1049]),
],[('panel2','fit2'),('fit2','fe2')],dashed=[('panel2','fit2'),('fit2','fe2')])

meta={'annotation':old['annotation']+' Extension applies the same semantic classes, including tank bodies, spray endpoints, instrument bubbles and complete off-page markers. Motors separate from the tank/pump body. Clarifier effluent pipes enter the collection tank separately, with no direct adjacency across the intermediate tank; branch cliques stop at semantic equipment. Truck and hose-reel pictograms, pipe reducers and disconnected equipment-name ovals are explicitly outside this target ontology. All visible target glyphs in each stated ROI are labeled, no post-inference exclusions.',
      'exposure':'The four whole pages and original35 labels were evaluated in v9. In v10 the original35 labels are development preservation data. Expanded labels were made from source-only crops while development localization/routing on the old ROIs ran; candidate output outside old ROIs was not inspected by annotation. Extension labels are withheld from selection, but whole-image and project exposure exists. Do not call this new-project/first-ever image-blind testing. None of these pages was used for gradient training.',
      'rights':'Government hosting does not imply redistribution rights. Source/crops/overlays/full OCR stay private ignored storage. Only normalized annotation, scores and original charts go to Git.',
      'sheets':sheets}
path=ROOT/'evaluation/v10/expanded-labels.json';path.write_text(json.dumps(meta,indent=2)+'\n')
summary=[]
for k,s in sheets.items():
    old_area=(s['old_roi'][2]-s['old_roi'][0])*(s['old_roi'][3]-s['old_roi'][1]);area=(s['roi'][2]-s['roi'][0])*(s['roi'][3]-s['roi'][1])
    summary.append({'sheet':k,'symbols':len(s['nodes']),'edges':len(s['edges']),'old_symbols':len(s['old_node_ids']),'old_edges':len(s['old_edges']),'area_ratio':area/old_area,'fraction_of_page':area})
print(json.dumps(summary,indent=2))
