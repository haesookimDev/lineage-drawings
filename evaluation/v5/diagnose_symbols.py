"""Post-freeze error audit: labels enter diagnostics only, never extraction."""
import sys,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend.learned_symbols import proposals,features,probabilities,model,prepare_gray
from evaluation.v5.train import overlaps
labels=json.loads((ROOT/'evaluation/v5/real-labels.json').read_text())['sheets'];classifier,meta=model();result=[]
for key,region in labels.items():
 if region['group'] not in ('virden','ruttan'):continue
 path=ROOT/'.benchmark-data'/region['group']/(region['id']+'.png');im=Image.open(path).convert('RGB');gray=prepare_gray(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2GRAY));h,w=gray.shape
 pred=json.loads((ROOT/'evaluation/v5/results'/f"{region['group']}-{region['id']}-v4-predictions.json").read_text())['elements'];base=[e['bbox'] for e in pred];boxes=proposals(gray,base,include_large=True);scores=probabilities(classifier,features(gray,boxes));truth=[tuple(round(v*[w,h][i%2]) for i,v in enumerate(n['bbox'])) for n in region['nodes']];iou=overlaps(boxes,truth);exact=probabilities(classifier,features(gray,truth));rows=[]
 for j,n in enumerate(region['nodes']):
  eligible=np.flatnonzero(iou[:,j]>=.5);best=int(iou[:,j].argmax());rows.append({'id':n['id'],'kind':n['kind'],'bbox_px':truth[j],'best_proposal_iou':float(iou[best,j]),'exact_box_objectness':float(exact[j]),'best_eligible_score':float(scores[eligible].max()) if len(eligible) else None})
 result.append({'key':key,'proposals':len(boxes),'nodes':rows})
print(json.dumps(result,indent=2));(ROOT/'evaluation/v5/post-freeze-symbol-audit.json').write_text(json.dumps(result,indent=2)+'\n')
