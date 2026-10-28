"""Training/validation source loader; test labels are deliberately separate."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_gt

def sources(role):
 plan=json.loads((ROOT/'evaluation/v6/PLAN.json').read_text());old=json.loads((ROOT/'evaluation/v5/PLAN.json').read_text());regions=json.loads((ROOT/'evaluation/v5/real-labels.json').read_text())['sheets'];pahala=json.loads((ROOT/'evaluation/v4/pahala-labels.json').read_text())['sheets']
 for group,ids in plan[role].items():
  if group=='synthetic':group='v5-synthetic';ids=[Path(x).stem for x in old['training' if role=='training' else 'threshold_validation']['synthetic_members']]
  for ident in ids:
   region=pahala[ident] if group=='pahala' and ident in pahala else regions.get(group+'-'+ident)
   yield group,ident,region

def read(group,ident,region,highres=True):
 p=ROOT/'.benchmark-data'/group/(ident+'.png');im=Image.open(p).convert('RGB');original=im.size
 if highres and group in ('pahala','virden','ruttan'):
  p=ROOT/'.benchmark-data/v6-highres'/group/(ident+'.png');im=Image.open(p).convert('RGB')
 im.thumbnail((4800,4800));w,h=im.size
 nodes=region['nodes'] if region else load_gt(ROOT/'.benchmark-data'/group/(ident+'.graphml'),original)[0]
 boxes=np.asarray([[v*[w,h][i%2] for i,v in enumerate(n['bbox'])] for n in nodes],np.float32)
 return im,boxes
