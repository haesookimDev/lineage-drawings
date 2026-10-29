"""Explicit role loader; unseen projects are not accessible from development loops."""
import json,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v6.data import sources,read
from evaluation.benchmark import load_gt

def validation():
 for group,ident,region in sources('validation'):
  high,_=read(group,ident,region);original=Image.open(ROOT/'.benchmark-data'/group/(ident+'.png')).size
  gt,truth=(region['nodes'],set(tuple(e) for e in region['edges'])) if region else load_gt(ROOT/'.benchmark-data'/group/(ident+'.graphml'),original)[:2]
  yield group,ident,region,high,gt,truth
 for key,s in json.loads((ROOT/'evaluation/v8r2/new-page-labels.json').read_text())['sheets'].items():
  path=ROOT/'.benchmark-data/v8r2-pages'/s['group']/(s['id']+'-4800.png')
  with Image.open(path) as raw:high=raw.convert('RGB');high.thumbnail((4800,4800))
  yield s['group'],s['id'],s,high,s['nodes'],set(tuple(e) for e in s['edges'])
