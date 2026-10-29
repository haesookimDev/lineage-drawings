"""Development and final-only labels have separate access paths."""
import json,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_gt
from evaluation.v9.data import validation

def development():
    for group,ident,region,high,gt,truth in validation():
        path=ROOT/'.benchmark-data'/group/(ident+'.png')
        if not path.exists():path=ROOT/'.benchmark-data/v8r2-pages'/group/(ident+'-4800.png')
        yield group,ident,region,path,gt,truth
    for ident in range(6,12):
        path=ROOT/'.benchmark-data/open100'/f'{ident}.png'
        with Image.open(path) as im:gt,truth,*_=load_gt(path.with_suffix('.graphml'),im.size)
        yield 'open100-repair',str(ident),None,path,gt,truth
    for region in json.loads((ROOT/'evaluation/v9/new-project-labels.json').read_text())['sheets'].values():
        yield region['group']+'-preservation',region['id'],region,ROOT/'.benchmark-data/v9-held'/region['group']/(region['id']+'-4800.png'),region['nodes'],set(tuple(e) for e in region['edges'])

def images(path):
    with Image.open(path) as raw:
        high=raw.convert('RGB');viewer=high.copy()
        high.thumbnail((4800,4800));viewer.thumbnail((2400,2400))
    return viewer,high
