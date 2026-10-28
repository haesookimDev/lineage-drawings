#!/usr/bin/env python3
"""Reconstruct public page rasters, hash-check PDFs, keep them outside Git."""
from pathlib import Path
import hashlib,json,sys
import fitz
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v5.fetch_data import main as fetch_v5

def main():
 fetch_v5();plan=json.loads((ROOT/'evaluation/v6/PLAN.json').read_text());labels=json.loads((ROOT/'evaluation/v6/new-page-labels.json').read_text())['sheets']
 for group in ('pahala','virden','ruttan'):
  ids=sorted(set(plan['training'].get(group,[])+plan['validation'].get(group,[])+plan['new_page_test'].get(group,[])+(['p21'] if group=='pahala' else [])));out=ROOT/'.benchmark-data/v6-highres'/group;out.mkdir(parents=True,exist_ok=True)
  with fitz.open(ROOT/'.benchmark-data'/group/'source.pdf') as doc:
   for ident in ids:
    page=doc[int(ident[1:])-1]
    for side,path in [(4800,out/(ident+'.png')),(2400,ROOT/'.benchmark-data'/group/(ident+'.png'))]:
     scale=side/max(page.rect.width,page.rect.height);page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(path)
     key=group+'-'+ident
     if side==2400 and key in labels:assert hashlib.sha256(path.read_bytes()).hexdigest()==labels[key]['image_sha256'],key
  print(group,ids,flush=True)
 print('Use existing OPEN100 and v3 confirmation fetch scripts for historical sources.')
if __name__=='__main__':main()
