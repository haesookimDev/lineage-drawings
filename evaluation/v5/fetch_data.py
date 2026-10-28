#!/usr/bin/env python3
"""Fetch hash-pinned v5 source files; source drawings stay outside Git."""
from pathlib import Path
import hashlib,json,sys,zipfile
import fitz,httpx
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.fetch_public_benchmark import RemoteZip
H=Path(__file__).parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 sources=json.loads((H/'real-sources.json').read_text())
 plan=json.loads((ROOT/'evaluation/v4/PLAN.json').read_text())
 sources['pahala']={'url':plan['new_real_source'],'pdf_sha256':plan['new_real_source_sha256'],'pages':[20,21,23,24]}
 for group,meta in sources.items():
  if not isinstance(meta,dict):continue
  out=ROOT/'.benchmark-data'/group;out.mkdir(parents=True,exist_ok=True);pdf=out/'source.pdf'
  if not pdf.exists():
   r=httpx.get(meta['url'],follow_redirects=True,timeout=120);r.raise_for_status();pdf.write_bytes(r.content)
  assert digest(pdf)==meta['pdf_sha256'],group
  with fitz.open(pdf) as document:
   for n in meta['pages']:
    page=document[n-1];scale=2400/max(page.rect.width,page.rect.height);path=out/f'p{n}.png';page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(path)
    expected=next((p['png_sha256'] for p in meta.get('pages_metadata',[]) if p['page_one_based']==n),None)
    if expected:assert digest(path)==expected,(group,n)
  print(group,'pinned source and pages ready',flush=True)
 files=json.loads((H/'synthetic-data.json').read_text())['files'];out=ROOT/'.benchmark-data/v5-synthetic';out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(RemoteZip()) as archive:
  for f in files:
   path=out/f['file']
   if not path.exists():path.write_bytes(archive.read(f['member']))
   assert digest(path)==f['sha256'],f['file']
 print('Pinned v5 data verified. Fetch OPEN100 and historical synthetic data with existing scripts.')
if __name__=='__main__':main()
