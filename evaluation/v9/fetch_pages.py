"""Fetch original public PDFs and reproduce pinned2400/4800 renders locally."""
import json,hashlib,sys
from pathlib import Path
import fitz,httpx
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if __name__=='__main__':
 rows=json.loads((ROOT/'evaluation/v9/NEW-SOURCES.json').read_text());pages=json.loads((ROOT/'evaluation/v9/SOURCE-PAGES.json').read_text())
 with httpx.Client(follow_redirects=True,timeout=90) as c:
  for s in rows:
   p=ROOT/'.benchmark-data/v9-held'/s['project']/'source.pdf';p.parent.mkdir(parents=True,exist_ok=True)
   if not p.exists():
    with c.stream('GET',s['url']) as r:
     r.raise_for_status();n=0
     with p.with_suffix('.part').open('wb') as f:
      for b in r.iter_bytes():
       n+=len(b);assert n<=100_000_000;f.write(b)
    p.with_suffix('.part').replace(p)
   assert sha(p)==s['sha256'],s['project']
   with fitz.open(p) as doc:
    for row in pages:
     if row['project']!=s['project']:continue
     page=doc[row['page']-1];scale=row['side']/max(page.rect.width,page.rect.height);out=p.parent/f"p{row['page']}-{row['side']}.png";page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(out);assert sha(out)==row['sha256'],str(out)
 print('Verified4 public pages from2 collections at two resolutions; no source files are redistributed')
