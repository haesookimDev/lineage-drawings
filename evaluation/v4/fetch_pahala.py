#!/usr/bin/env python3
"""Read the pinned public EPA concept design PDF; do not redistribute it."""
from pathlib import Path
import hashlib,json
import httpx,fitz
ROOT=Path(__file__).resolve().parents[2];plan=json.loads((Path(__file__).parent/'PLAN.json').read_text());out=ROOT/'.benchmark-data/pahala';out.mkdir(parents=True,exist_ok=True);p=out/'source.pdf'
if not p.exists():
    r=httpx.get(plan['new_real_source'],follow_redirects=True,timeout=120);r.raise_for_status();p.write_bytes(r.content)
assert hashlib.sha256(p.read_bytes()).hexdigest()==plan['new_real_source_sha256']
with fitz.open(p) as doc:
    for n in plan['new_real_pages_one_based']:
        page=doc[n-1];scale=2400/page.rect.width;page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(out/f'p{n}.png')
print('Pinned source and two raster pages ready; manual labels remain separate.')
