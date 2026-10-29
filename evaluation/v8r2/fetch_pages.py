"""Render source PDF pages and verify exact pinned raster hashes; local data only."""
import hashlib,json,sys
from pathlib import Path
import fitz
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v6.fetch_data import main as fetch_previous
if __name__=='__main__':
 fetch_previous()
 sources=json.loads((ROOT/'evaluation/v5/real-sources.json').read_text())
 for row in json.loads((ROOT/'evaluation/v8r2/SOURCE-PAGES.json').read_text()):
  g=row['group'];p=ROOT/'.benchmark-data'/g/'source.pdf';digest=sources['virden']['pdf_sha256'] if g=='virden' else '6b61266b4fc32d6bd209648d8509d9e75be2d9bc584f499d855c2d4f16b8d0af';assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
  with fitz.open(p) as doc:
   page=doc[int(row['id'][1:])-1];factor=row['side']/max(page.rect.width,page.rect.height);out=ROOT/'.benchmark-data'/row['folder']/g/(f"{row['id']}-{row['side']}.png");out.parent.mkdir(exist_ok=True,parents=True);page.get_pixmap(matrix=fitz.Matrix(factor,factor),alpha=False).save(out);assert hashlib.sha256(out.read_bytes()).hexdigest()==row['sha256'],str(out)
 print('Verified6 source pages at two resolutions')
