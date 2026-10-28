"""Render the three v7 new-page sources, verifying existing source PDF hashes."""
import sys,json,hashlib
from pathlib import Path
import fitz
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v6.fetch_data import main as fetch_previous

def main():
    fetch_previous();roles={'virden':['p9','p11'],'pahala':['p19']};expected={(r['group'],r['id'],r['side']):r['sha256'] for r in json.loads((ROOT/'evaluation/v7/new-page-sources.json').read_text())}
    for group,ids in roles.items():
        path=ROOT/'.benchmark-data'/group/'source.pdf'
        if group=='virden':digest=json.loads((ROOT/'evaluation/v5/real-sources.json').read_text())['virden']['pdf_sha256']
        else:digest='6b61266b4fc32d6bd209648d8509d9e75be2d9bc584f499d855c2d4f16b8d0af'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        with fitz.open(path) as pdf:
            for ident in ids:
                page=pdf[int(ident[1:])-1]
                for side in (2400,4800):
                    out=ROOT/'.benchmark-data'/group/(ident+'.png') if side==2400 else ROOT/'.benchmark-data/v7-highres'/group/(ident+'.png');out.parent.mkdir(exist_ok=True,parents=True);factor=side/max(page.rect.width,page.rect.height);page.get_pixmap(matrix=fitz.Matrix(factor,factor),alpha=False).save(out);assert hashlib.sha256(out.read_bytes()).hexdigest()==expected[group,ident,side]
        print(group,ids,flush=True)
if __name__=='__main__':main()
