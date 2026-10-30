import io
import fitz
from PIL import Image,ImageDraw
from backend.extract import extract_pdf

def test_native_pdf_ocr_only_raster_region_and_keeps_distinct_repeated_tags(tmp_path,monkeypatch):
    im=Image.new('RGB',(100,100),'white');ImageDraw.Draw(im).text((10,10),'P-101',fill='black');buf=io.BytesIO();im.save(buf,format='PNG');d=fitz.open();p=d.new_page(width=400,height=300);p.insert_text((20,30),'P-101');p.insert_image(fitz.Rect(200,100,300,200),stream=buf.getvalue());source=tmp_path/'source.pdf';d.save(source);out=tmp_path/'out';out.mkdir();calls=[]
    monkeypatch.setattr('backend.neural_symbols.model',lambda:(None,None))
    def ocr(image,warnings,*args):
        calls.append(image.size);return [{'id':'ocr-tag','label':'P-101','kind':'text','bbox':[.1,.1,.2,.2],'source':'ocr','status':'needs_review','concept_id':None,'properties':{}}]
    monkeypatch.setattr('backend.extract.ocr',ocr);pages,warnings=extract_pdf(source,out,[]),[]
    page=pages[0];assert calls and calls[0][0]<page['width']*.4 and calls[0][1]<page['height']*.5
    tags=[e for e in page['elements'] if e['label']=='P-101'];assert len(tags)==2
    native=next(e for e in tags if e['source']=='pdf_text');raster=next(e for e in tags if e['source']=='ocr');assert native['bbox'][0]<.1 and raster['bbox'][0]>.5
    assert page['transform']['analysis_timing']['ocr_mode']=='source_regions'

def test_small_logo_does_not_trigger_full_page_ocr(tmp_path,monkeypatch):
    im=Image.new('RGB',(10,10),'black');b=io.BytesIO();im.save(b,format='PNG');d=fitz.open();p=d.new_page(width=400,height=300);p.insert_text((20,30),'NATIVE TEXT');p.insert_image(fitz.Rect(350,10,365,25),stream=b.getvalue());source=tmp_path/'logo.pdf';d.save(source);out=tmp_path/'out';out.mkdir()
    monkeypatch.setattr('backend.neural_symbols.model',lambda:(None,None));monkeypatch.setattr('backend.extract.ocr',lambda *args:(_ for _ in ()).throw(AssertionError('Logo must not trigger OCR')))
    page=extract_pdf(source,out,[])[0];assert any(e['source']=='pdf_text' for e in page['elements'])
