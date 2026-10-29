from types import SimpleNamespace
from PIL import Image
from backend import extract

def test_cached_ocr_preserves_words_and_creates_new_ids_without_shared_boxes(monkeypatch):
    calls=[]
    monkeypatch.setattr(extract.shutil,'which',lambda _: '/fake/tesseract')
    def run(args,**kwargs):
        calls.append(args)
        if '--list-langs' in args:return SimpleNamespace(stdout='List\neng\nkor\n')
        return SimpleNamespace(stdout='header\n5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t95\tP-101\n')
    monkeypatch.setattr(extract.subprocess,'run',run);extract.ocr_language.cache_clear();extract.clear_ocr_cache()
    try:
        im=Image.new('RGB',(100,100),'white');d={};first=extract.ocr(im,[],d);assert d['ocr_cache_hit'] is False
        first[0]['bbox'][0]=.99
        second=extract.ocr(im,[],d);assert d['ocr_cache_hit'] is True and second[0]['id']!=first[0]['id'] and second[0]['bbox'][0]==.1
        assert second[0]['label']=='P-101' and len(calls)==2
        im.putpixel((0,0),(0,0,0));extract.ocr(im,[],d);assert d['ocr_cache_hit'] is False and len(calls)==3
        for i in range(12):
            im.putpixel((0,1),(i,0,0));extract.ocr(im,[])
        assert len(extract._OCR_CACHE)==8
    finally:extract.clear_ocr_cache();extract.ocr_language.cache_clear()
