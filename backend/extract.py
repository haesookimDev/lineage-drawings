"""Deterministic extraction baselines. No semantic class or standard ID is invented."""
import math
import shutil
import subprocess
import tempfile
import uuid
import time
import hashlib
from collections import OrderedDict
from functools import lru_cache
from threading import Lock
from pathlib import Path

import ezdxf
import fitz
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ezdxf import bbox as dxf_bbox
from ezdxf.disassemble import recursive_decompose
from ezdxf.path import make_path

from backend.geometry import geometric_candidates, trace_adjacency

MAX_PAGES = 25
MAX_SIDE = 2400
MAX_ELEMENTS = 500
Image.MAX_IMAGE_PIXELS = 50_000_000
_OCR_CACHE=OrderedDict()
_OCR_CACHE_LOCK=Lock()

def clear_ocr_cache():
    with _OCR_CACHE_LOCK:_OCR_CACHE.clear()

@lru_cache(maxsize=4)
def ocr_language(binary):
    langs=subprocess.run([binary,'--list-langs'],capture_output=True,text=True,timeout=10).stdout
    return 'eng+kor' if '\nkor' in langs else 'eng'


def uid():
    return uuid.uuid4().hex


def normalized(box, w, h):
    x1, y1, x2, y2 = map(float, box)
    return [max(0., min(1., x1/w)), max(0., min(1., y1/h)),
            max(0., min(1., x2/w)), max(0., min(1., y2/h))]


def element(label, kind, box, source, properties=None):
    return dict(id=uid(), label=label[:300], kind=kind, bbox=box, source=source,
                status='needs_review', properties=properties or {}, concept_id=None)


def font(size):
    for p in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def ocr(image, warnings, diagnostics=None):
    binary = shutil.which('tesseract')
    if not binary:
        warnings.append('OCR 실행기가 없어 문자 추출을 생략했습니다. 요소는 직접 추가할 수 있습니다.')
        return []
    try:language=ocr_language(binary)
    except subprocess.SubprocessError:
        warnings.append('OCR 언어 정보를 읽지 못했습니다.');return []
    # Bounded process-local cache contains word tuples only, never image bytes,
    # credentials or review edits. Recreate IDs for every analysis/page.
    key=(binary,language,image.mode,image.size,hashlib.sha256(image.tobytes()).hexdigest())
    with _OCR_CACHE_LOCK:
        cached=_OCR_CACHE.get(key)
        if cached is not None:_OCR_CACHE.move_to_end(key)
    if cached is not None:
        if diagnostics is not None:diagnostics['ocr_cache_hit']=True
        return [element(text,'text',list(box),'ocr',{'ocr_score':score,'language':language}) for text,box,score in cached]
    if diagnostics is not None:diagnostics['ocr_cache_hit']=False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'page.png'
        image.save(path)
        try:
            result = subprocess.run([binary, str(path), 'stdout', '-l', language, '--psm', '11', 'tsv'],
                                    capture_output=True, text=True, timeout=45, check=True)
            words = []
            for line in result.stdout.splitlines()[1:]:
                v = line.split('\t', 11)
                if len(v) < 12 or not v[11].strip() or float(v[10]) < 35:
                    continue
                x,y,w,h = map(int, v[6:10])
                words.append(element(v[11], 'text', normalized((x,y,x+w,y+h), *image.size),
                                     'ocr', {'ocr_score': float(v[10]), 'language': language}))
            words=words[:MAX_ELEMENTS]
            with _OCR_CACHE_LOCK:
                _OCR_CACHE[key]=tuple((w['label'],tuple(w['bbox']),w['properties']['ocr_score']) for w in words)
                while len(_OCR_CACHE)>8:_OCR_CACHE.popitem(last=False)
            return words
        except (subprocess.SubprocessError, ValueError):
            warnings.append('문자 추출이 완료되지 않았습니다. 원본에서 직접 검토해 주세요.')
            return []


def raster_candidates(image, warnings, with_ocr=True, high_resolution=None, timings=None):
    started=time.perf_counter()
    texts = (ocr(image,warnings,timings) if timings is not None else ocr(image,warnings)) if with_ocr else []
    if timings is not None:timings['ocr_seconds']=time.perf_counter()-started
    started=time.perf_counter()
    shapes,segments,limited = geometric_candidates(image,texts,high_resolution=high_resolution)
    if timings is not None:timings['symbols_and_lines_seconds']=time.perf_counter()-started
    candidates = [element(f'요소 {i+1}','unknown',box,'neural_symbol_candidate' if properties.get('geometry')=='neural_symbol' else 'learned_symbol_candidate' if properties.get('geometry')=='learned_symbol' else 'image_candidate',properties)
                  for i,(box,properties) in enumerate(shapes)]
    if limited:
        warnings.append('이미지 요소 후보를 200개까지 표시합니다. 영역 단위 검토를 권장합니다.')
    # Do not let dense OCR text consume the entire non-text proposal budget.
    return texts[:MAX_ELEMENTS-len(candidates)]+candidates,segments


def infer_connections(elements, segments, w, h, image=None, diagnostics=None):
    details={} if diagnostics is None else diagnostics
    pairs=trace_adjacency(elements,segments,w,h,image,details)
    if any(e.get('properties',{}).get('symbol_detector') for e in elements):
        details['symbol_detector']='hog-rtrees-symbol-v1'
        details['learned_symbol_candidates']=sum(e.get('properties',{}).get('geometry')=='learned_symbol' for e in elements)
        details['symbol_model_sha256']=next((e['properties']['symbol_model_sha256'] for e in elements if e.get('properties',{}).get('symbol_model_sha256')),None)
        neural=[e for e in elements if e.get('source')=='neural_symbol_candidate']
        if neural:
            details['symbol_detector']=neural[0]['properties'].get('model','yolox-nano-symbol-v1')
            details['symbol_model_sha256']=neural[0]['properties']['symbol_model_sha256']
            details['neural_symbol_candidates']=len(neural)
            details['neural_source_size']=neural[0]['properties']['source_size']
            details['neural_tile_count']=neural[0]['properties']['tile_count']
    source='line_trace_candidate' if details.get('algorithm') in ('legacy-components-v2','visible-components-v4') else 'port_graph_candidate'
    observed={tuple(sorted((p['from_id'],p['to_id']))) for p in details.get('paths',[])};extra={tuple(sorted(p)) for p in details.get('context_extra_pairs',[])}
    return [dict(id=uid(),from_id=a,to_id=b,kind='connectedTo',source='recognized_path_candidate' if (source!='line_trace_candidate' and tuple(sorted((a,b))) in observed) or tuple(sorted((a,b))) in extra else source,status='needs_review') for a,b in pairs]


def finish(image, elements, segments, coordinate, transform, folder, number, warnings):
    pid=uid();image.save(folder/f'{pid}.png')
    if len(elements)>MAX_ELEMENTS:warnings.append('요소 수가 많아 500개까지 추출했습니다.')
    elements=elements[:MAX_ELEMENTS]
    details={};started=time.perf_counter();connections=infer_connections(elements,segments,*image.size,image=image,diagnostics=details)
    timing={**transform.get('analysis_timing',{}),'connections_seconds':time.perf_counter()-started}
    transform={**transform,'connection_inference':details,'analysis_timing':timing}
    if details.get('algorithm') in ('legacy-components-v2','visible-components-v4'):
        warnings.append('지원 형상·연속 표지가 부족해 일반 선 추적 후보를 사용했습니다. 교차의 접속 여부와 추가 연결 후보는 직접 검토해 주세요.')
    if details.get('unmarked_crossings'):
        warnings.append('접속점 없는 4방 교차는 비접속 후보로 처리했습니다. 도면의 교차 표기 관례를 확인해 주세요.')
    if details.get('pair_budget_reached') or details.get('connections_limited'):
        warnings.append('복잡한 선망의 연결 후보 상한에 도달했습니다. 영역을 나누어 검토해 주세요.')
    return dict(id=pid,number=number,width=image.width,height=image.height,coordinate_system=coordinate,
                transform=transform,elements=elements,connections=connections)


def extract_pdf(path,folder,warnings):
    pages=[]
    with fitz.open(path) as doc:
        if not doc.is_pdf or doc.needs_pass:raise ValueError('암호화되었거나 읽을 수 없는 PDF입니다.')
        if len(doc)>MAX_PAGES:raise ValueError(f'PDF는 최대 {MAX_PAGES}페이지까지 지원합니다.')
        for n,page in enumerate(doc):
            page_started=time.perf_counter();timings={}
            if page.rect.width<=0 or page.rect.height<=0:raise ValueError('PDF 페이지 크기가 유효하지 않습니다.')
            scale=min(2.,MAX_SIDE/max(page.rect.width,page.rect.height))
            pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False,colorspace=fitz.csRGB)
            image=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
            # Keep the 2400px viewer/OCR coordinates; only neural detection
            # receives a bounded higher-resolution view of the original page.
            high_resolution=None
            from .neural_symbols import model as neural_model,MAX_SOURCE_SIDE
            if neural_model()[0] is not None:
                high_scale=MAX_SOURCE_SIDE/max(page.rect.width,page.rect.height)
                high_pix=page.get_pixmap(matrix=fitz.Matrix(high_scale,high_scale),alpha=False,colorspace=fitz.csRGB)
                high_resolution=Image.frombytes('RGB',(high_pix.width,high_pix.height),high_pix.samples)
                del high_pix
            words=page.get_text('words');elements=[];segments=[]
            # Text/graphics return unrotated coordinates. Transform before normalizing.
            matrix=page.rotation_matrix
            for x1,y1,x2,y2,text,*_ in words[:MAX_ELEMENTS]:
                r=fitz.Rect(x1,y1,x2,y2)*matrix
                box=normalized((r.x0-page.rect.x0,r.y0-page.rect.y0,r.x1-page.rect.x0,r.y1-page.rect.y0),page.rect.width,page.rect.height)
                if box[2]>box[0] and box[3]>box[1]:elements.append(element(text,'text',box,'pdf_text'))
            for drawing in page.get_drawings()[:2000]:
                for item in drawing['items']:
                    if item[0]=='l':
                        a,b=item[1]*matrix,item[2]*matrix
                        segments.append([(a.x-page.rect.x0)*scale,(a.y-page.rect.y0)*scale,(b.x-page.rect.x0)*scale,(b.y-page.rect.y0)*scale])
                # Closed vector groups can be symbols, tables, or frames: leave unknown.
                r=drawing['rect']*matrix
                if drawing.get('closePath') and r.width>5 and r.height>5 and r.width<page.rect.width*.45 and r.height<page.rect.height*.45:
                    elements.append(element(f'벡터 요소 {len(elements)+1}','unknown',normalized((r.x0-page.rect.x0,r.y0-page.rect.y0,r.x1-page.rect.x0,r.y1-page.rect.y0),page.rect.width,page.rect.height),'pdf_path'))
            if not words:
                extra,lines=raster_candidates(image,warnings,high_resolution=high_resolution,timings=timings);elements+=extra
                if not segments:segments=lines
            elif page.get_images():
                extra,lines=raster_candidates(image,warnings,with_ocr=True,high_resolution=high_resolution,timings=timings)
                for e in extra:
                    if e['kind']=='text' and any(e['label']==t['label'] for t in elements if t['kind']=='text'):continue
                    elements.append(e)
                segments+=lines
                warnings.append(f'{n+1}페이지는 벡터·이미지 혼합 분석 결과입니다. 중복과 누락을 검토해 주세요.')
            elif high_resolution is not None:
                # Vector text can coexist with symbols made of open paths.
                # Use direct neural candidates without duplicating OCR/text.
                from .neural_symbols import detect as neural_detect
                for box,props in neural_detect(high_resolution):
                    elements.append(element(f'학습 요소 {len(elements)+1}','unknown',box,'neural_symbol_candidate',props))
            if high_resolution is not None:
                # Native PDF text must not consume every symbol slot.
                symbols=[e for e in elements if e['kind']!='text'][:200]
                elements=[e for e in elements if e['kind']=='text'][:MAX_ELEMENTS-len(symbols)]+symbols
            output=finish(image,elements,segments,'pdf_page',{'scale':scale,'rotation':page.rotation,'page_rect':list(page.rect),'analysis_timing':timings},folder,n+1,warnings);output['transform']['analysis_timing']['page_total_seconds']=time.perf_counter()-page_started;pages.append(output)
    return pages


def extract_dxf(path,folder,warnings):
    doc=ezdxf.readfile(path)
    msp=doc.modelspace();entities=list(msp)
    if len(entities)>20000:raise ValueError('현재 DXF는 modelspace 엔티티 20,000개까지 지원합니다.')
    bounds=dxf_bbox.extents(entities,fast=True)
    if not bounds.has_data:raise ValueError('표시할 수 있는 DXF 형상이 없습니다.')
    minx,miny=bounds.extmin.x,bounds.extmin.y;maxx,maxy=bounds.extmax.x,bounds.extmax.y
    dx,dy=maxx-minx,maxy-miny
    if not all(math.isfinite(v) for v in (minx,miny,maxx,maxy)) or dx<=0 or dy<=0:raise ValueError('DXF 좌표 범위가 유효하지 않습니다.')
    scale=min(2200/dx,1600/dy);w=max(300,int(dx*scale)+100);h=max(300,int(dy*scale)+100)
    image=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(image)
    def pt(x,y):return (50+(x-minx)*scale,50+(maxy-y)*scale)
    def box(bounds):
        x1,y1=pt(bounds.extmin.x,bounds.extmax.y);x2,y2=pt(bounds.extmax.x,bounds.extmin.y)
        return normalized((x1-3,y1-3,x2+3,y2+3),w,h)
    elements=[];segments=[];unsupported=set()
    for e in entities:
        if e.dxftype()=='INSERT':
            b=dxf_bbox.extents([e],fast=True)
            if b.has_data:
                attrs={a.dxf.tag:a.dxf.text for a in e.attribs}
                elements.append(element(attrs.get('TAG',e.dxf.name),'unknown',box(b),'dxf_block',{'block_name':e.dxf.name,'layer':e.dxf.layer,'attributes':attrs,'handle':e.dxf.handle}))
    for e in recursive_decompose(entities):
        kind=e.dxftype()
        try:
            if kind in ('TEXT','MTEXT','ATTRIB'):
                text=e.plain_text() if hasattr(e,'plain_text') else e.dxf.text
                x,y=pt(e.dxf.insert.x,e.dxf.insert.y);size=max(9,min(64,int((e.dxf.char_height if kind=='MTEXT' else e.dxf.height)*scale)))
                draw.text((x,y-size),text,font=font(size),fill='#263949')
                b=dxf_bbox.extents([e],fast=True)
                bb=box(b) if b.has_data else normalized((x,y-size,x+len(text)*size*.6,y),w,h)
                elements.append(element(text,'text',bb,'dxf_text',{'layer':e.dxf.layer,'handle':e.dxf.handle}))
            elif kind=='LINE':
                a=pt(e.dxf.start.x,e.dxf.start.y);b=pt(e.dxf.end.x,e.dxf.end.y)
                draw.line([a,b],fill='#344653',width=2);segments.append([*a,*b])
            else:
                points=[pt(v.x,v.y) for v in make_path(e).flattening(max(dx,dy)/2000)]
                if len(points)>1:draw.line(points,fill='#344653',width=2)
        except (TypeError,ValueError,AttributeError,NotImplementedError):unsupported.add(kind)
    if unsupported:warnings.append('미지원/부분 표시 엔티티: '+', '.join(sorted(unsupported)))
    warnings.append('DXF modelspace 2D 미리보기입니다. 레이어 색·선종·회전 글자·layout·XREF의 정밀 표현은 지원하지 않습니다.')
    return [finish(image,elements,segments,'cad_world',{'unit_code':doc.units,'world_to_pixel':[scale,0,0,-scale,50-minx*scale,50+maxy*scale],'world_bounds':[minx,miny,maxx,maxy]},folder,1,warnings)]


def extract(path,extension,folder):
    folder.mkdir(parents=True,exist_ok=True);warnings=[]
    if extension=='pdf':pages=extract_pdf(path,folder,warnings)
    elif extension=='dxf':pages=extract_dxf(path,folder,warnings)
    else:
        with Image.open(path) as img:
            if img.format not in ('PNG','JPEG'):raise ValueError('PNG/JPEG 이미지 내용이 아닙니다.')
            img.verify()
        with Image.open(path) as raw:
            if raw.width*raw.height>50_000_000:raise ValueError('이미지는 최대 5천만 화소까지 지원합니다.')
            high_resolution=ImageOps.exif_transpose(raw).convert('RGB');original=list(high_resolution.size)
            from .neural_symbols import MAX_SOURCE_SIDE
            image=high_resolution.copy();image.thumbnail((MAX_SIDE,MAX_SIDE))
            high_resolution.thumbnail((MAX_SOURCE_SIDE,MAX_SOURCE_SIDE))
        timings={};started=time.perf_counter();elements,segments=raster_candidates(image,warnings,high_resolution=high_resolution,timings=timings)
        warnings.append('이미지 요소와 연결은 기하 기반 후보입니다. 작은 기호·점선·교차 접속·물리 단위·의미는 원본과 대조해 주세요.')
        pages=[finish(image,elements,segments,'image_pixel',{'original_size':original,'rendered_size':list(image.size),'physical_scale':None,'analysis_timing':timings},folder,1,warnings)];pages[0]['transform']['analysis_timing']['page_inference_seconds']=time.perf_counter()-started
    return pages,list(dict.fromkeys(warnings))


def make_sample(path):
    doc=ezdxf.new('R2010');m=doc.modelspace()
    for name in ('PUMP','VALVE','TANK'):
        b=doc.blocks.new(name)
        if name=='PUMP':b.add_circle((0,0),8);b.add_lwpolyline([(-4,-5),(5,0),(-4,5)],close=True)
        elif name=='VALVE':b.add_lwpolyline([(-8,-6),(8,6),(8,-6),(-8,6)],close=True)
        else:b.add_lwpolyline([(-10,-16),(10,-16),(10,16),(-10,16)],close=True)
    for name,x,label in [('PUMP',20,'P-101'),('VALVE',65,'V-201'),('TANK',110,'T-301')]:
        m.add_blockref(name,(x,40));m.add_text(label,dxfattribs={'height':3,'insert':(x-8,61)})
    m.add_line((28,40),(57,40));m.add_line((73,40),(100,40))
    m.add_text('COOLING WATER - REVIEW SAMPLE',dxfattribs={'height':3,'insert':(10,85)})
    doc.saveas(path)
