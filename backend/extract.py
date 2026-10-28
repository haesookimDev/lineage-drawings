"""Deterministic extraction baselines. No semantic class or standard ID is invented."""
import io
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import cv2
import ezdxf
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ezdxf import bbox as dxf_bbox
from ezdxf.disassemble import recursive_decompose
from ezdxf.path import make_path

MAX_PAGES = 25
MAX_SIDE = 2400
MAX_ELEMENTS = 500
Image.MAX_IMAGE_PIXELS = 50_000_000


def uid():
    return uuid.uuid4().hex


def normalized(box, w, h):
    x1, y1, x2, y2 = box
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


def ocr(image, warnings):
    binary = shutil.which('tesseract')
    if not binary:
        warnings.append('OCR 실행기가 없어 문자 추출을 생략했습니다. 요소는 직접 추가할 수 있습니다.')
        return []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'page.png'
        image.save(path)
        try:
            langs = subprocess.run([binary, '--list-langs'], capture_output=True, text=True, timeout=10).stdout
            language = 'eng+kor' if '\nkor' in langs else 'eng'
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
            return words[:MAX_ELEMENTS]
        except (subprocess.SubprocessError, ValueError):
            warnings.append('문자 추출이 완료되지 않았습니다. 원본에서 직접 검토해 주세요.')
            return []


def raster_candidates(image, warnings, with_ocr=True):
    w,h = image.size
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
    texts = ocr(image,warnings) if with_ocr else []
    mask = bw.copy()
    for item in texts:
        x1,y1,x2,y2=item['bbox']; mask[max(0,int(y1*h)-2):min(h,int(y2*h)+2),max(0,int(x1*w)-2):min(w,int(x2*w)+2)] = 0
    horizontal=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(25,w//35),1)))
    vertical=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(25,h//35))))
    line_mask=cv2.bitwise_or(horizontal,vertical)
    symbols=cv2.subtract(mask,line_mask)
    symbols=cv2.morphologyEx(symbols,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,_,stats,_=cv2.connectedComponentsWithStats(symbols)
    candidates=[]
    for x,y,cw,ch,area in stats[1:]:
        if area<30 or cw<8 or ch<8 or cw>w*.7 or ch>h*.7 or cw*ch>w*h*.25:
            continue
        candidates.append(element(f'요소 {len(candidates)+1}', 'unknown',
                                  normalized((int(x),int(y),int(x+cw),int(y+ch)),w,h),'image_candidate'))
        if len(candidates)>=200:
            warnings.append('이미지 요소 후보를 200개까지 표시합니다. 영역 단위 검토를 권장합니다.');break
    lines=cv2.HoughLinesP(line_mask,1,np.pi/180,threshold=35,minLineLength=30,maxLineGap=7)
    segments=[] if lines is None else [list(map(int,line[0])) for line in lines[:2000]]
    return (texts+candidates)[:MAX_ELEMENTS],segments


def infer_connections(elements, segments, w, h):
    """A line endpoint near two different symbol boxes creates an undirected candidate."""
    symbols=[e for e in elements if e['kind']!='text']
    if len(symbols)>200:return []
    def nearest(x,y):
        best=None;distance=max(10,min(w,h)*.018)
        for e in symbols:
            a,b,c,d=e['bbox'];a*=w;c*=w;b*=h;d*=h
            dx=max(a-x,0,x-c);dy=max(b-y,0,y-d);dist=math.hypot(dx,dy)
            if dist<distance:distance=dist;best=e['id']
        return best
    seen=set();out=[]
    for x1,y1,x2,y2 in segments:
        a,b=nearest(x1,y1),nearest(x2,y2)
        if a and b and a!=b and tuple(sorted((a,b))) not in seen:
            seen.add(tuple(sorted((a,b))));out.append(dict(id=uid(),from_id=a,to_id=b,kind='connectedTo',source='line_candidate',status='needs_review'))
    return out[:500]


def finish(image, elements, segments, coordinate, transform, folder, number, warnings):
    pid=uid();image.save(folder/f'{pid}.png')
    if len(elements)>MAX_ELEMENTS:warnings.append('요소 수가 많아 500개까지 추출했습니다.')
    elements=elements[:MAX_ELEMENTS]
    return dict(id=pid,number=number,width=image.width,height=image.height,coordinate_system=coordinate,
                transform=transform,elements=elements,connections=infer_connections(elements,segments,*image.size))


def extract_pdf(path,folder,warnings):
    pages=[]
    with fitz.open(path) as doc:
        if not doc.is_pdf or doc.needs_pass:raise ValueError('암호화되었거나 읽을 수 없는 PDF입니다.')
        if len(doc)>MAX_PAGES:raise ValueError(f'PDF는 최대 {MAX_PAGES}페이지까지 지원합니다.')
        for n,page in enumerate(doc):
            if page.rect.width<=0 or page.rect.height<=0:raise ValueError('PDF 페이지 크기가 유효하지 않습니다.')
            scale=min(2.,MAX_SIDE/max(page.rect.width,page.rect.height))
            pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False,colorspace=fitz.csRGB)
            image=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
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
                extra,lines=raster_candidates(image,warnings);elements+=extra
                if not segments:segments=lines
            elif page.get_images():
                extra,lines=raster_candidates(image,warnings,with_ocr=True)
                for e in extra:
                    if e['kind']=='text' and any(e['label']==t['label'] for t in elements if t['kind']=='text'):continue
                    elements.append(e)
                segments+=lines
                warnings.append(f'{n+1}페이지는 벡터·이미지 혼합 분석 결과입니다. 중복과 누락을 검토해 주세요.')
            pages.append(finish(image,elements,segments,'pdf_page',{'scale':scale,'rotation':page.rotation,'page_rect':list(page.rect)},folder,n+1,warnings))
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
            image=ImageOps.exif_transpose(raw).convert('RGB');original=list(image.size)
            image.thumbnail((MAX_SIDE,MAX_SIDE))
        elements,segments=raster_candidates(image,warnings)
        warnings.append('이미지 요소와 연결은 기초 추출 후보입니다. 물리 단위·의미·접속을 원본과 대조해 주세요.')
        pages=[finish(image,elements,segments,'image_pixel',{'original_size':original,'rendered_size':list(image.size),'physical_scale':None},folder,1,warnings)]
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
