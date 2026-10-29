import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .db import connect, data_dir, decode, initialize
from .extract import extract, make_sample, uid
from .workflows import router as workflow_router,record_analysis

log=logging.getLogger('lineage')
MAX_UPLOAD=40*1024*1024
STATUS=Literal['needs_review','approved','rejected']
RELATION=Literal['connectedTo','feeds','controls','adjacentTo']


def now():return datetime.now(timezone.utc).isoformat()
def encode(v):return json.dumps(v,ensure_ascii=False)
def local_mode():return os.getenv('ALLOW_INSECURE_LOCAL')=='1' and not os.getenv('APP_PASSWORD_HASH')
def digest(s):return hashlib.sha256(s.encode()).hexdigest()


def password_hash(password):
    salt=secrets.token_hex(16)
    return salt+':'+hashlib.pbkdf2_hmac('sha256',password.encode(),salt.encode(),600000).hex()


def check_password(password):
    stored=os.getenv('APP_PASSWORD_HASH','')
    try:
        salt,expected=stored.split(':',1)
        actual=hashlib.pbkdf2_hmac('sha256',password.encode(),salt.encode(),600000).hex()
        return hmac.compare_digest(actual,expected)
    except ValueError:return False


def audit(db,target,action,before,after):
    db.execute('INSERT INTO audits(target_id,action,before_value,after_value,created) VALUES(?,?,?,?,?)',
               (target,action,encode(before),encode(after),now()))


def require_row(db,table,ident):
    if table not in ('documents','pages','elements','connections','concepts'):raise ValueError('Invalid table')
    row=db.execute(f'SELECT * FROM {table} WHERE id=?',(ident,)).fetchone()
    if row is None:raise HTTPException(404,'항목을 찾을 수 없습니다.')
    return decode(row)


def worker(stop):
    while not stop.wait(.7):
        with connect() as db:
            row=db.execute("SELECT * FROM documents WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if not row:continue
            doc=dict(row);db.execute("UPDATE documents SET status='processing',updated=? WHERE id=?",(now(),doc['id']))
        try:
            folder=data_dir()/doc['id']
            pages,warnings=extract(folder/('original.'+doc['format']),doc['format'],folder)
            with (folder/('original.'+doc['format'])).open('rb') as original:
                source_sha256=hashlib.file_digest(original,'sha256').hexdigest()
            with connect() as db:
                db.execute('DELETE FROM pages WHERE document_id=?',(doc['id'],))
                for page in pages:
                    db.execute('INSERT INTO pages VALUES(?,?,?,?,?,?,?)',(page['id'],doc['id'],page['number'],page['width'],page['height'],page['coordinate_system'],encode(page['transform'])))
                    for e in page['elements']:
                        db.execute('INSERT INTO elements VALUES(?,?,?,?,?,?,?,?,?)',(e['id'],page['id'],e['label'],e['kind'],encode(e['bbox']),e['source'],e['status'],encode(e['properties']),None))
                    for edge in page['connections']:
                        db.execute('INSERT INTO connections VALUES(?,?,?,?,?,?,?)',(edge['id'],page['id'],edge['from_id'],edge['to_id'],edge['kind'],edge['source'],edge['status']))
                    record_analysis(db,page,source_sha256)
                db.execute("UPDATE documents SET status='ready',warnings=?,error=NULL,updated=? WHERE id=?",(encode(warnings),now(),doc['id']))
        except Exception as exc:
            log.warning('Extraction failed for %s (%s)',doc['id'],type(exc).__name__)
            message=str(exc)[:300] if isinstance(exc,ValueError) else '파일을 분석하지 못했습니다. 파일 형식·손상 여부와 지원 범위를 확인해 주세요.'
            with connect() as db:db.execute("UPDATE documents SET status='failed',error=?,updated=? WHERE id=?",(message,now(),doc['id']))


@asynccontextmanager
async def lifespan(app):
    if not os.getenv('APP_PASSWORD_HASH') and not local_mode():raise RuntimeError('APP_PASSWORD_HASH is required; use ALLOW_INSECURE_LOCAL=1 only on loopback.')
    initialize();stop=threading.Event();thread=threading.Thread(target=worker,args=(stop,),daemon=True);thread.start()
    yield
    stop.set();thread.join(timeout=3)

app=FastAPI(title='LINEAGE 도면 검토 API',version='0.1.0',lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
login_attempts={}


@app.middleware('http')
async def session_guard(request:Request,call_next):
    if request.method == 'POST' and request.url.path == '/api/documents':
        try:
            if int(request.headers.get('content-length','0')) > MAX_UPLOAD + 1024*1024:
                return JSONResponse({'detail':'파일은 최대 40MB입니다.'},status_code=413)
        except ValueError:
            return JSONResponse({'detail':'요청 크기가 유효하지 않습니다.'},status_code=400)
    request.state.csrf='local-development';request.state.authorized=local_mode()
    path=request.url.path
    token=request.cookies.get('lineage_session')
    if token:
        with connect() as db:
            session=db.execute('SELECT * FROM sessions WHERE token_hash=? AND expires>?',(digest(token),time.time())).fetchone()
        if session:request.state.authorized=True;request.state.csrf=session['csrf']
    public=path in ('/api/login','/api/session','/healthz')
    if path.startswith('/api/') and not public and not request.state.authorized:
        return JSONResponse({'detail':'로그인이 필요합니다.'},status_code=401)
    if request.method in ('POST','PATCH','PUT','DELETE') and path.startswith('/api/'):
        origin=request.headers.get('origin')
        allowed=[x.strip() for x in os.getenv('PUBLIC_ORIGIN','http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000').split(',')]
        if origin and origin not in allowed:return JSONResponse({'detail':'허용되지 않은 요청 출처입니다.'},status_code=403)
        if not public and not local_mode() and not hmac.compare_digest(request.headers.get('x-csrf-token',''),request.state.csrf):
            return JSONResponse({'detail':'요청 검증에 실패했습니다. 화면을 새로고침해 주세요.'},status_code=403)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['X-Frame-Options']='DENY'
    response.headers['Referrer-Policy']='same-origin'
    response.headers['Cache-Control']='no-store' if path.startswith('/api/') else 'no-cache'
    response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    return response


class Login(BaseModel):password:str=Field(min_length=1,max_length=256)
@app.get('/healthz')
def health():return {'status':'ok','version':'0.1.0'}
@app.get('/api/session')
def session(request:Request):return {'authenticated':request.state.authorized,'csrf':request.state.csrf if request.state.authorized else None,'local_mode':local_mode()}
@app.post('/api/login')
def login(body:Login,request:Request):
    address=request.client.host if request.client else 'unknown';recent=[t for t in login_attempts.get(address,[]) if t>time.time()-300]
    if len(recent)>=8:raise HTTPException(429,'잠시 후 다시 시도해 주세요.')
    if not check_password(body.password):login_attempts[address]=recent+[time.time()];raise HTTPException(401,'비밀번호를 확인해 주세요.')
    login_attempts.pop(address,None);token=secrets.token_urlsafe(32);csrf=secrets.token_urlsafe(24)
    with connect() as db:
        db.execute('DELETE FROM sessions WHERE expires<?',(time.time(),))
        db.execute('INSERT INTO sessions VALUES(?,?,?)',(digest(token),csrf,time.time()+12*3600))
    response=JSONResponse({'authenticated':True,'csrf':csrf,'local_mode':False})
    response.set_cookie('lineage_session',token,max_age=12*3600,httponly=True,samesite='lax',secure=os.getenv('COOKIE_SECURE','1')=='1')
    return response
@app.post('/api/logout')
def logout(request:Request):
    with connect() as db:db.execute('DELETE FROM sessions WHERE token_hash=?',(digest(request.cookies.get('lineage_session','')),))
    result=JSONResponse({'ok':True});result.delete_cookie('lineage_session');return result


@app.get('/api/documents')
def documents():
    with connect() as db:
        rows=db.execute('''SELECT d.*, (SELECT count(*) FROM pages p WHERE p.document_id=d.id) page_count,
        (SELECT count(*) FROM elements e JOIN pages p ON e.page_id=p.id WHERE p.document_id=d.id) element_count,
        (SELECT count(*) FROM elements e JOIN pages p ON e.page_id=p.id WHERE p.document_id=d.id AND e.status='needs_review') review_count
        FROM documents d ORDER BY created DESC''').fetchall()
    return [decode(row) for row in rows]


async def save_upload(file):
    name=Path((file.filename or 'drawing').replace('\\','/')).name[:240]
    ext=name.rsplit('.',1)[-1].lower()
    if ext not in ('png','jpg','jpeg','pdf','dxf'):raise HTTPException(415,'PNG/JPG·PDF·DXF 파일을 선택해 주세요.')
    ident=uid();folder=data_dir()/ident;folder.mkdir();path=folder/('original.'+ext);size=0
    try:
        with path.open('wb') as f:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>MAX_UPLOAD:raise HTTPException(413,'파일은 최대 40MB까지 업로드할 수 있습니다.')
                f.write(chunk)
        if size==0:raise HTTPException(400,'빈 파일은 업로드할 수 없습니다.')
        with connect() as db:db.execute('INSERT INTO documents(id,name,format,status,created,updated) VALUES(?,?,?,?,?,?)',(ident,name,ext,'queued',now(),now()))
    except Exception:
        path.unlink(missing_ok=True);folder.rmdir();raise
    finally:await file.close()
    return {'id':ident,'status':'queued'}


@app.post('/api/documents',status_code=201)
async def upload(file:UploadFile=File(...)):return await save_upload(file)
@app.post('/api/examples',status_code=201)
def example():
    ident=uid();folder=data_dir()/ident;folder.mkdir();make_sample(folder/'original.dxf')
    with connect() as db:db.execute('INSERT INTO documents(id,name,format,status,created,updated) VALUES(?,?,?,?,?,?)',(ident,'냉각수 라인 · 예제.dxf','dxf','queued',now(),now()))
    return {'id':ident,'status':'queued'}
@app.get('/api/documents/{ident}')
def document(ident:str):
    with connect() as db:
        doc=require_row(db,'documents',ident)
        doc['pages']=[decode(r) for r in db.execute('SELECT * FROM pages WHERE document_id=? ORDER BY number',(ident,))]
    return doc
@app.post('/api/documents/{ident}/retry')
def retry(ident:str):
    with connect() as db:
        doc=require_row(db,'documents',ident)
        if doc['status']!='failed':raise HTTPException(409,'실패한 분석만 다시 실행할 수 있습니다. 검토 결과는 덮어쓰지 않습니다.')
        db.execute("UPDATE documents SET status='queued',error=NULL,updated=? WHERE id=?",(now(),ident))
    return {'ok':True}
@app.get('/api/documents/{ident}/original')
def original(ident:str):
    with connect() as db:doc=require_row(db,'documents',ident)
    return FileResponse(data_dir()/ident/('original.'+doc['format']),media_type='application/octet-stream',filename=doc['name'])
@app.get('/api/documents/{ident}/export')
def export(ident:str):
    doc=document(ident)
    doc['schema_version']='lineage-0.1'
    with connect() as db:
        doc['pages']=[page_detail(p['id']) for p in doc['pages']]
        doc['concepts']=[decode(r) for r in db.execute('SELECT * FROM concepts ORDER BY name')]
        target_ids=[ident]+[p['id'] for p in doc['pages']]+[e['id'] for p in doc['pages'] for e in p['elements']]+[c['id'] for p in doc['pages'] for c in p['connections']]
        doc['history']=[decode(r) for r in db.execute('SELECT * FROM audits ORDER BY id') if r['target_id'] in target_ids]
    return JSONResponse(doc,headers={'Content-Disposition':f'attachment; filename="lineage-{ident[:8]}.json"'})
@app.get('/api/pages/{ident}')
def page_detail(ident:str):
    with connect() as db:
        page=require_row(db,'pages',ident)
        page['elements']=[decode(r) for r in db.execute('SELECT * FROM elements WHERE page_id=? ORDER BY rowid',(ident,))]
        page['connections']=[decode(r) for r in db.execute('SELECT * FROM connections WHERE page_id=? ORDER BY rowid',(ident,))]
    return page
@app.get('/api/pages/{ident}/image')
def page_image(ident:str):
    with connect() as db:page=require_row(db,'pages',ident)
    return FileResponse(data_dir()/page['document_id']/(ident+'.png'),media_type='image/png')


class ElementFields(BaseModel):
    label:str=Field(min_length=1,max_length=300)
    kind:str=Field(min_length=1,max_length=100)
    bbox:list[float]=Field(min_length=4,max_length=4)
    status:STATUS='needs_review'
    concept_id:str|None=None
    @field_validator('bbox')
    @classmethod
    def valid_bbox(cls,v):
        if not all(math.isfinite(n) and 0<=n<=1 for n in v) or v[0]>=v[2] or v[1]>=v[3]:raise ValueError('Bounding box must have positive area in normalized [0,1] coordinates')
        return v


def validate_concept(db,ident):
    if ident:require_row(db,'concepts',ident)

@app.post('/api/pages/{ident}/elements',status_code=201)
def create_element(ident:str,body:ElementFields):
    eid=uid()
    with connect() as db:
        require_row(db,'pages',ident);validate_concept(db,body.concept_id)
        db.execute('INSERT INTO elements VALUES(?,?,?,?,?,?,?,?,?)',(eid,ident,body.label,body.kind,encode(body.bbox),'manual',body.status,'{}',body.concept_id))
        audit(db,eid,'create_element',None,body.model_dump())
    return {'id':eid}
@app.put('/api/elements/{ident}')
def update_element(ident:str,body:ElementFields):
    with connect() as db:
        previous=require_row(db,'elements',ident);validate_concept(db,body.concept_id)
        db.execute('UPDATE elements SET label=?,kind=?,bbox=?,status=?,concept_id=? WHERE id=?',(body.label,body.kind,encode(body.bbox),body.status,body.concept_id,ident))
        audit(db,ident,'update_element',previous,body.model_dump())
    return {'ok':True}
@app.delete('/api/elements/{ident}')
def delete_element(ident:str):
    with connect() as db:
        previous=require_row(db,'elements',ident)
        edges=[decode(r) for r in db.execute('SELECT * FROM connections WHERE from_id=? OR to_id=?',(ident,ident))]
        audit(db,ident,'delete_element',{'element':previous,'connections':edges},None)
        db.execute('DELETE FROM elements WHERE id=?',(ident,))
    return {'ok':True}


class ConnectionFields(BaseModel):
    from_id:str
    to_id:str
    kind:RELATION='connectedTo'
    status:STATUS='needs_review'

def validate_endpoints(db,page,body):
    if body.from_id==body.to_id:raise HTTPException(422,'서로 다른 두 요소를 선택해 주세요.')
    for ident in (body.from_id,body.to_id):
        e=require_row(db,'elements',ident)
        if e['page_id']!=page:raise HTTPException(422,'같은 페이지의 요소만 연결할 수 있습니다.')

@app.post('/api/pages/{ident}/connections',status_code=201)
def create_connection(ident:str,body:ConnectionFields):
    cid=uid()
    with connect() as db:
        require_row(db,'pages',ident);validate_endpoints(db,ident,body)
        db.execute('INSERT INTO connections VALUES(?,?,?,?,?,?,?)',(cid,ident,body.from_id,body.to_id,body.kind,'manual',body.status))
        audit(db,cid,'create_connection',None,body.model_dump())
    return {'id':cid}
@app.put('/api/connections/{ident}')
def update_connection(ident:str,body:ConnectionFields):
    with connect() as db:
        previous=require_row(db,'connections',ident);validate_endpoints(db,previous['page_id'],body)
        db.execute('UPDATE connections SET from_id=?,to_id=?,kind=?,status=? WHERE id=?',(body.from_id,body.to_id,body.kind,body.status,ident))
        audit(db,ident,'update_connection',previous,body.model_dump())
    return {'ok':True}
@app.delete('/api/connections/{ident}')
def delete_connection(ident:str):
    with connect() as db:
        previous=require_row(db,'connections',ident);audit(db,ident,'delete_connection',previous,None)
        db.execute('DELETE FROM connections WHERE id=?',(ident,))
    return {'ok':True}


class ConceptFields(BaseModel):
    name:str=Field(min_length=1,max_length=150)
    definition:str=Field(min_length=1,max_length=5000)
    scope:str=Field(min_length=1,max_length=150)
    unit:str=Field(default='',max_length=50)
    @field_validator('name','definition','scope')
    @classmethod
    def not_blank(cls,v):
        if not v.strip():raise ValueError('This field cannot be blank')
        return v.strip()
@app.get('/api/concepts')
def concepts():
    with connect() as db:return [decode(r) for r in db.execute('SELECT * FROM concepts ORDER BY name')]
@app.post('/api/concepts',status_code=201)
def create_concept(body:ConceptFields):
    ident=uid()
    try:
        with connect() as db:
            db.execute('INSERT INTO concepts(id,name,definition,scope,unit,created,updated) VALUES(?,?,?,?,?,?,?)',(ident,body.name,body.definition,body.scope,body.unit,now(),now()))
            audit(db,ident,'create_concept',None,body.model_dump())
    except sqlite3.IntegrityError:raise HTTPException(409,'같은 범위에 동일한 이름의 의미가 있습니다.')
    return {'id':ident}
@app.put('/api/concepts/{ident}')
def update_concept(ident:str,body:ConceptFields):
    try:
        with connect() as db:
            previous=require_row(db,'concepts',ident)
            db.execute('UPDATE concepts SET name=?,definition=?,scope=?,unit=?,updated=?,version=version+1 WHERE id=?',(body.name,body.definition,body.scope,body.unit,now(),ident))
            audit(db,ident,'update_concept',previous,body.model_dump())
    except sqlite3.IntegrityError:raise HTTPException(409,'같은 범위에 동일한 이름의 의미가 있습니다.')
    return {'ok':True}
@app.get('/api/history/{ident}')
def history(ident:str):
    with connect() as db:return [decode(r) for r in db.execute('SELECT * FROM audits WHERE target_id=? ORDER BY id DESC LIMIT 100',(ident,))]

app.include_router(workflow_router)
frontend=Path(__file__).resolve().parents[1]/'frontend'/'dist'
if frontend.exists():app.mount('/',StaticFiles(directory=frontend,html=True),name='frontend')
