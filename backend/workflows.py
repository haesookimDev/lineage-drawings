"""Versioned human evidence, immutable evaluation, semantics and revisions.

Original extraction observations are never changed by topology review. Ground
truth and predictions have separate records; project-level splits are explicit.
"""
import hashlib,json,math,uuid
from collections import defaultdict,deque
from datetime import datetime,timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from fastapi import APIRouter,HTTPException,Query
from fastapi.responses import Response
from pydantic import BaseModel,Field,field_validator
from .db import connect,decode

router=APIRouter(prefix='/api')
Status=Literal['needs_review','approved','rejected']
Split=Literal['unassigned','training','validation','test']
def now():return datetime.now(timezone.utc).isoformat()
def uid():return uuid.uuid4().hex
def dump(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def sha(value):return hashlib.sha256(dump(value).encode()).hexdigest()
def audit(db,target,action,before,after):
    db.execute('INSERT INTO audits(target_id,action,before_value,after_value,created) VALUES(?,?,?,?,?)',(target,action,dump(before) if before is not None else None,dump(after) if after is not None else None,now()))
def required(db,table,ident):
    assert table in ('pages','elements','connections','documents','projects','concepts','annotations','analysis_runs','comparisons','evaluations','concept_relations')
    row=db.execute(f'SELECT * FROM {table} WHERE id=?',(ident,)).fetchone()
    if row is None:raise HTTPException(404,'항목을 찾을 수 없습니다.')
    return dict(row)
def page_data(db,ident):
    p=decode(required(db,'pages',ident))
    p['elements']=[decode(r) for r in db.execute('SELECT * FROM elements WHERE page_id=? ORDER BY rowid',(ident,))]
    p['connections']=[decode(r) for r in db.execute('SELECT * FROM connections WHERE page_id=? ORDER BY rowid',(ident,))]
    return p
@lru_cache(maxsize=1)
def runtime_digest():
    root=Path(__file__).parent
    return sha({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ('.py','.onnx','.json','.npz') or p.name.endswith('.xml.gz'))})
def record_analysis(db,page,source_sha256):
    # Called inside the worker transaction before a page can be reviewed.
    db.execute('INSERT INTO analysis_runs VALUES(?,?,?,?,?,?)',(uid(),page['id'],source_sha256,runtime_digest(),now(),dump(page)))

def model_topology(page):
    rec=page.get('transform',{}).get('connection_inference',{});ports=[]
    for e in page['elements']:
        for i,p in enumerate(e.get('properties',{}).get('ports',[])):
            ports.append({'id':f"port-{e['id']}-{i}",'element_id':e['id'],'point':p['point'],'side':p['side'],'status':'needs_review','source':'model_observation','score':p.get('score')})
    junctions=[{'id':f'junction-{i}','point':j['point'],'kind':j['type'] if j['type'] in ('tee','dot','plain_cross','bend') else 'unknown','status':'needs_review','source':'model_observation','score':j.get('score')} for i,j in enumerate(rec.get('junctions',[]))]
    edges={tuple(sorted((e['from_id'],e['to_id']))):e['id'] for e in page['connections']}
    paths=[{'id':f'path-{i}','connection_id':edges[tuple(sorted((p['from_id'],p['to_id'])))],'points':p['points'],'status':'needs_review','source':'model_observation','from_port_id':None,'to_port_id':None} for i,p in enumerate(rec.get('paths',[])) if tuple(sorted((p['from_id'],p['to_id']))) in edges]
    return {'ports':ports[:2400],'junctions':junctions[:500],'paths':paths[:500]}
def reviewed(db,page):
    row=db.execute('SELECT * FROM topology_reviews WHERE page_id=? ORDER BY version DESC LIMIT 1',(page['id'],)).fetchone()
    if row:
        payload=json.loads(row['payload']);elements={e['id']:e for e in page['elements']};connections={e['id']:e for e in page['connections']}
        for p in payload['ports']:
            owner=elements.get(p['element_id']);p['stale']=not owner or owner['status']=='rejected' or p.get('owner_bbox')!=owner['bbox']
        for p in payload['paths']:
            edge=connections.get(p['connection_id']);p['stale']=not edge or edge['status']=='rejected' or p.get('endpoint_signature')!=endpoint_signature(edge,elements)
        return {**dict(row),'payload':payload}
    return {'page_id':page['id'],'version':0,'reviewer':'','updated':None,'payload':model_topology(page)}

def endpoint_signature(edge,elements):
    return [[ident,elements[ident]['bbox']] for ident in (edge['from_id'],edge['to_id']) if ident in elements]

class Point(BaseModel):
    point:list[float]=Field(min_length=2,max_length=2)
    @field_validator('point')
    @classmethod
    def coordinates(cls,v):
        if not all(math.isfinite(x) and 0<=x<=1 for x in v):raise ValueError('좌표는0–1 유한값이어야 합니다.')
        return v
class Port(Point):
    id:str=Field(min_length=1,max_length=90)
    element_id:str=Field(min_length=1,max_length=90)
    side:Literal['left','right','top','bottom']
    status:Status='needs_review'
class Junction(Point):
    id:str=Field(min_length=1,max_length=90)
    kind:Literal['tee','dot','plain_cross','bend','unknown']
    status:Status='needs_review'
class Route(BaseModel):
    id:str=Field(min_length=1,max_length=90)
    connection_id:str=Field(min_length=1,max_length=90)
    points:list[list[float]]=Field(min_length=2,max_length=128)
    from_port_id:str|None=Field(default=None,max_length=90)
    to_port_id:str|None=Field(default=None,max_length=90)
    status:Status='needs_review'
    @field_validator('points')
    @classmethod
    def coordinates(cls,v):
        for point in v:Point(point=point)
        return v
class ReviewFields(BaseModel):
    expected_version:int=Field(ge=0)
    reviewer:str=Field(min_length=1,max_length=100)
    ports:list[Port]=Field(default_factory=list,max_length=2400)
    junctions:list[Junction]=Field(default_factory=list,max_length=500)
    paths:list[Route]=Field(default_factory=list,max_length=500)
    @field_validator('reviewer')
    @classmethod
    def name(cls,v):
        if not v.strip():raise ValueError('검토자 이름이 필요합니다.')
        return v.strip()
def check_review(page,payload):
    elements={e['id']:e for e in page['elements']};connections={e['id']:e for e in page['connections']};ports={p['id']:p for p in payload['ports']}
    for field in ('ports','junctions','paths'):
        if len({r['id'] for r in payload[field]})!=len(payload[field]):raise HTTPException(422,'중복된 관찰 ID입니다.')
    if any(p['element_id'] not in elements or elements[p['element_id']]['kind']=='text' for p in ports.values()):raise HTTPException(422,'포트는 이 페이지의 기호에 속해야 합니다.')
    for route in payload['paths']:
        if route['connection_id'] not in connections:raise HTTPException(422,'이 페이지의 연결을 선택해 주세요.')
        edge=connections[route['connection_id']]
        for field,owner,index in [('from_port_id',edge['from_id'],0),('to_port_id',edge['to_id'],-1)]:
            pid=route.get(field)
            if pid:
                if pid not in ports or ports[pid]['element_id']!=owner or ports[pid]['status']=='rejected':raise HTTPException(422,'경로 포트의 소유 요소가 일치하지 않습니다.')
                if math.dist(route['points'][index],ports[pid]['point'])>1e-6:raise HTTPException(422,'경로 끝점을 선택한 포트에 맞춰 주세요.')
    if len({r['connection_id'] for r in payload['paths']})!=len(payload['paths']):raise HTTPException(422,'연결당 경로는 하나만 저장할 수 있습니다.')
def save_review(db,page,body):
    current=reviewed(db,page)
    if body.expected_version!=current['version']:raise HTTPException(409,'다른 검토가 저장됐습니다. 최신 버전을 다시 불러오세요.')
    payload={k:[r.model_dump() for r in getattr(body,k)] for k in ('ports','junctions','paths')};check_review(page,payload)
    previous={r['id']:r for k in ('ports','junctions','paths') for r in current['payload'][k]}
    for field,rows in payload.items():
        for row in rows:
            prior=previous.get(row['id']);row['source']='human_review'
            if prior and prior.get('source')=='model_observation':row['observation']=prior
            elif prior and prior.get('observation'):row['observation']=prior['observation']
            if field=='ports':row['owner_bbox']=next(e['bbox'] for e in page['elements'] if e['id']==row['element_id'])
            if field=='paths':row['endpoint_signature']=endpoint_signature(next(e for e in page['connections'] if e['id']==row['connection_id']),{e['id']:e for e in page['elements']})
    version=current['version']+1
    db.execute('INSERT INTO topology_reviews VALUES(?,?,?,?,?)',(page['id'],version,body.reviewer,now(),dump(payload)))
    audit(db,page['id'],'review_topology',current,{'version':version,'reviewer':body.reviewer,'payload':payload})
    return {'version':version,'payload':payload,'reviewer':body.reviewer}
@router.get('/pages/{ident}/topology-review')
def topology_get(ident:str):
    with connect() as db:return reviewed(db,page_data(db,ident))
@router.put('/pages/{ident}/topology-review')
def topology_save(ident:str,body:ReviewFields):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE');return save_review(db,page_data(db,ident),body)
@router.get('/pages/{ident}/topology-history')
def topology_history(ident:str):
    with connect() as db:
        required(db,'pages',ident)
        return [{k:r[k] for k in ('version','reviewer','updated')} for r in db.execute('SELECT * FROM topology_reviews WHERE page_id=? ORDER BY version DESC LIMIT 50',(ident,))]
class Restore(BaseModel):
    expected_version:int=Field(ge=0)
    version:int=Field(ge=0)
    reviewer:str=Field(min_length=1,max_length=100)
@router.post('/pages/{ident}/topology-restore')
def topology_restore(ident:str,body:Restore):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE');page=page_data(db,ident)
        if body.version==0:payload=model_topology(page)
        else:
            row=db.execute('SELECT payload FROM topology_reviews WHERE page_id=? AND version=?',(ident,body.version)).fetchone()
            if not row:raise HTTPException(404,'해당 검토 버전이 없습니다.')
            payload=json.loads(row['payload'])
        return save_review(db,page,ReviewFields(expected_version=body.expected_version,reviewer=body.reviewer,**payload))

class ProjectFields(BaseModel):
    name:str=Field(min_length=1,max_length=150)
    split:Split='unassigned'
    @field_validator('name')
    @classmethod
    def nonempty(cls,v):
        if not v.strip():raise ValueError('이름이 필요합니다.')
        return v.strip()
@router.get('/projects')
def projects():
    with connect() as db:return [dict(r) for r in db.execute('SELECT p.*,(SELECT count(*) FROM document_meta m WHERE m.project_id=p.id) document_count FROM projects p ORDER BY p.name')]
@router.post('/projects',status_code=201)
def project_create(body:ProjectFields):
    import sqlite3
    ident=uid()
    try:
        with connect() as db:
            db.execute('INSERT INTO projects VALUES(?,?,?,?,?)',(ident,body.name,body.split,now(),now()));audit(db,ident,'create_project',None,body.model_dump())
    except sqlite3.IntegrityError:raise HTTPException(409,'같은 이름의 프로젝트가 있습니다.')
    return {'id':ident}
@router.put('/projects/{ident}')
def project_update(ident:str,body:ProjectFields):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        before=required(db,'projects',ident)
        if before['split']!=body.split and db.execute('SELECT 1 FROM annotations WHERE project_id=? LIMIT 1',(ident,)).fetchone():raise HTTPException(409,'정답이 있는 프로젝트의 분할은 잠겨 있습니다. 기존 평가와 학습 분리를 보존합니다.')
        import sqlite3
        try:db.execute('UPDATE projects SET name=?,split=?,updated=? WHERE id=?',(body.name,body.split,now(),ident))
        except sqlite3.IntegrityError:raise HTTPException(409,'같은 이름의 프로젝트가 있습니다.')
        audit(db,ident,'update_project',before,body.model_dump())
    return {'ok':True}
class DocumentMeta(BaseModel):
    project_id:str
    revision:str=Field(default='',max_length=100)
@router.get('/documents/{ident}/project')
def document_project(ident:str):
    with connect() as db:
        required(db,'documents',ident);r=db.execute('SELECT m.*,p.name,p.split FROM document_meta m JOIN projects p ON p.id=m.project_id WHERE document_id=?',(ident,)).fetchone()
        return dict(r) if r else None
@router.put('/documents/{ident}/project')
def document_project_save(ident:str,body:DocumentMeta):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        required(db,'documents',ident);required(db,'projects',body.project_id);r=db.execute('SELECT * FROM document_meta WHERE document_id=?',(ident,)).fetchone();before=dict(r) if r else None
        if before and before['project_id']!=body.project_id and db.execute('SELECT 1 FROM annotations a JOIN pages p ON a.page_id=p.id WHERE p.document_id=? LIMIT 1',(ident,)).fetchone():raise HTTPException(409,'정답을 만든 문서는 프로젝트를 옮길 수 없습니다.')
        db.execute('INSERT INTO document_meta VALUES(?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET project_id=excluded.project_id,revision=excluded.revision,updated=excluded.updated',(ident,body.project_id,body.revision,now()));audit(db,ident,'assign_project',before,body.model_dump())
    return {'ok':True}
@router.get('/projects/{ident}/pages')
def project_pages(ident:str):
    with connect() as db:
        required(db,'projects',ident)
        return [dict(r) for r in db.execute('SELECT p.id,p.number,p.width,p.height,d.id document_id,d.name,m.revision FROM pages p JOIN documents d ON p.document_id=d.id JOIN document_meta m ON m.document_id=d.id WHERE m.project_id=? ORDER BY d.created DESC,p.number LIMIT 1000',(ident,))]

def inside(box,roi):
    return roi[0]<=(box[0]+box[2])/2<=roi[2] and roi[1]<=(box[1]+box[3])/2<=roi[3]
def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.
def match(pred,truth,threshold=.5):
    neighbors={p['id']:[g['id'] for g in sorted(truth,key=lambda g:(-iou(p['bbox'],g['bbox']),g['id'])) if iou(p['bbox'],g['bbox'])>=threshold] for p in pred};assigned={}
    def augment(pid,seen):
        for gid in neighbors[pid]:
            if gid in seen:continue
            seen.add(gid)
            if gid not in assigned or augment(assigned[gid],seen):assigned[gid]=pid;return True
        return False
    for p in pred:augment(p['id'],set())
    return {pid:gid for gid,pid in assigned.items()}
def counts(tp,fp,fn):return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
def score_prediction(pred,truth):
    roi=truth['roi'];pp=[e for e in pred['elements'] if e['kind']!='text' and inside(e['bbox'],roi)];ids={p['id'] for p in pp};gg=truth['elements'];mapping=match(pp,gg);expected={tuple(sorted((e['from_id'],e['to_id']))) for e in truth['connections']};edges={tuple(sorted((mapping.get(e['from_id'],'unmatched:'+e['from_id']),mapping.get(e['to_id'],'unmatched:'+e['to_id'])))) for e in pred['connections'] if e['from_id'] in ids and e['to_id'] in ids};matched=set(mapping.values());causes=defaultdict(int)
    for edge in expected:causes['recovered' if edge in edges else 'missing_symbol' if not set(edge)<=matched else 'routing']+=1
    return {'symbols':counts(len(mapping),len(pp)-len(mapping),len(gg)-len(mapping)),'connections':counts(len(edges&expected),len(edges-expected),len(expected-edges)),'miss_causes':dict(causes),'unmatched_gt':[g['id'] for g in gg if g['id'] not in matched],'false_connections':[list(e) for e in sorted(edges-expected)],'timings':pred.get('transform',{}).get('analysis_timing',{}),'class_agnostic':True}
class AnnotationFields(BaseModel):
    reviewer:str=Field(min_length=1,max_length=100)
    basis:Literal['source_manual','ai_assisted','second_review']='ai_assisted'
    roi:list[float]=Field(default_factory=lambda:[0,0,1,1],min_length=4,max_length=4)
    confirmed:bool=False
    @field_validator('roi')
    @classmethod
    def bounds(cls,v):
        if not all(math.isfinite(x) and 0<=x<=1 for x in v) or v[0]>=v[2] or v[1]>=v[3]:raise ValueError('올바른 정답 범위가 필요합니다.')
        return v
    @field_validator('reviewer')
    @classmethod
    def reviewer_name(cls,v):
        if not v.strip():raise ValueError('주석자가 필요합니다.')
        return v.strip()
@router.post('/pages/{ident}/annotations',status_code=201)
def annotation_create(ident:str,body:AnnotationFields):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE');page=page_data(db,ident);meta=db.execute('SELECT m.*,p.split FROM document_meta m JOIN projects p ON m.project_id=p.id WHERE document_id=?',(page['document_id'],)).fetchone()
        if not meta or meta['split']=='unassigned':raise HTTPException(422,'프로젝트와 학습/검증/테스트 분할을 먼저 지정해 주세요.')
        scoped=[e for e in page['elements'] if e['kind']!='text' and inside(e['bbox'],body.roi)];ids={e['id'] for e in scoped};edges=[e for e in page['connections'] if e['from_id'] in ids and e['to_id'] in ids]
        if body.confirmed and (not scoped or any(e['status']=='needs_review' for e in scoped+edges)):raise HTTPException(422,'범위 안의 모든 기호·연결을 검토한 뒤 확정하세요. 누락 기호도 직접 추가해 주세요.')
        approved=[e for e in scoped if e['status']=='approved'];accepted={e['id'] for e in approved};ground={'schema':'lineage-annotation-1','roi':body.roi,'elements':approved,'connections':[e for e in edges if e['status']=='approved' and e['from_id'] in accepted and e['to_id'] in accepted],'topology':reviewed(db,page),'page_number':page['number'],'width':page['width'],'height':page['height'],'split':meta['split'],'document_revision':meta['revision'],'complete_scope_self_attested':body.confirmed,'basis':body.basis,'expert_certified':False}
        run=db.execute('SELECT id,source_sha256,runtime_sha256 FROM analysis_runs WHERE page_id=?',(ident,)).fetchone();ground['analysis']=dict(run) if run else None
        revision=db.execute('SELECT COALESCE(MAX(revision),0)+1 FROM annotations WHERE page_id=?',(ident,)).fetchone()[0];aid=uid();db.execute('INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?)',(aid,ident,meta['project_id'],revision,body.reviewer,body.basis,int(body.confirmed),now(),dump(ground)));audit(db,ident,'snapshot_annotation',None,{'id':aid,'revision':revision,'confirmed':body.confirmed,'roi':body.roi,'reviewer':body.reviewer})
    return {'id':aid,'revision':revision,'symbols':len(approved),'connections':len(ground['connections'])}
@router.get('/pages/{ident}/annotations')
def annotations(ident:str):
    with connect() as db:
        required(db,'pages',ident)
        return [{k:r[k] for k in ('id','project_id','revision','reviewer','basis','confirmed','created')} for r in db.execute('SELECT * FROM annotations WHERE page_id=? ORDER BY revision DESC',(ident,))]
@router.get('/datasets/export')
def dataset_export(split:Split='test'):
    with connect() as db:
        records=[{**dict(r),'payload':json.loads(r['payload'])} for r in db.execute('SELECT a.*,p.name project_name,p.split FROM annotations a JOIN projects p ON p.id=a.project_id WHERE a.confirmed=1 AND p.split=? ORDER BY a.created LIMIT 500',(split,))]
    return {'schema':'lineage-dataset-1','split_unit':'project','split':split,'annotations':records,'expert_certified':False}
class EvaluationFields(BaseModel):
    annotation_ids:list[str]=Field(min_length=1,max_length=50)
    prediction_pages:dict[str,str]=Field(default_factory=dict,max_length=50)
    split:Literal['training','validation','test']='test'
@router.post('/evaluations',status_code=201)
def evaluate(body:EvaluationFields):
    with connect() as db:
        rows=[]
        for aid in dict.fromkeys(body.annotation_ids):
            annotation=required(db,'annotations',aid);project=required(db,'projects',annotation['project_id']);truth=json.loads(annotation['payload'])
            if not annotation['confirmed'] or project['split']!=body.split:raise HTTPException(422,'확정 정답과 선택한 프로젝트 분할이 일치해야 합니다.')
            page_id=body.prediction_pages.get(aid,annotation['page_id']);page=required(db,'pages',page_id);meta=db.execute('SELECT project_id FROM document_meta WHERE document_id=?',(page['document_id'],)).fetchone()
            if not meta or meta['project_id']!=annotation['project_id']:raise HTTPException(422,'예측 페이지의 프로젝트가 다릅니다.')
            run=db.execute('SELECT * FROM analysis_runs WHERE page_id=?',(page_id,)).fetchone()
            if not run or not truth['analysis']:raise HTTPException(409,'최초 예측 스냅샷이 없습니다. 원본을 새 문서로 업로드해 평가하세요.')
            if run['source_sha256']!=truth['analysis']['source_sha256'] or page['number']!=truth['page_number'] or (page['width'],page['height'])!=(truth['width'],truth['height']):raise HTTPException(422,'동일한 원본·페이지·좌표계의 예측만 평가할 수 있습니다.')
            result=score_prediction(json.loads(run['payload']),truth);rows.append({'annotation_id':aid,'prediction_page_id':page_id,'project_id':project['id'],'project':project['name'],'runtime_sha256':run['runtime_sha256'],'scores':result,'review_basis':annotation['basis']})
        aggregate={m:counts(**{k:sum(r['scores'][m][k] for r in rows) for k in ('tp','fp','fn')}) for m in ('symbols','connections')};result={'schema':'lineage-evaluation-1','split':body.split,'role':'project_test' if body.split=='test' else 'development_diagnostic','rows':rows,'aggregate':aggregate,'uses_immutable_initial_predictions':True,'expert_certified':False};ident=uid();db.execute('INSERT INTO evaluations VALUES(?,?,?,?)',(ident,now(),dump(body.model_dump()),dump(result)))
    return {'id':ident,**result}
@router.get('/evaluations/{ident}')
def evaluation_get(ident:str):
    with connect() as db:return {'id':ident,**json.loads(required(db,'evaluations',ident)['result'])}
@router.get('/pages/{ident}/evaluations')
def evaluation_history(ident:str):
    with connect() as db:
        required(db,'pages',ident)
        records=[]
        for r in db.execute('SELECT * FROM evaluations ORDER BY created DESC LIMIT 1000'):
            result=json.loads(r['result'])
            if any(row['prediction_page_id']==ident for row in result['rows']):records.append({'id':r['id'],'created':r['created'],'split':result['split'],'aggregate':result['aggregate']})
        return records[:50]


class FactFields(BaseModel):
    identifier:str=Field(default='',max_length=200)
    source_url:str=Field(default='',max_length=2000)
    status:Status='needs_review'
    reviewer:str=Field(default='',max_length=100)
    note:str=Field(default='',max_length=3000)
    expected_version:int=Field(default=0,ge=0)
    @field_validator('source_url')
    @classmethod
    def citation(cls,v):
        from urllib.parse import urlparse
        if v and (urlparse(v).scheme!='https' or not urlparse(v).hostname or urlparse(v).username):raise ValueError('출처는 자격증명 없는 HTTPS URL이어야 합니다.')
        return v
@router.get('/concepts/{ident}/standard')
def fact_get(ident:str):
    with connect() as db:
        required(db,'concepts',ident);r=db.execute('SELECT * FROM concept_facts WHERE concept_id=?',(ident,)).fetchone()
        return dict(r) if r else {'concept_id':ident,'version':0,'identifier':'','source_url':'','status':'needs_review','reviewer':'','note':''}
@router.put('/concepts/{ident}/standard')
def fact_save(ident:str,body:FactFields):
    if body.status=='approved' and not (body.identifier.strip() and body.source_url and body.reviewer.strip()):raise HTTPException(422,'표준 대응 확인에는 식별자·출처·검토자가 필요합니다.')
    with connect() as db:
        db.execute('BEGIN IMMEDIATE');required(db,'concepts',ident);r=db.execute('SELECT * FROM concept_facts WHERE concept_id=?',(ident,)).fetchone();before=dict(r) if r else None;version=r['version'] if r else 0
        if body.expected_version!=version:raise HTTPException(409,'표준 대응이 변경됐습니다. 다시 불러오세요.')
        values=(ident,body.identifier.strip(),body.source_url,body.status,body.reviewer.strip(),body.note,version+1,now());db.execute('INSERT INTO concept_facts VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(concept_id) DO UPDATE SET identifier=excluded.identifier,source_url=excluded.source_url,status=excluded.status,reviewer=excluded.reviewer,note=excluded.note,version=excluded.version,updated=excluded.updated',values);audit(db,ident,'review_standard_mapping',before,body.model_dump())
    return {'version':version+1}
class ConceptRelationFields(BaseModel):
    from_id:str
    to_id:str
    kind:Literal['isA','partOf','measures','associatedWith']
    status:Status='needs_review'
    evidence:str=Field(default='',max_length=3000)
    expected_version:int=Field(default=0,ge=0)
def check_hierarchy(db,body,ignore=None):
    if body.from_id==body.to_id:raise HTTPException(422,'서로 다른 정의를 선택하세요.')
    required(db,'concepts',body.from_id);required(db,'concepts',body.to_id)
    if body.status=='approved' and not body.evidence.strip():raise HTTPException(422,'관계 확인 근거를 입력해 주세요.')
    if body.kind not in ('isA','partOf') or body.status!='approved':return
    adj=defaultdict(list)
    for r in db.execute('SELECT * FROM concept_relations WHERE kind=? AND status=?',(body.kind,'approved')):
        if r['id']!=ignore:adj[r['from_id']].append(r['to_id'])
    todo=[body.to_id];seen=set()
    while todo:
        at=todo.pop()
        if at==body.from_id:raise HTTPException(422,'확인된 계층 관계에 순환이 생깁니다.')
        if at in seen:continue
        seen.add(at);todo.extend(adj[at])
@router.get('/concept-relations')
def concept_relations():
    with connect() as db:return [dict(r) for r in db.execute('SELECT * FROM concept_relations ORDER BY updated DESC LIMIT 2000')]
@router.post('/concept-relations',status_code=201)
def relation_create(body:ConceptRelationFields):
    import sqlite3
    ident=uid()
    with connect() as db:
        db.execute('BEGIN IMMEDIATE');check_hierarchy(db,body)
        try:db.execute('INSERT INTO concept_relations VALUES(?,?,?,?,?,?,?,?)',(ident,body.from_id,body.to_id,body.kind,body.status,body.evidence,1,now()))
        except sqlite3.IntegrityError:raise HTTPException(409,'같은 관계가 있습니다. 기존 관계를 수정하세요.')
        audit(db,ident,'create_concept_relation',None,body.model_dump())
    return {'id':ident,'version':1}
@router.put('/concept-relations/{ident}')
def relation_update(ident:str,body:ConceptRelationFields):
    import sqlite3
    with connect() as db:
        db.execute('BEGIN IMMEDIATE');before=required(db,'concept_relations',ident)
        if body.expected_version!=before['version']:raise HTTPException(409,'다른 관계 수정이 저장됐습니다.')
        check_hierarchy(db,body,ident)
        try:db.execute('UPDATE concept_relations SET from_id=?,to_id=?,kind=?,status=?,evidence=?,version=version+1,updated=? WHERE id=?',(body.from_id,body.to_id,body.kind,body.status,body.evidence,now(),ident))
        except sqlite3.IntegrityError:raise HTTPException(409,'같은 관계가 있습니다.')
        audit(db,ident,'update_concept_relation',before,body.model_dump())
    return {'version':before['version']+1}
def semantic_graph(db,page,include_uncertain=False):
    accept=lambda item:item['status']!='rejected' and (include_uncertain or item['status']=='approved')
    elements=[e for e in page['elements'] if e['kind']!='text' and accept(e)];ids={e['id'] for e in elements};nodes=[];edges=[];warnings=[]
    for e in elements:
        nodes.append({'id':'element:'+e['id'],'type':e['kind'],'label':e['label'],'status':e['status'],'bbox':e['bbox'],'evidence':{'page_id':page['id'],'source':e['source']}})
        if e['concept_id']:edges.append({'id':'meaning:'+e['id'],'from':'element:'+e['id'],'to':'concept:'+e['concept_id'],'kind':'denotes','status':e['status'],'evidence':{'page_id':page['id'],'bbox':e['bbox']}})
    for e in page['connections']:
        if e['from_id'] in ids and e['to_id'] in ids and accept(e):edges.append({'id':e['id'],'from':'element:'+e['from_id'],'to':'element:'+e['to_id'],'kind':e['kind'],'status':e['status'],'evidence':{'page_id':page['id'],'source':e['source']}})
    topology=reviewed(db,page)
    for p in topology['payload']['ports']:
        if p['element_id'] not in ids or not accept(p) or p.get('stale'):continue
        ident='port:'+page['id']+':'+p['id'];nodes.append({'id':ident,'type':'Port','label':p['side'],'status':p['status'],'point':p['point'],'evidence':{'page_id':page['id'],'review_version':topology['version']}});edges.append({'id':'owner:'+ident,'from':ident,'to':'element:'+p['element_id'],'kind':'portOf','status':p['status']})
    # Junction observations are independent of equipment adjacency, never used
    # to invent a physical/causal link by ontology serialization.
    for j in topology['payload']['junctions']:
        if accept(j):nodes.append({'id':'junction:'+page['id']+':'+j['id'],'type':'Junction','label':j['kind'],'status':j['status'],'point':j['point'],'evidence':{'page_id':page['id'],'review_version':topology['version']}})
    relations=[dict(r) for r in db.execute('SELECT * FROM concept_relations ORDER BY id LIMIT 2000')];needed={e['concept_id'] for e in elements if e['concept_id']};changed=True
    while changed:
        changed=False
        for r in relations:
            if accept(r) and r['from_id'] in needed and r['to_id'] not in needed:needed.add(r['to_id']);changed=True
    for cid in sorted(needed):
        c=required(db,'concepts',cid);r=db.execute('SELECT * FROM concept_facts WHERE concept_id=?',(cid,)).fetchone();fact=dict(r) if r else None
        nodes.append({'id':'concept:'+cid,'type':'Concept','label':c['name'],'definition':c['definition'],'scope':c['scope'],'unit':c['unit'],'version':c['version'],'standard_mapping':fact,'status':'approved','evidence':{'concept_id':cid}})
    for r in relations:
        if accept(r) and r['from_id'] in needed and r['to_id'] in needed:edges.append({'id':r['id'],'from':'concept:'+r['from_id'],'to':'concept:'+r['to_id'],'kind':r['kind'],'status':r['status'],'evidence':r['evidence']})
    if len(relations)==2000:warnings.append('정의 관계가2000개 상한에 도달했습니다.')
    if not include_uncertain and not elements:warnings.append('확정된 기호가 없습니다. 검토하거나 미확정 포함을 켜세요.')
    return {'schema':'lineage-ontology-1','page_id':page['id'],'nodes':nodes,'edges':edges,'include_uncertain':include_uncertain,'warnings':warnings,'topology_version':topology['version'],'engineering_causality_confirmed':False}
def rdf_graph(graph):
    base='https://lineage.haesoo.dev/vocab/';uri=lambda s:'<urn:lineage:'+quote(str(s),safe=':')+'>';literal=lambda s:json.dumps(str(s),ensure_ascii=False);lines=[]
    for node in graph['nodes']:
        subject=uri(node['id']);lines.extend([f'{subject} <http://www.w3.org/2000/01/rdf-schema#label> {literal(node["label"])} .',f'{subject} <{base}reviewStatus> {literal(node["status"])} .',f'{subject} <http://www.w3.org/ns/prov#wasDerivedFrom> {uri("page:"+graph["page_id"])} .'])
        if node.get('definition'):lines.append(f'{subject} <http://www.w3.org/2004/02/skos/core#definition> {literal(node["definition"])} .')
        fact=node.get('standard_mapping')
        if fact and fact['status']=='approved':
            lines.extend([f'{subject} <{base}standardIdentifier> {literal(fact["identifier"])} .',f'{subject} <http://www.w3.org/2000/01/rdf-schema#seeAlso> {literal(fact["source_url"])} .'])
    for edge in graph['edges']:
        if edge['status']=='approved':lines.append(f'{uri(edge["from"])} <{base+edge["kind"]}> {uri(edge["to"])} .')
    return '\n'.join(lines)+'\n'
@router.get('/pages/{ident}/ontology')
def ontology(ident:str,include_uncertain:bool=False,format:Literal['json','nt']='json'):
    with connect() as db:graph=semantic_graph(db,page_data(db,ident),include_uncertain)
    if format=='nt':return Response(rdf_graph(graph),media_type='application/n-triples',headers={'Content-Disposition':f'attachment; filename="lineage-{ident[:12]}.nt"'})
    return graph

class ImpactFields(BaseModel):
    origins:list[str]=Field(min_length=1,max_length=20)
    depth:int=Field(default=4,ge=1,le=10)
    include_uncertain:bool=False
    include_adjacent:bool=False
def impact_candidates(page,body):
    import heapq
    elements={e['id']:e for e in page['elements'] if e['kind']!='text' and e['status']!='rejected'}
    if any(i not in elements for i in body.origins):raise HTTPException(422,'이 페이지의 기호를 영향 시작점으로 선택하세요.')
    if not body.include_uncertain and any(elements[i]['status']!='approved' for i in body.origins):raise HTTPException(422,'확정되지 않은 시작점입니다. 먼저 검토하거나 미확정 포함을 선택하세요.')
    adj=defaultdict(list)
    for edge in page['connections']:
        if edge['status']=='rejected' or edge['from_id'] not in elements or edge['to_id'] not in elements:continue
        if edge['kind']=='adjacentTo' and not body.include_adjacent:continue
        uncertain=edge['status']!='approved' or any(elements[e]['status']!='approved' for e in (edge['from_id'],edge['to_id']))
        if uncertain and not body.include_uncertain:continue
        adj[edge['from_id']].append((edge['to_id'],edge,uncertain))
        if edge['kind'] in ('connectedTo','adjacentTo'):adj[edge['to_id']].append((edge['from_id'],edge,uncertain))
    heap=[];best={};paths={};origins=set(body.origins)
    for origin in sorted(origins):
        cost=(int(elements[origin]['status']!='approved'),0);best[origin]=cost;paths[origin]=[];heapq.heappush(heap,(*cost,origin))
    truncated=False
    while heap:
        uncertain,hops,at=heapq.heappop(heap)
        if (uncertain,hops)!=best[at]:continue
        if hops>=body.depth:continue
        for target,edge,weak in sorted(adj[at],key=lambda v:(v[0],v[1]['id'])):
            cost=(max(uncertain,int(weak)),hops+1)
            if cost>=best.get(target,(2,999)):continue
            if target not in best and len(best)>=500:truncated=True;continue
            best[target]=cost;paths[target]=paths[at]+[{'connection_id':edge['id'],'from_id':at,'to_id':target,'kind':edge['kind'],'status':edge['status'],'source':edge['source']}];heapq.heappush(heap,(*cost,target))
    return {'schema':'lineage-impact-candidates-1','origins':sorted(origins),'results':[{'element_id':i,'label':elements[i]['label'],'bbox':elements[i]['bbox'],'status':elements[i]['status'],'uncertain':bool(best[i][0]),'hops':best[i][1],'path':paths[i]} for i in sorted(best,key=lambda i:(*best[i],i)) if i not in origins],'depth':body.depth,'truncated':truncated,'direction_policy':'feeds/controls forward; connectedTo undirected; adjacency only when explicitly requested','engineering_causality_confirmed':False,'note':'저장된 관계 경로에 따른 영향 후보이며 물리적 인과·안전성·운전 결과를 확정하지 않습니다.'}
@router.post('/pages/{ident}/impact')
def impact(ident:str,body:ImpactFields):
    with connect() as db:
        page=page_data(db,ident);result=impact_candidates(page,body);result['input_sha256']=sha(page);result['page_id']=ident
        affected=set(body.origins)|{r['element_id'] for r in result['results']};concepts={e['concept_id'] for e in page['elements'] if e['id'] in affected and e['concept_id']}
        refs=[]
        for cid in sorted(concepts):
            for r in db.execute("SELECT e.id,e.label,e.page_id,p.document_id,p.number,d.name FROM elements e JOIN pages p ON e.page_id=p.id JOIN documents d ON p.document_id=d.id WHERE e.concept_id=? AND e.page_id!=? AND e.status='approved' ORDER BY e.id LIMIT 200",(cid,ident)):
                refs.append({**dict(r),'concept_id':cid,'basis':'shared_definition_reference','physical_connection_inferred':False})
        result['related_document_references']=refs[:200];result['reference_limit_reached']=len(refs)>200
        return result

class ComparisonFields(BaseModel):
    base_page_id:str
    current_page_id:str
    transform:list[float]=Field(default_factory=lambda:[1,0,0,0,1,0],min_length=6,max_length=6)
    alignment_confirmed:bool=False
    tolerance:float=Field(default=.005,ge=.00001,le=.1)
    @field_validator('transform')
    @classmethod
    def valid_transform(cls,v):
        if not all(math.isfinite(x) and abs(x)<=20 for x in v) or abs(v[0]*v[4]-v[1]*v[3])<1e-8:raise ValueError('정렬 변환이 유효하지 않습니다.')
        return v
def transform_point(p,m):return [m[0]*p[0]+m[1]*p[1]+m[2],m[3]*p[0]+m[4]*p[1]+m[5]]
def transformed_box(box,m):
    points=[transform_point([x,y],m) for x in (box[0],box[2]) for y in (box[1],box[3])]
    return [min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]
def compare_pages(base,current,body):
    import re
    aa=[e for e in base['elements'] if e['kind']!='text' and e['status']!='rejected'];bb=[{**e,'bbox':transformed_box(e['bbox'],body.transform)} for e in current['elements'] if e['kind']!='text' and e['status']!='rejected']
    if max(len(aa),len(bb))>1500:raise HTTPException(422,'개정 비교는 페이지당1500기호까지 지원합니다.')
    a={e['id']:e for e in aa};b={e['id']:e for e in bb};tags_a=defaultdict(list);tags_b=defaultdict(list)
    for rows,tags in [(aa,tags_a),(bb,tags_b)]:
        for e in rows:
            label=e['label'].strip().casefold()
            if label and not re.match(r'^(요소|학습 요소|벡터 요소)\s*\d+',label):tags[label].append(e['id'])
    paired={};methods={}
    for label in sorted(set(tags_a)&set(tags_b)):
        if len(tags_a[label])==len(tags_b[label])==1:paired[tags_b[label][0]]=tags_a[label][0];methods[tags_b[label][0]]='unique_label'
    remaining=match([e for e in bb if e['id'] not in paired],[e for e in aa if e['id'] not in set(paired.values())],.35)
    paired.update(remaining);methods.update({i:'bbox_iou' for i in remaining});changes=[];ambiguities=[]
    for bid,aid in sorted(paired.items()):
        before,after=a[aid],b[bid];fields={k:{'before':before.get(k),'after':after.get(k)} for k in ('label','kind','concept_id','status') if before.get(k)!=after.get(k)}
        if max(abs(x-y) for x,y in zip(before['bbox'],after['bbox']))>body.tolerance:fields['bbox']={'before':before['bbox'],'after_aligned':after['bbox']}
        alternatives=[g['id'] for g in aa if iou(after['bbox'],g['bbox'])>=.35]
        if len(alternatives)>1 or (methods[bid]=='unique_label' and iou(before['bbox'],after['bbox'])<.1):ambiguities.append({'current_id':bid,'base_id':aid,'alternatives':alternatives,'reason':'복수 후보 또는 위치가 멀어진 태그 대응'})
        if fields:changes.append({'base_id':aid,'current_id':bid,'label':after['label'],'fields':fields,'method':methods[bid],'iou':iou(before['bbox'],after['bbox']),'uncertain':not body.alignment_confirmed or before['status']!='approved' or after['status']!='approved' or len(alternatives)>1})
    def edge_key(edge,mapping=None):
        src=mapping.get(edge['from_id'],'new:'+edge['from_id']) if mapping is not None else edge['from_id'];dst=mapping.get(edge['to_id'],'new:'+edge['to_id']) if mapping is not None else edge['to_id']
        if edge['kind'] in ('connectedTo','adjacentTo'):src,dst=sorted((src,dst))
        return src,dst,edge['kind']
    old_edges={edge_key(e):e for e in base['connections'] if e['status']!='rejected' and e['from_id'] in a and e['to_id'] in a};new_edges={edge_key(e,paired):e for e in current['connections'] if e['status']!='rejected' and e['from_id'] in b and e['to_id'] in b};edge_changes=[{'base':old_edges[k],'current':new_edges[k]} for k in old_edges.keys()&new_edges.keys() if old_edges[k]['status']!=new_edges[k]['status']]
    return {'schema':'lineage-revision-1','base_page_id':base['id'],'current_page_id':current['id'],'transform_current_to_base':body.transform,'alignment_confirmed':body.alignment_confirmed,'tolerance':body.tolerance,'matches':[{'base_id':aid,'current_id':bid,'method':methods[bid],'iou':iou(a[aid]['bbox'],b[bid]['bbox'])} for bid,aid in sorted(paired.items())],'added':[e for e in bb if e['id'] not in paired],'removed':[e for e in aa if e['id'] not in set(paired.values())],'changed':changes,'ambiguous_matches':ambiguities,'connections_added':[new_edges[k] for k in sorted(new_edges.keys()-old_edges.keys())],'connections_removed':[old_edges[k] for k in sorted(old_edges.keys()-new_edges.keys())],'connections_changed':edge_changes,'base_sha256':sha(base),'current_sha256':sha(current),'basis':'normalized2D coordinates + unique labels/IoU; uncertain candidate correspondence, not CAD object identity','engineering_causality_confirmed':False}
@router.post('/comparisons',status_code=201)
def compare(body:ComparisonFields):
    if body.base_page_id==body.current_page_id:raise HTTPException(422,'다른 두 페이지를 선택하세요.')
    with connect() as db:
        base=page_data(db,body.base_page_id);current=page_data(db,body.current_page_id);metas=[db.execute('SELECT * FROM document_meta WHERE document_id=?',(p['document_id'],)).fetchone() for p in (base,current)]
        if any(m is None for m in metas) or metas[0]['project_id']!=metas[1]['project_id']:raise HTTPException(422,'같은 프로젝트에 속한 두 페이지를 선택해 주세요.')
        result=compare_pages(base,current,body)
        result['base_elements']=[e for e in base['elements'] if e['kind']!='text'];result['current_elements']=[e for e in current['elements'] if e['kind']!='text']
        result['topology_changes']=compare_topology(base,current,reviewed(db,base),reviewed(db,current),result,body)
        result.update(base_revision=metas[0]['revision'],current_revision=metas[1]['revision'],project_id=metas[0]['project_id']);ident=uid();db.execute('INSERT INTO comparisons VALUES(?,?,?,?)',(ident,now(),dump(body.model_dump()),dump(result)))
    return {'id':ident,**result}
@router.get('/comparisons/{ident}')
def comparison_get(ident:str):
    with connect() as db:return {'id':ident,**json.loads(required(db,'comparisons',ident)['result'])}
@router.get('/projects/{ident}/comparisons')
def comparison_history(ident:str):
    with connect() as db:
        required(db,'projects',ident)
        return [{'id':r['id'],'created':r['created'],'base_page_id':v['base_page_id'],'current_page_id':v['current_page_id'],'base_revision':v['base_revision'],'current_revision':v['current_revision']} for r in db.execute('SELECT * FROM comparisons ORDER BY created DESC LIMIT 1000') if (v:=json.loads(r['result']))['project_id']==ident][:50]

def compare_topology(base,current,old,new,result,body):
    """Separate reviewed geometry deltas; model observations stay on each page."""
    mapping={m['current_id']:m['base_id'] for m in result['matches']};changes={};tol=body.tolerance
    def valid(rows):return [r for r in rows if r['status']!='rejected' and not r.get('stale')]
    for field in ('ports','junctions','paths'):
        aa=valid(old['payload'][field]) if old['version'] else [];bb=valid(new['payload'][field]) if new['version'] else [];pairs=[];used=set()
        for row in bb:
            def comparable(candidate):
                if field=='ports':return mapping.get(row['element_id'])==candidate['element_id'] and row['side']==candidate['side']
                if field=='junctions':return math.dist(candidate['point'],transform_point(row['point'],body.transform))<=tol*3
                ce=next((e for e in current['connections'] if e['id']==row['connection_id']),None);ae=next((e for e in base['connections'] if e['id']==candidate['connection_id']),None)
                return bool(ce and ae and {mapping.get(ce['from_id']),mapping.get(ce['to_id'])}=={ae['from_id'],ae['to_id']})
            options=[a for a in aa if a['id'] not in used and comparable(a)]
            if not options:continue
            def distance(a):return math.dist(a['point'],transform_point(row['point'],body.transform)) if field!='paths' else abs(len(a['points'])-len(row['points']))
            candidate=min(options,key=lambda a:(distance(a),a['id']));used.add(candidate['id']);pairs.append((candidate,row,len(options)>1))
        changed=[]
        for a,b,ambiguous in pairs:
            ap=a.get('points',[a.get('point')]);bp=[transform_point(p,body.transform) for p in b.get('points',[b.get('point')])]
            if field=='paths' and len(ap)==len(bp) and sum(math.dist(p,q) for p,q in zip(ap,bp[::-1]))<sum(math.dist(p,q) for p,q in zip(ap,bp)):bp=bp[::-1]
            if len(ap)!=len(bp) or max(math.dist(p,q) for p,q in zip(ap,bp))>tol or a['status']!=b['status'] or a.get('kind')!=b.get('kind'):changed.append({'base':a,'current':b,'uncertain':ambiguous or not body.alignment_confirmed})
        changes[field]={'added':[r for r in bb if r['id'] not in {b['id'] for _,b,_ in pairs}],'removed':[r for r in aa if r['id'] not in used],'changed':changed}
    return {'base_version':old['version'],'current_version':new['version'],'reviewed_geometry_only':True,**changes}
