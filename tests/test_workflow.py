import io
import json
import time
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.main import app, password_hash
from backend.extract import make_sample
from backend.db import connect


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('DATA_DIR',str(tmp_path/'data'))
    monkeypatch.setenv('ALLOW_INSECURE_LOCAL','1')
    monkeypatch.delenv('APP_PASSWORD_HASH',raising=False)
    with TestClient(app) as c:yield c


def wait_ready(client,ident):
    deadline=time.time()+15
    while time.time()<deadline:
        response=client.get('/api/documents/'+ident).json()
        if response['status'] in ('ready','failed'):
            assert response['status']=='ready',response
            return response
        time.sleep(.15)
    pytest.fail('extraction timed out')


def image_bytes(format):
    im=Image.new('RGB',(900,500),'white');d=ImageDraw.Draw(im)
    d.ellipse((150,140,250,240),outline='black',width=4)
    d.rectangle((600,120,700,260),outline='black',width=4)
    d.line((250,190,600,190),fill='black',width=3)
    buf=io.BytesIO();im.save(buf,format=format);return buf.getvalue()


@pytest.mark.parametrize('extension,format',[('png','PNG'),('jpg','JPEG')])
def test_raster_upload(client,extension,format):
    response=client.post('/api/documents',files={'file':('diagram.'+extension,image_bytes(format))})
    assert response.status_code==201
    doc=wait_ready(client,response.json()['id']);p=doc['pages'][0]
    assert p['coordinate_system']=='image_pixel'
    image=client.get('/api/pages/'+p['id']+'/image')
    assert image.status_code==200 and image.headers['content-type']=='image/png'
    details=client.get('/api/pages/'+p['id']).json()
    assert details['elements']
    assert all(e['status']=='needs_review' for e in details['elements'])


def test_pdf_multipage_and_rotated_coordinates(client):
    pdf=fitz.open()
    for rotate in (0,90):
        p=pdf.new_page(width=400,height=300);p.insert_text((50,70),'P-101');p.draw_rect(fitz.Rect(100,100,170,190));p.set_rotation(rotate)
    response=client.post('/api/documents',files={'file':('drawing.pdf',pdf.tobytes())});assert response.status_code==201
    doc=wait_ready(client,response.json()['id']);assert len(doc['pages'])==2
    for page in doc['pages']:
        detail=client.get('/api/pages/'+page['id']).json();text=next(e for e in detail['elements'] if e['label']=='P-101')
        assert 0<=text['bbox'][0]<text['bbox'][2]<=1
        assert 0<=text['bbox'][1]<text['bbox'][3]<=1
    assert doc['pages'][1]['width']<doc['pages'][1]['height']


def test_dxf_review_dictionary_connections_persist(client,tmp_path):
    path=tmp_path/'sample.dxf';make_sample(path)
    created=client.post('/api/documents',files={'file':('sample.dxf',path.read_bytes())}).json()
    doc=wait_ready(client,created['id']);pid=doc['pages'][0]['id'];page=client.get('/api/pages/'+pid).json()
    blocks=[e for e in page['elements'] if e['source']=='dxf_block']
    assert len(blocks)==3 and len(page['connections'])==2
    assert any(e['label']=='V-201' for e in page['elements'])
    concept=client.post('/api/concepts',json={'name':'냉각수 밸브','definition':'냉각수의 유량을 조절하는 밸브','scope':'공정 A','unit':'m3/h'})
    assert concept.status_code==201;cid=concept.json()['id']
    duplicate=client.post('/api/concepts',json={'name':'냉각수 밸브','definition':'중복','scope':'공정 A'});assert duplicate.status_code==409
    valve=next(e for e in blocks if e['label']=='VALVE')
    body={**valve,'label':'V-201','kind':'valve','status':'approved','concept_id':cid}
    assert client.put('/api/elements/'+valve['id'],json=body).status_code==200
    manual=client.post('/api/pages/'+pid+'/elements',json={'label':'J-01','kind':'junction','bbox':[.1,.2,.15,.25]});assert manual.status_code==201
    relation=client.post('/api/pages/'+pid+'/connections',json={'from_id':valve['id'],'to_id':manual.json()['id'],'kind':'controls'});assert relation.status_code==201
    rid=relation.json()['id']
    assert client.put('/api/connections/'+rid,json={'from_id':manual.json()['id'],'to_id':valve['id'],'kind':'connectedTo','status':'approved'}).status_code==200
    export=client.get('/api/documents/'+doc['id']+'/export').json()
    assert export['schema_version']=='lineage-0.1' and export['concepts'][0]['id']==cid
    assert any(e['label']=='V-201' and e['concept_id']==cid and e['status']=='approved' for e in export['pages'][0]['elements'])
    assert client.get('/api/history/'+valve['id']).json()[0]['action']=='update_element'
    with connect() as db:assert db.execute('SELECT label FROM elements WHERE id=?',(valve['id'],)).fetchone()['label']=='V-201'
    assert client.delete('/api/elements/'+manual.json()['id']).status_code==200
    assert not any(e['id']==rid for e in client.get('/api/pages/'+pid).json()['connections'])
    assert client.get('/api/documents/'+doc['id']+'/original').content==path.read_bytes()


def test_validation_and_cross_page_edges(client):
    assert client.post('/api/documents',files={'file':('code.svg',b'<svg/>')}).status_code==415
    assert client.post('/api/documents',files={'file':('empty.png',b'')}).status_code==400
    assert client.post('/api/concepts',json={'name':' ','definition':'x','scope':'a'}).status_code==422
    a=wait_ready(client,client.post('/api/examples').json()['id'])['pages'][0]['id']
    b=wait_ready(client,client.post('/api/examples').json()['id'])['pages'][0]['id']
    ea=client.get('/api/pages/'+a).json()['elements'][0]['id'];eb=client.get('/api/pages/'+b).json()['elements'][0]['id']
    assert client.post('/api/pages/'+a+'/connections',json={'from_id':ea,'to_id':eb}).status_code==422
    assert client.post('/api/pages/'+a+'/connections',json={'from_id':ea,'to_id':ea}).status_code==422
    assert client.post('/api/pages/'+a+'/elements',json={'label':'bad','kind':'x','bbox':[.5,.5,.1,.2]}).status_code==422
    assert client.post('/api/documents/'+a+'/retry').status_code==404


def test_bad_file_job_can_retry_without_overwrite(client):
    ident=client.post('/api/documents',files={'file':('bad.png',b'not an image')}).json()['id']
    for _ in range(50):
        doc=client.get('/api/documents/'+ident).json()
        if doc['status']=='failed':break
        time.sleep(.1)
    assert doc['status']=='failed'
    assert client.post('/api/documents/'+ident+'/retry').status_code==200


def test_auth_csrf_and_cookie(tmp_path,monkeypatch):
    monkeypatch.setenv('DATA_DIR',str(tmp_path))
    monkeypatch.setenv('APP_PASSWORD_HASH',password_hash('test-password'))
    monkeypatch.setenv('COOKIE_SECURE','0')
    monkeypatch.delenv('ALLOW_INSECURE_LOCAL',raising=False)
    with TestClient(app) as c:
        assert c.get('/api/documents').status_code==401
        assert not c.get('/api/session').json()['authenticated']
        assert c.post('/api/login',json={'password':'wrong'}).status_code==401
        assert c.post('/api/login',json={'password':'test-password'},headers={'Origin':'https://evil.invalid'}).status_code==403
        response=c.post('/api/login',json={'password':'test-password'});assert response.status_code==200
        assert 'HttpOnly' in response.headers['set-cookie']
        assert c.get('/api/documents').status_code==200
        assert c.post('/api/concepts',json={'name':'a','definition':'b','scope':'c'}).status_code==403
        csrf=response.json()['csrf']
        assert c.post('/api/concepts',json={'name':'a','definition':'b','scope':'c'},headers={'X-CSRF-Token':csrf}).status_code==201
        assert c.post('/api/logout',headers={'X-CSRF-Token':csrf}).status_code==200
        assert c.get('/api/documents').status_code==401


def test_restart_preserves_review_and_dictionary(tmp_path,monkeypatch):
    monkeypatch.setenv('DATA_DIR',str(tmp_path));monkeypatch.setenv('ALLOW_INSECURE_LOCAL','1');monkeypatch.delenv('APP_PASSWORD_HASH',raising=False)
    with TestClient(app) as c:
        doc=wait_ready(c,c.post('/api/examples').json()['id'])
        pid=doc['pages'][0]['id'];e=c.get('/api/pages/'+pid).json()['elements'][0]
        cid=c.post('/api/concepts',json={'name':'테스트 정의','definition':'재시작 보존 검증','scope':'테스트'}).json()['id']
        assert c.put('/api/elements/'+e['id'],json={**e,'concept_id':cid,'status':'approved'}).status_code==200
    with TestClient(app) as c:
        persisted=c.get('/api/pages/'+pid).json()['elements'][0]
        assert persisted['concept_id']==cid and persisted['status']=='approved'
        assert c.get('/api/concepts').json()[0]['id']==cid


def test_recognition_evidence_survives_review_and_export(client,tmp_path,monkeypatch):
    from backend import topology
    def recognized(elements,*args):
        symbols=[e for e in elements if e['kind']!='text'];a,b=symbols[:2];port={'point':[.3,.4],'pixel':[30,40],'side':'right','score':.8,'status':'needs_review'}
        return [],{'paths':[{'from_id':a['id'],'to_id':b['id'],'points':[[.3,.4],[.6,.4]],'status':'needs_review'}],'ports':{a['id']:[{k:v for k,v in port.items() if k!='pixel'}]},'port_count':1,'junctions':[{'point':[.4,.4],'type':'tee','status':'needs_review'}]},{a['id']:[port]}
    monkeypatch.setattr(topology,'route',recognized)
    path=tmp_path/'ports.dxf';make_sample(path)
    doc=wait_ready(client,client.post('/api/documents',files={'file':('ports.dxf',path.read_bytes())}).json()['id']);pid=doc['pages'][0]['id'];page=client.get('/api/pages/'+pid).json()
    target=next(e for e in page['elements'] if e['properties'].get('ports'))
    original=target['properties'];evidence=page['transform']['connection_inference']
    assert client.put('/api/elements/'+target['id'],json={**target,'status':'approved','properties':{}}).status_code==200
    fresh=client.get('/api/pages/'+pid).json();stored=next(e for e in fresh['elements'] if e['id']==target['id'])
    assert stored['properties']==original and stored['properties']['ports'][0]['status']=='needs_review'
    assert fresh['transform']['connection_inference']==evidence
    exported=client.get('/api/documents/'+doc['id']+'/export').json()['pages'][0]
    assert exported['transform']['connection_inference']==evidence
    assert any(e['properties']==original for e in exported['elements'])
