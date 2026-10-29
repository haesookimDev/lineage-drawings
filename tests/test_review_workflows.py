"""Human corrections must not silently become model predictions or leak splits."""
import copy,json,time
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.db import connect
from backend.workflows import impact_candidates,ImpactFields,compare_pages,ComparisonFields

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('DATA_DIR',str(tmp_path));monkeypatch.setenv('ALLOW_INSECURE_LOCAL','1');monkeypatch.delenv('APP_PASSWORD_HASH',raising=False)
    with TestClient(app) as c:yield c
def ready(c,doc):
    for _ in range(100):
        d=c.get('/api/documents/'+doc).json()
        if d['status'] in ('ready','failed'):
            assert d['status']=='ready',d
            return c.get('/api/pages/'+d['pages'][0]['id']).json()
        time.sleep(.1)
    pytest.fail('Worker did not finish')
def sample(c):
    doc=c.post('/api/examples').json()['id'];return doc,ready(c,doc)
def project(c,doc,split='test',name='Project'):
    p=c.post('/api/projects',json={'name':name,'split':split});assert p.status_code==201,p.text
    pid=p.json()['id'];assert c.put('/api/documents/'+doc+'/project',json={'project_id':pid,'revision':'A'}).status_code==200
    return pid
def approve(c,page):
    for e in page['elements']:
        if e['kind']!='text':assert c.put('/api/elements/'+e['id'],json={**e,'status':'approved'}).status_code==200
    for e in page['connections']:assert c.put('/api/connections/'+e['id'],json={**e,'status':'approved'}).status_code==200
def test_topology_version_conflict_restore_and_immutable_observations(client):
    c=client;doc,p=sample(c);endpoint='/api/pages/'+p['id'];before=copy.deepcopy(p['transform']);edge=p['connections'][0]
    ports=[{'id':'p1','element_id':edge['from_id'],'point':[.2,.3],'side':'right','status':'approved'},{'id':'p2','element_id':edge['to_id'],'point':[.7,.3],'side':'left','status':'approved'}]
    body={'expected_version':0,'reviewer':'test reviewer','ports':ports,'junctions':[{'id':'j','point':[.5,.3],'kind':'plain_cross','status':'approved'}],'paths':[{'id':'route','connection_id':edge['id'],'points':[[.2,.3],[.7,.3]],'from_port_id':'p1','to_port_id':'p2','status':'approved'}]}
    r=c.put(endpoint+'/topology-review',json=body);assert r.status_code==200,r.text;assert r.json()['version']==1
    assert c.put(endpoint+'/topology-review',json=body).status_code==409
    bad=copy.deepcopy(body);bad['expected_version']=1;bad['paths'][0]['from_port_id']='p2';assert c.put(endpoint+'/topology-review',json=bad).status_code==422
    bad=copy.deepcopy(body);bad['expected_version']=1;bad['ports'][0]['point']=[1.1,.5];assert c.put(endpoint+'/topology-review',json=bad).status_code==422
    changed=copy.deepcopy(body);changed['expected_version']=1;changed['junctions'][0]['kind']='tee';assert c.put(endpoint+'/topology-review',json=changed).json()['version']==2
    restored=c.post(endpoint+'/topology-restore',json={'expected_version':2,'version':1,'reviewer':'restorer'});assert restored.status_code==200,restored.text;assert restored.json()['payload']['junctions'][0]['kind']=='plain_cross'
    assert len(c.get(endpoint+'/topology-history').json())==3
    assert c.get(endpoint).json()['transform']==before
    assert c.get('/api/history/'+p['id']).json()[0]['action']=='review_topology'
def test_port_cannot_reference_other_page(client):
    c=client;_,p=sample(c);_,q=sample(c)
    bad={'expected_version':0,'reviewer':'Reviewer','ports':[{'id':'foreign','element_id':q['elements'][0]['id'],'point':[.1,.2],'side':'left'}]}
    assert c.put('/api/pages/'+p['id']+'/topology-review',json=bad).status_code==422
def test_immutable_prediction_evaluation_and_project_split_lock(client):
    c=client;doc,p=sample(c);pid=project(c,doc)
    snapshot={'reviewer':'AI pilot','basis':'ai_assisted','confirmed':True}
    assert c.post('/api/pages/'+p['id']+'/annotations',json=snapshot).status_code==422
    approve(c,p)
    extra=c.post('/api/pages/'+p['id']+'/elements',json={'label':'MISSING-4','kind':'valve','bbox':[.01,.8,.05,.9],'status':'approved'}).json()['id']
    source=next(e for e in p['elements'] if e['kind']!='text')
    assert c.post('/api/pages/'+p['id']+'/connections',json={'from_id':source['id'],'to_id':extra,'status':'approved'}).status_code==201
    annotation=c.post('/api/pages/'+p['id']+'/annotations',json=snapshot);assert annotation.status_code==201,annotation.text;aid=annotation.json()['id']
    result=c.post('/api/evaluations',json={'annotation_ids':[aid],'split':'test'});assert result.status_code==201,result.text;data=result.json();assert data['uses_immutable_initial_predictions']
    assert data['aggregate']['symbols']['tp']==3 and data['aggregate']['symbols']['fn']==1
    assert data['aggregate']['connections']['tp']==2 and data['aggregate']['connections']['fn']==1
    assert c.get('/api/evaluations/'+data['id']).json()['aggregate']==data['aggregate']
    assert c.put('/api/projects/'+pid,json={'name':'Project','split':'training'}).status_code==409
    assert c.post('/api/evaluations',json={'annotation_ids':[aid],'split':'training'}).status_code==422
    exported=c.get('/api/datasets/export?split=test').json();assert len(exported['annotations'])==1 and exported['expert_certified'] is False
    # A new inference of the exact source is allowed, without using manual edits.
    original=c.get('/api/documents/'+doc+'/original').content;newdoc=c.post('/api/documents',files={'file':('copy.dxf',original)}).json()['id'];q=ready(c,newdoc)
    c.put('/api/documents/'+newdoc+'/project',json={'project_id':pid,'revision':'B'})
    r=c.post('/api/evaluations',json={'annotation_ids':[aid],'prediction_pages':{aid:q['id']}});assert r.status_code==201,r.text;assert r.json()['aggregate']['symbols']['fn']==1
    with connect() as db:db.execute('DELETE FROM analysis_runs WHERE page_id=?',(q['id'],))
    assert c.post('/api/evaluations',json={'annotation_ids':[aid],'prediction_pages':{aid:q['id']}}).status_code==409
def test_standard_provenance_hierarchy_and_rdf(client):
    c=client;_,p=sample(c);approve(c,p)
    defs=[]
    for name in ['Type "A"','Type B']:
        defs.append(c.post('/api/concepts',json={'name':name,'definition':'definition\nwith "quote"','scope':'tests'}).json()['id'])
    e=next(e for e in p['elements'] if e['kind']!='text');c.put('/api/elements/'+e['id'],json={**e,'status':'approved','concept_id':defs[0]})
    rel={'from_id':defs[0],'to_id':defs[1],'kind':'isA','status':'approved','evidence':'Source reviewed'}
    assert c.post('/api/concept-relations',json=rel).status_code==201
    assert c.post('/api/concept-relations',json={**rel,'from_id':defs[1],'to_id':defs[0]}).status_code==422
    path='/api/concepts/'+defs[0]+'/standard';body={'identifier':'TEST-ONLY-123','source_url':'https://example.org/spec','reviewer':'Reviewer','status':'approved','expected_version':0}
    assert c.put(path,json={**body,'source_url':''}).status_code==422
    assert c.put(path,json=body).status_code==200
    assert c.put(path,json=body).status_code==409
    graph=c.get('/api/pages/'+p['id']+'/ontology').json();assert any(e['kind']=='isA' for e in graph['edges']);assert any(n['type']=='Concept' for n in graph['nodes']);assert graph['engineering_causality_confirmed'] is False
    text=c.get('/api/pages/'+p['id']+'/ontology?format=nt').text;assert 'TEST-ONLY-123' in text and '\\nwith \\"quote\\"' in text
def element(ident,box,status='approved'):
    return {'id':ident,'label':ident,'bbox':box,'kind':'equipment','status':status,'concept_id':None,'source':'manual'}
def edge(ident,a,b,kind='connectedTo',status='approved'):
    return {'id':ident,'from_id':a,'to_id':b,'kind':kind,'status':status,'source':'manual'}
def test_impact_prefers_confirmed_path_and_obeys_direction():
    p={'elements':[element(k,[0,0,.1,.1]) for k in 'abcd'],'connections':[edge('ab','a','b','feeds'),edge('bc','b','c','controls'),edge('ac','a','c','connectedTo','needs_review'),edge('cd','c','d','adjacentTo')]}
    r=impact_candidates(p,ImpactFields(origins=['a'],include_uncertain=True));by={e['element_id']:e for e in r['results']};assert by['c']['hops']==2 and by['c']['uncertain'] is False and 'd' not in by
    assert not impact_candidates(p,ImpactFields(origins=['b']))['results'][0]['element_id']=='a'
    p['connections'].append(edge('cycle','c','a','controls'));assert len(impact_candidates(p,ImpactFields(origins=['a'],depth=10))['results'])==2
def test_revision_alignment_direction_change_and_snapshot(client):
    a={'id':'base','elements':[element('P-101',[.1,.1,.2,.2]),element('V-2',[.4,.4,.5,.5])],'connections':[edge('a','P-101','V-2','feeds')]}
    b={'id':'current','elements':[{**element('p-new',[.2,.1,.3,.2]),'label':'P-101'},element('NEW-3',[.7,.7,.8,.8])],'connections':[]}
    result=compare_pages(a,b,ComparisonFields(base_page_id='base',current_page_id='current',transform=[1,0,-.1,0,1,0],alignment_confirmed=True))
    assert len(result['added'])==len(result['removed'])==1 and not result['changed'] and len(result['connections_removed'])==1
    c=client;d,p=sample(c);pid=project(c,d);qdoc,q=sample(c);c.put('/api/documents/'+qdoc+'/project',json={'project_id':pid,'revision':'B'})
    r=c.post('/api/comparisons',json={'base_page_id':p['id'],'current_page_id':q['id'],'alignment_confirmed':True});assert r.status_code==201,r.text;result=r.json();assert not result['added'] and not result['removed'];assert c.get('/api/comparisons/'+result['id']).json()['base_sha256']==result['base_sha256']
    assert c.post('/api/comparisons',json={'base_page_id':p['id'],'current_page_id':q['id'],'transform':[0,0,0,0,0,0]}).status_code==422

def test_stale_review_geometry_excluded_from_graph_and_versioned(client):
    c=client;doc,p=sample(c);approve(c,p);e=next(e for e in p['elements'] if e['kind']!='text');path='/api/pages/'+p['id']
    body={'expected_version':0,'reviewer':'AI test','ports':[{'id':'manual-port','element_id':e['id'],'point':[.2,.3],'side':'right','status':'approved'}]}
    assert c.put(path+'/topology-review',json=body).status_code==200
    assert any(n['type']=='Port' for n in c.get(path+'/ontology').json()['nodes'])
    moved={**e,'bbox':[.01,.01,.05,.05],'status':'approved'};assert c.put('/api/elements/'+e['id'],json=moved).status_code==200
    r=c.get(path+'/topology-review').json();assert r['payload']['ports'][0]['stale']
    assert not any(n['type']=='Port' for n in c.get(path+'/ontology').json()['nodes'])
    body['expected_version']=1;body['ports'][0]['point']=[.05,.03];assert c.put(path+'/topology-review',json=body).status_code==200
    assert not c.get(path+'/topology-review').json()['payload']['ports'][0]['stale']

def test_revision_tracks_reviewed_geometry_and_immutable_history(client):
    c=client;d,p=sample(c);pid=project(c,d);qd,q=sample(c);c.put('/api/documents/'+qd+'/project',json={'project_id':pid,'revision':'B'})
    for page,point in [(p,[.2,.3]),(q,[.2,.4])]:
        e=next(e for e in page['elements'] if e['kind']!='text')
        assert c.put('/api/pages/'+page['id']+'/topology-review',json={'expected_version':0,'reviewer':'AI test','ports':[{'id':'port','element_id':e['id'],'point':point,'side':'right','status':'approved'}]}).status_code==200
    comparison=c.post('/api/comparisons',json={'base_page_id':p['id'],'current_page_id':q['id'],'alignment_confirmed':True}).json()
    assert len(comparison['topology_changes']['ports']['changed'])==1
    history=c.get('/api/projects/'+pid+'/comparisons').json();assert history[0]['id']==comparison['id']
    assert c.get('/api/comparisons/'+comparison['id']).json()==comparison

def test_shared_definition_reference_does_not_invent_physical_connection(client):
    c=client;_,p=sample(c);_,q=sample(c);approve(c,p);approve(c,q)
    cid=c.post('/api/concepts',json={'name':'Shared meaning','definition':'For API test','scope':'test'}).json()['id']
    for page in (p,q):
        e=next(e for e in page['elements'] if e['kind']!='text');assert c.put('/api/elements/'+e['id'],json={**e,'status':'approved','concept_id':cid}).status_code==200
    origin=next(e for e in p['elements'] if e['kind']!='text')['id'];r=c.post('/api/pages/'+p['id']+'/impact',json={'origins':[origin]}).json()
    assert len(r['related_document_references'])==1
    assert r['related_document_references'][0]['physical_connection_inferred'] is False
    assert all(e['element_id'] not in {e['id'] for e in q['elements']} for e in r['results'])
