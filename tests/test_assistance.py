import copy,json
import pytest
from tests.test_review_workflows import client,sample,project
from backend.db import connect

def pack(c,page):
 r=c.get('/api/pages/'+page['id']+'/correction-pack');assert r.status_code==200,r.text;return r.json()
def body(p,proposal):return {'schema':'lineage-correction-proposals-1','snapshot_sha256':p['snapshot_sha256'],'author_model':'offline contract fixture, no actual LLM call','proposals':[proposal]}
def rename(p,e,label='Corrected tag'):
 return {'operation':'rename','target_id':e['id'],'label':label,'evidence_ids':['element:'+e['id']],'reason':'Source image review fixture','confidence':.8}
def test_llm_contract_rejects_fabricated_or_stale_evidence_and_keeps_raw_prediction(client):
 c=client;_,p=sample(c);path='/api/pages/'+p['id'];initial=copy.deepcopy(p);e=p['elements'][0];context=pack(c,p)
 assert context['external_api_called'] is False
 item=rename(context,e)
 assert c.post(path+'/llm-proposals',json=body(context,{**item,'evidence_ids':['element:invented']})).status_code==422
 assert c.post(path+'/llm-proposals',json=body(context,{**item,'standard_id':'ISO-FAKE'})).status_code==422
 run=c.post(path+'/llm-proposals',json=body(context,item));assert run.status_code==201,run.text
 assert c.get(path).json()==initial
 assert run.json()['metadata']['actual_llm_execution_verified'] is False
 proposal=run.json()['proposal_ids'][0]
 updated={**e,'label':'Human changed first'};c.put('/api/elements/'+e['id'],json=updated)
 assert c.post('/api/proposals/'+proposal+'/review',json={'action':'apply','expected_version':1,'reviewer':'Reviewer'}).status_code==409
 assert c.post(path+'/llm-proposals',json=body(context,item)).status_code==409
 fresh=pack(c,p);run=c.post(path+'/llm-proposals',json=body(fresh,rename(fresh,updated)));assert run.status_code==201,run.text
 ident=run.json()['proposal_ids'][0];r=c.post('/api/proposals/'+ident+'/review',json={'action':'apply','expected_version':1,'reviewer':'Reviewer'});assert r.status_code==200,r.text
 after=c.get(path).json();stored=next(v for v in after['elements'] if v['id']==e['id']);assert stored['label']=='Corrected tag' and stored['status']=='needs_review' and stored['properties']==e['properties'];assert after['transform']==initial['transform']
 with connect() as db:
  original=json.loads(db.execute('SELECT payload FROM analysis_runs WHERE page_id=?',(p['id'],)).fetchone()[0]);assert original['elements'][0]['label']==initial['elements'][0]['label']
 assert c.post('/api/proposals/'+ident+'/review',json={'action':'apply','expected_version':1,'reviewer':'Reviewer'}).status_code==409

def test_llm_cannot_connect_without_matching_observed_path_or_cross_page(client):
 c=client;_,p=sample(c);_,q=sample(c);context=pack(c,p);path='/api/pages/'+p['id'];e=p['elements'][0];item=rename(context,e);item['target_id']=q['elements'][0]['id'];assert c.post(path+'/llm-proposals',json=body(context,item)).status_code==422
 bad={'operation':'connect','from_id':p['elements'][0]['id'],'to_id':p['elements'][1]['id'],'path_id':'path:fake','evidence_ids':['region:0'],'reason':'Unsupported','confidence':.9};assert c.post(path+'/llm-proposals',json=body(context,bad)).status_code==422
 bad={'operation':'add_symbol','bbox':[.99,.99,1.,1.],'label':'Imagined','evidence_ids':['region:0'],'reason':'No image evidence','confidence':.99};assert c.post(path+'/llm-proposals',json=body(context,bad)).status_code==422

def test_legend_project_scope_versions_and_proposal_only_matching(client):
 c=client;doc,p=sample(c);pid=project(c,doc);_,q=sample(c);e=next(v for v in p['elements'] if v['kind']!='text');fields={'bbox':e['bbox'],'name':'Fixture symbol','mode':'outline','reviewer':'Source reviewer','status':'approved'}
 r=c.post('/api/pages/'+p['id']+'/legends',json=fields);assert r.status_code==201,r.text;lid=r.json()['id']
 assert c.get('/api/projects/'+pid+'/legends').json()[0]['id']==lid
 assert c.post('/api/pages/'+q['id']+'/legend-proposals',json={'template_ids':[lid]}).status_code==422
 before=c.get('/api/pages/'+p['id']).json();r=c.post('/api/pages/'+p['id']+'/legend-proposals',json={'template_ids':[lid],'threshold':.8});assert r.status_code==201,r.text;assert c.get('/api/pages/'+p['id']).json()==before
 assert c.put('/api/legends/'+lid,json={**fields,'expected_version':0}).status_code==409
 assert c.put('/api/legends/'+lid,json={**fields,'status':'rejected','expected_version':1}).status_code==200
 assert c.post('/api/pages/'+p['id']+'/legend-proposals',json={'template_ids':[lid]}).status_code==422

def test_regional_ocr_maps_coordinates_and_requires_review(client,monkeypatch):
 c=client;_,p=sample(c);path='/api/pages/'+p['id'];initial=copy.deepcopy(p)
 monkeypatch.setattr('backend.extract.ocr',lambda image,warnings:[{'label':'REGION-TAG','bbox':[.1,.2,.5,.6],'properties':{'ocr_score':95}}])
 r=c.post(path+'/ocr-proposals',json={'regions':[{'bbox':[.1,.1,.4,.4]}]});assert r.status_code==201,r.text;ident=r.json()['proposal_ids'][0];proposal=c.get(path+'/assistance').json()['proposals'][0];w,h=p['width'],p['height'];a,b=int(.1*w),int(.1*h);import math;c1,d=math.ceil(.4*w),math.ceil(.4*h);assert proposal['payload']['bbox']==pytest.approx([(a+.1*(c1-a))/w,(b+.2*(d-b))/h,(a+.5*(c1-a))/w,(b+.6*(d-b))/h]);assert c.get(path).json()==initial
 r=c.post('/api/proposals/'+ident+'/review',json={'action':'apply','expected_version':1,'reviewer':'Reviewer'});assert r.status_code==200,r.text
 fresh=c.get(path).json();text=next(e for e in fresh['elements'] if e['label']=='REGION-TAG');assert text['kind']=='text' and text['status']=='needs_review';assert fresh['transform']==initial['transform']

def test_proposal_evaluation_uses_original_snapshot_and_does_not_apply(client):
 from tests.test_review_workflows import approve,ready
 c=client;doc,p=sample(c);pid=project(c,doc);approve(c,p);ann=c.post('/api/pages/'+p['id']+'/annotations',json={'reviewer':'Fixture','basis':'ai_assisted','confirmed':True}).json()['id'];raw=c.get('/api/documents/'+doc+'/original').content;newdoc=c.post('/api/documents',files={'file':('copy.dxf',raw)}).json()['id'];q=ready(c,newdoc);c.put('/api/documents/'+newdoc+'/project',json={'project_id':pid,'revision':'B'});context=pack(c,q);item=rename(context,q['elements'][0]);run=c.post('/api/pages/'+q['id']+'/llm-proposals',json=body(context,item)).json()['id'];before=c.get('/api/pages/'+q['id']).json();r=c.post('/api/pages/'+q['id']+'/proposal-evaluation',json={'annotation_id':ann,'run_id':run});assert r.status_code==201,r.text;result=r.json();assert result['dry_run'] and result['uses_immutable_initial_predictions'] and result['actual_llm_execution_verified'] is False;assert result['before']['symbols']==result['after']['symbols'];assert c.get('/api/pages/'+q['id']).json()==before
 c.put('/api/elements/'+q['elements'][0]['id'],json={**q['elements'][0],'label':'Human edited'});context=pack(c,q);run=c.post('/api/pages/'+q['id']+'/llm-proposals',json=body(context,rename(context,{**q['elements'][0],'label':'Human edited'}))).json()['id'];assert c.post('/api/pages/'+q['id']+'/proposal-evaluation',json={'annotation_id':ann,'run_id':run}).status_code==409

def test_raster_region_coordinates_use_actual_crop_pixels():
 from PIL import Image
 from backend.region_ocr import read_regions
 image=Image.new('RGB',(101,103),'white')
 words=read_regions(image,[],lambda im,w,d:[{'label':'T','bbox':[0,0,1,1],'properties':{}}],[[.105,.113,.405,.413]])
 assert words[0]['bbox']==pytest.approx([10/101,11/103,41/101,43/103])
