"""Sparse recognition and context-gated additions retain explicit review state."""
import json
import numpy as np
import pytest
from PIL import Image,ImageDraw
from backend import sparse_topology as sparse,topology_context as context,geometry
from backend.extract import infer_connections


def symbol(k,b):return {'id':k,'kind':'unknown','bbox':b,'properties':{}}

@pytest.mark.parametrize('kind,expected',[('plain',2),('tee',3),('dot',6)])
def test_sparse_router_obeys_contact_geometry(monkeypatch,kind,expected):
    im=Image.new('RGB',(256,256),'white');d=ImageDraw.Draw(im)
    d.line((40,128,216,128),fill='black',width=2);d.line((128,40,128,128 if kind=='tee' else 216),fill='black',width=2)
    if kind=='dot':d.ellipse((122,122,134,134),fill='black')
    els=[symbol('a',[.05,.45,40/256,.55]),symbol('b',[216/256,.45,.95,.55]),symbol('c',[.45,.05,.55,40/256])]
    if kind!='tee':els.append(symbol('d',[.45,216/256,.55,.95]))
    monkeypatch.setattr(sparse,'model',lambda:(object(),object(),{}));monkeypatch.setattr(sparse,'features',lambda g,i,p:np.array([q['geometry'] for q in p],np.float32).reshape(-1,6))
    def votes(m,x,classes):
        if classes==2:return np.tile([0.,1.],(len(x),1))
        out=np.zeros((len(x),4),np.float32)
        for j,g in enumerate(x):
            n=round(g[0]*4);label=0 if n<3 else 2 if g[1] else 1 if n==3 else 3;out[j,label]=1.
        return out
    monkeypatch.setattr(sparse,'votes',votes);pairs,details,ports=sparse.route(els,[],256,256,im)
    assert len(pairs)==expected
    assert all(p['status']=='needs_review' and len(p['points'])<=128 for p in details['paths'])
    assert any(p['type']==('plain_cross' if kind=='plain' else kind) for p in details['junctions'])


def test_two_arm_bend_is_not_displayed_as_junction(monkeypatch):
    im=Image.new('RGB',(128,128),'white');d=ImageDraw.Draw(im);d.line((20,60,60,60,60,110),fill='black',width=2)
    gray,ink=context.prepare(im);rows=sparse.intersections([[20,60,60,60],[60,60,60,110]],[[],[]],ink)
    assert rows and rows[0]['arms']==2


def test_vectorized_context_matches_scalar_reference():
    rng=np.random.default_rng(8);ink=(rng.random((70,90))>.8).astype(np.uint8)*255;pts=rng.uniform([0,0],[89,69],(30,2))
    assert np.array_equal(np.array([context.rays(ink,*p) for p in pts]),context.ray_batch(ink,pts))


def test_hybrid_adds_only_scored_paths_and_preserves_old(monkeypatch):
    from backend import topology_context
    els=[symbol(k,[.1+i*.2,.2,.2+i*.2,.3]) for i,k in enumerate('abc')];im=Image.new('RGB',(100,100),'white')
    monkeypatch.setattr(geometry,'_trace_validated',lambda *a:[('a','b')]);monkeypatch.setattr(topology_context,'model',lambda:(object(),object(),{'extra_edge_threshold':.7}))
    paths=[{'from_id':a,'to_id':b,'points':[[.1,.2],[.4,.2]],'score':score,'status':'needs_review'} for a,b,score in [('b','c',.9),('a','c',.6)]]
    monkeypatch.setattr(sparse,'route',lambda *a:([],{'recognition_algorithm':'sparse-line-port-v8','paths':paths},{}))
    details={};edges=infer_connections(els,[],100,100,im,details)
    assert {frozenset((e['from_id'],e['to_id'])) for e in edges}=={frozenset(('a','b')),frozenset(('b','c'))}
    assert details['context_supplemental_connections']==1
    assert all(e['status']=='needs_review' for e in edges)
    assert next(e for e in edges if {e['from_id'],e['to_id']}=={'b','c'})['source']=='recognized_path_candidate'


def test_context_integrity_check(tmp_path,monkeypatch):
    (tmp_path/'topology-context.json').write_text(json.dumps({'port_file':'p.xml.gz','port_sha256':'bad'}));(tmp_path/'p.xml.gz').write_bytes(b'bad')
    monkeypatch.setattr(context,'DIR',tmp_path);context.model.cache_clear()
    try:
        with pytest.raises(ValueError):context.model()
    finally:context.model.cache_clear()
