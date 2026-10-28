"""Recognition must add reviewable evidence without changing validated edges."""
import copy
import cv2
import numpy as np
import pytest
from PIL import Image,ImageDraw
from backend import geometry,topology,topology_features
from backend.extract import infer_connections


def element(ident,box):return {'id':ident,'kind':'unknown','bbox':box,'properties':{}}

def fixture(kind):
    image=Image.new('RGB',(256,256),'white');d=ImageDraw.Draw(image)
    lines=[[40,128,216,128],[128,40,128,216 if kind!='tee' else 128]]
    for line in lines:d.line(tuple(line),fill='black',width=2)
    if kind=='dot':d.ellipse((122,122,134,134),fill='black')
    els=[element('a',[.05,.45,40/256,.55]),element('b',[216/256,.45,.95,.55]),element('c',[.45,.05,.55,40/256])]
    if kind!='tee':els.append(element('d',[.45,216/256,.55,.95]))
    maps=np.zeros((6,256,256),np.uint8);maps[:3]=255;maps[3 if kind=='tee' else 4 if kind=='dot' else 5]=255
    return image,lines,els,maps

@pytest.mark.parametrize('kind,count',[('tee',3),('dot',6),('plain',2)])
def test_explicit_ports_and_crossing_policy(monkeypatch,kind,count):
    image,lines,els,maps=fixture(kind)
    monkeypatch.setattr(topology,'model',lambda:(object(),{}))
    gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
    monkeypatch.setattr(topology,'line_segments',lambda *a:(lines,topology.line_ink(gray)))
    pairs,details,_=topology.route(els,lines,256,256,image,recognized=(maps,{}))
    assert len(pairs)==count
    assert details['port_count']>=len(els)
    assert all(len(p['points'])<=128 for p in details['paths'])
    assert any(j['type']==kind.replace('plain','plain_cross') for j in details['junctions'])
    assert all(j['status']=='needs_review' for j in details['junctions'])


def test_advisory_recognition_never_adds_or_approves_edges(monkeypatch):
    image,lines,els,_=fixture('plain');original=copy.deepcopy(els)
    monkeypatch.setattr(geometry,'_trace_validated',lambda *a:[('a','b')])
    monkeypatch.setattr(topology_features,'model',lambda:(object(),{}))
    pp={'a':[{'point':[.1,.5],'pixel':[25.6,128],'score':.9,'side':'right','status':'needs_review'}]}
    paths=[{'from_id':'a','to_id':b,'points':[[.1,.5],[.9,.5]]} for b in ('b','c')]
    monkeypatch.setattr(topology,'route',lambda *a:([('a','c')],{'paths':paths,'algorithm':'experimental','port_count':1},pp))
    details={};edges=infer_connections(els,lines,256,256,image,details)
    assert [(e['from_id'],e['to_id']) for e in edges]==[('a','b')]
    assert edges[0]['status']=='needs_review' and edges[0]['source']=='recognized_path_candidate'
    assert len(details['paths'])==1 and details['paths'][0]['from_bbox']==original[0]['bbox']
    assert details['recognition_is_advisory']
    assert 'pixel' not in els[0]['properties']['ports'][0]


def test_no_symbol_still_reports_crossings(monkeypatch):
    image,lines,_,maps=fixture('plain');monkeypatch.setattr(topology,'model',lambda:(object(),{}))
    monkeypatch.setattr(topology,'line_segments',lambda *a:(lines,topology.line_ink(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY))))
    pairs,details,_=topology.route([],lines,256,256,image,recognized=(maps,{}))
    assert pairs==[] and details['junctions'][0]['type']=='plain_cross'


def test_model_hash_mismatch_rejected(tmp_path,monkeypatch):
    (tmp_path/'topology.json').write_text('{"weights_sha256":"wrong"}')
    (tmp_path/'topology.onnx').write_bytes(b'not a network')
    monkeypatch.setattr(topology_features,'MODEL_DIR',tmp_path);topology_features.model.cache_clear()
    try:
        with pytest.raises(ValueError):topology_features.model()
    finally:topology_features.model.cache_clear()


def test_diagonal_stem_and_blank_boundary(monkeypatch):
    ink=np.zeros((128,128),np.uint8);cv2.line(ink,(40,40),(75,75),255,2);prob=np.full((6,128,128),255,np.uint8)
    es=[element('a',[20/128,20/128,40/128,40/128]),element('blank',[.7,.05,.85,.2])]
    pp=topology.recognize_ports(es,ink,prob,128,128,{})
    assert pp['a'] and not pp['blank']
