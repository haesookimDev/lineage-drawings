"""Regression scenes for missing ports, small stubs and coarse template search."""
import cv2
import numpy as np
import pytest
from PIL import Image,ImageDraw
from backend.extract import infer_connections
from backend.network import axis_segments
from backend.symbols import template_candidates


def symbol(ident,box,properties=None):
    return {'id':ident,'kind':'unknown','bbox':box,'properties':properties or {}}


def test_two_pixel_outline_to_line_offset_is_recovered():
    elements=[symbol('a',[.05,.4,.16,.6]),symbol('b',[.84,.4,.95,.6])]
    edges=infer_connections(elements,[[18,50,82,50]],100,100)
    assert len(edges)==1 and {edges[0]['from_id'],edges[0]['to_id']}=={'a','b'}


def test_weak_circle_is_a_leaf_but_does_not_cut_a_through_route():
    weak={'geometry':'round_loop','connection_eligible':False}
    elements=[symbol('a',[.05,.4,.15,.6]),symbol('b',[.85,.4,.95,.6]),symbol('weak',[.45,.4,.55,.6],weak)]
    through=infer_connections(elements,[[15,50,85,50]],100,100)
    assert len(through)==1 and {through[0]['from_id'],through[0]['to_id']}=={'a','b'}
    leaf=infer_connections(elements[:1]+elements[2:],[[15,50,45,50]],100,100)
    assert len(leaf)==1 and {leaf[0]['from_id'],leaf[0]['to_id']}=={'a','weak'}


def test_short_visible_pipe_stub_survives_line_detection():
    im=Image.new('RGB',(100,100),'white');d=ImageDraw.Draw(im)
    d.line((30,40,40,40),fill='black',width=2)
    segments=axis_segments(im)
    assert any(abs(y-40)<2 and x<=32 and u>=38 for x,y,u,v in segments)


@pytest.mark.parametrize('offset',[(0,0),(1,1),(1,0)])
def test_coarse_search_recovers_odd_position_full_resolution_shape(monkeypatch,offset):
    import backend.symbols as module
    template=np.zeros((15,21),np.float32)
    pts=np.array([[1,2],[19,12],[19,2],[1,12],[1,2]],np.int32)
    cv2.polylines(template,[pts],False,1,1)
    monkeypatch.setattr(module,'prototypes',lambda:[(template,'valve')])
    x,y=50+offset[0],40+offset[1]
    gray=np.full((100,140),255,np.uint8);gray[y:y+15,x:x+21]=(255-template*255).astype(np.uint8)
    gray[y+7,x-15:x]=0;gray[y+7,x+21:x+36]=0
    result=template_candidates(gray)
    assert any(abs(box[0]*140-x)<=1 and abs(box[1]*100-y)<=1 and props['match_score']>.95 for box,props in result)


def test_generic_short_pass_preserves_old_connections(monkeypatch):
    from backend import legacy_network,network
    im=Image.new('RGB',(100,100),'white')
    elements=[symbol(c,[.1+i*.2,.1,.2+i*.2,.2],{'profile':'generic_fallback'}) for i,c in enumerate('abc')]
    old=[[1,2,3,4]];new=[[5,6,7,8]]
    monkeypatch.setattr(network,'axis_segments',lambda _:new)
    monkeypatch.setattr(legacy_network,'trace_adjacency',lambda es,lines,w,h:[('a','b')] if lines is old else [('b','c')])
    details={};edges=infer_connections(elements,old,100,100,image=im,diagnostics=details)
    assert {frozenset((e['from_id'],e['to_id'])) for e in edges}=={frozenset(('a','b')),frozenset(('b','c'))}
    assert details['retained_legacy_connections']==details['supplemental_connections']==1
    assert all(e['status']=='needs_review' and e['source']=='line_trace_candidate' for e in edges)
