import json

from PIL import Image, ImageDraw

from backend.extract import infer_connections, raster_candidates
from evaluation.benchmark import counts, match_boxes, score


def symbol(ident, box):
    return {'id':ident,'kind':'unknown','bbox':box}


def pairs(edges):
    return {frozenset((e['from_id'],e['to_id'])) for e in edges}


def test_connection_follows_bend_and_stops_at_intermediate_symbol():
    elements = [symbol('A',[.1,.1,.2,.2]),symbol('B',[.7,.7,.8,.8])]
    edges = infer_connections(elements,[[20,15,50,15],[50,15,50,75],[50,75,70,75]],100,100)
    assert pairs(edges)=={frozenset(('A','B'))}
    assert all(e['status']=='needs_review' for e in edges)
    elements = [symbol('A',[.1,.4,.2,.6]),symbol('B',[.45,.4,.55,.6]),symbol('C',[.8,.4,.9,.6])]
    edges = infer_connections(elements,[[20,50,80,50]],100,100)
    assert pairs(edges)=={frozenset(('A','B')),frozenset(('B','C'))}


def test_close_symbols_do_not_connect_without_a_visible_line_path():
    elements = [symbol('A',[.1,.1,.2,.2]),symbol('B',[.4,.4,.5,.5])]
    assert infer_connections(elements,[],100,100)==[]
    # Two parallel lines with a clear gap must remain separate components.
    elements = [symbol('A',[.1,.1,.2,.2]),symbol('B',[.8,.3,.9,.4])]
    assert infer_connections(elements,[[20,15,70,15],[30,35,80,35]],100,100)==[]


def test_closed_symbol_survives_internal_ocr_and_serializes(monkeypatch):
    image = Image.new('RGB',(800,500),'white')
    draw = ImageDraw.Draw(image)
    draw.ellipse((175,175,225,225),outline='black',width=2)
    draw.line((200,225,200,400),fill='black',width=2)
    text = {'id':'tag','kind':'text','label':'PT','bbox':[.235,.37,.265,.43]}
    monkeypatch.setattr('backend.extract.ocr',lambda image,warnings:[text])
    elements,segments = raster_candidates(image,[])
    symbols = [e for e in elements if e['kind']!='text']
    assert match_boxes(symbols,[symbol('circle',[175/800,175/500,225/800,225/500])],.5)
    assert all(e['kind']=='unknown' and e['status']=='needs_review' for e in symbols)
    json.dumps({'elements':elements,'segments':segments})


def test_evaluation_matching_is_one_to_one_and_edges_are_not_hidden():
    gt=[symbol('a',[0,0,.2,.2]),symbol('b',[.5,.5,.7,.7])]
    pred=[symbol('p1',[0,0,.2,.2]),symbol('p2',[0,0,.2,.2]),symbol('p3',[.8,.8,1,1])]
    scored=score(pred,[{'from_id':'p1','to_id':'p3'}],gt,{('a','b')},.5)
    assert scored['localization']==counts(1,2,1)
    assert scored['connections']==counts(0,1,1)


def test_evaluation_uses_augmenting_paths_instead_of_greedy_matches():
    gt=[symbol('a',[0,0,.2,.2]),symbol('b',[.15,0,.35,.2])]
    pred=[symbol('wide',[0,0,.35,.2]),symbol('exact',[0,0,.2,.2])]
    assert len(match_boxes(pred,gt,.5))==2


def test_gt_adjacency_does_not_skip_intermediate_equipment(tmp_path):
    from evaluation.benchmark import load_gt
    path=tmp_path/'fixture.graphml'
    nodes=''.join(f'<node id="{i}"><data key="label">{kind}</data><data key="xmin">0</data><data key="ymin">0</data><data key="xmax">10</data><data key="ymax">10</data></node>' for i,kind in [('a','pump'),('j','connector'),('b','valve'),('k','crossing'),('c','tank')])
    keys=''.join(f'<key id="{k}" for="node" attr.name="{k}" attr.type="string"/>' for k in ['label','xmin','ymin','xmax','ymax'])
    edges=''.join(f'<edge source="{a}" target="{b}"/>' for a,b in [('a','j'),('j','b'),('b','k'),('k','c')])
    path.write_text(f'<graphml xmlns="http://graphml.graphdrawing.org/xmlns">{keys}<graph edgedefault="undirected">{nodes}{edges}</graph></graphml>')
    _,adjacent,_=load_gt(path,(100,100))
    assert adjacent=={('a','b'),('b','c')}
