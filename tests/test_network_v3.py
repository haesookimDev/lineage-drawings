from PIL import Image,ImageDraw

from backend.extract import infer_connections
from backend.network import port_graph


def symbol(ident,box):return {'id':ident,'kind':'unknown','bbox':box}
def pairs(edges):return {frozenset((e['from_id'],e['to_id'])) for e in edges}
def cross_scene(dot=False):
    image=Image.new('RGB',(100,100),'white');d=ImageDraw.Draw(image)
    d.line((15,50,85,50),fill='black',width=2);d.line((50,15,50,85),fill='black',width=2)
    if dot:d.ellipse((46,46,54,54),fill='black')
    elements=[symbol('L',[.05,.4,.15,.6]),symbol('R',[.85,.4,.95,.6]),symbol('T',[.4,.05,.6,.15]),symbol('B',[.4,.85,.6,.95])]
    return image,elements,[[15,50,85,50],[50,15,50,85]]


def test_plain_crossing_does_not_merge_the_two_routes():
    image,elements,lines=cross_scene()
    details={};edges=infer_connections(elements,lines,100,100,image=image,diagnostics=details)
    assert pairs(edges)=={frozenset(('L','R')),frozenset(('T','B'))}
    assert details['unmarked_crossings']==1
    assert all(e['status']=='needs_review' for e in edges)


def test_a_dot_connects_all_four_ports():
    image,elements,lines=cross_scene(dot=True)
    edges=infer_connections(elements,lines,100,100,image=image)
    assert len(pairs(edges))==6
    assert frozenset(('L','T')) in pairs(edges)


def test_a_tee_connects_without_needing_a_dot():
    _,elements,_=cross_scene();elements=elements[:3]
    edges=infer_connections(elements,[[15,50,85,50],[50,15,50,50]],100,100)
    assert pairs(edges)=={frozenset(('L','R')),frozenset(('L','T')),frozenset(('R','T'))}


def test_a_visible_gap_is_not_a_connection():
    elements=[symbol('L',[.05,.4,.15,.6]),symbol('T',[.4,.05,.6,.15])]
    edges=infer_connections(elements,[[15,50,43,50],[50,15,50,85]],100,100)
    assert edges==[]


def test_weak_round_shape_cannot_split_an_existing_route():
    elements=[symbol('A',[.05,.4,.15,.6]),symbol('B',[.85,.4,.95,.6]),
              {**symbol('weak',[.45,.4,.55,.6]),'properties':{'connection_eligible':False}}]
    edges=infer_connections(elements,[[15,50,85,50]],100,100)
    assert pairs(edges)=={frozenset(('A','B'))}


def test_prototypes_are_only_from_the_declared_development_sheets():
    import json,hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'backend/assets'
    meta=json.loads((root/'open100-dev-shapes.json').read_text())
    assert {p['sheet'] for p in meta['prototypes']}=={'0','1','2','3'}
    assert hashlib.sha256((root/'open100-dev-shapes.npz').read_bytes()).hexdigest()==meta['weights_sha256']
    from backend.symbols import prototypes
    assert len(prototypes())==73


def test_historical_extractor_imports_its_own_geometry(tmp_path,monkeypatch):
    import hashlib,subprocess
    from evaluation import benchmark
    root=tmp_path/'repo';backend=root/'backend';backend.mkdir(parents=True)
    (backend/'__init__.py').write_text('')
    (backend/'geometry.py').write_text('VALUE="historical geometry"\n')
    (backend/'extract.py').write_text('from backend.geometry import VALUE\n')
    subprocess.run(['git','init','-q',str(root)],check=True)
    subprocess.run(['git','add','backend'],cwd=root,check=True)
    subprocess.run(['git','-c','user.name=Evaluation Test','-c','user.email=evaluation@example.invalid','commit','-qm','fixture'],cwd=root,check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    monkeypatch.setattr(benchmark,'ROOT',root)
    import backend.geometry as current
    monkeypatch.setattr(current,'VALUE','working geometry',raising=False)
    frozen,_,sources=benchmark.load_extractor(revision)
    assert frozen.VALUE=='historical geometry'
    assert sources['backend/geometry.py']==hashlib.sha256((backend/'geometry.py').read_bytes()).hexdigest()


def test_unsupported_style_uses_unchanged_legacy_tracing():
    from backend.legacy_network import trace_adjacency as legacy
    image,elements,lines=cross_scene()
    for e in elements:e['properties']={'profile':'generic_fallback'}
    details={};edges=infer_connections(elements,lines,100,100,image=image,diagnostics=details)
    assert pairs(edges)=={frozenset(pair) for pair in legacy(elements,lines,100,100)}
    assert details['algorithm']=='visible-components-v4'
    assert all(e['source']=='line_trace_candidate' for e in edges)


def test_shape_assistance_requires_supported_continuation_markers(monkeypatch):
    from backend import symbols
    image=Image.new('RGB',(200,200),'white');legacy=[([.2,.2,.4,.4],{'geometry':'round_loop'})]
    def should_not_run(_):raise AssertionError('Unsupported style must not run expensive shape matching')
    monkeypatch.setattr(symbols,'template_candidates',should_not_run)
    result=symbols.refine_symbols(image,legacy)
    assert result[0][1]['profile']=='generic_fallback'


def test_connection_cap_is_independent_of_random_element_ids():
    import math
    records=[];lines=[]
    for i in range(34):
        angle=2*math.pi*i/34;x,y=500+300*math.cos(angle),500+300*math.sin(angle)
        records.append(symbol(f'{i:03}',[(x-8)/1000,(y-8)/1000,(x+8)/1000,(y+8)/1000]))
        lines.append([x,y,500,500])
    first,_=port_graph(records,lines,1000,1000)
    renamed=[{**r,'id':f'{33-i:03}'} for i,r in enumerate(records)]
    second,_=port_graph(renamed,lines,1000,1000)
    canonical=lambda pairs,ids:{tuple(sorted((ids[a],ids[b]))) for a,b in pairs}
    assert len(first)==len(second)==500
    assert canonical(first,{e['id']:i for i,e in enumerate(records)})==canonical(second,{e['id']:i for i,e in enumerate(renamed)})
