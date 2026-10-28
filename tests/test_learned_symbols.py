"""Runtime integrity, low-contrast ink, review semantics and data separation."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image,ImageDraw
from backend import learned_symbols as learned
from backend.extract import raster_candidates
from backend.network import line_ink

ROOT=Path(__file__).resolve().parents[1]


def test_trained_forest_integrity_and_feature_contract():
    classifier,meta=learned.model()
    assert classifier.isTrained() and classifier.getVarCount()==395
    assert hashlib.sha256((learned.MODEL_DIR/'symbol-objectness.xml.gz').read_bytes()).hexdigest()==meta['weights_sha256']
    assert meta['training_samples']==meta['positive_samples']+meta['negative_samples']
    assert {r['group'] for r in meta['training_sheets']}=={'open100','pahala','v5-synthetic'}
    assert {r['id'] for r in meta['training_sheets'] if r['group']=='pahala'}=={'p23','p24'}
    plan=json.loads((ROOT/'evaluation/v5/PLAN.json').read_text())
    assert set(plan['training']['open100']).isdisjoint(plan['threshold_validation']['open100'])
    assert set(plan['training']['synthetic_members']).isdisjoint(plan['threshold_validation']['synthetic_members'])


def test_corrupt_weights_are_rejected_before_inference(tmp_path,monkeypatch):
    (tmp_path/'symbol-objectness.json').write_text(json.dumps({'weights_sha256':'0'*64}))
    (tmp_path/'symbol-objectness.xml.gz').write_bytes(b'corrupt')
    learned.model.cache_clear()
    monkeypatch.setattr(learned,'MODEL_DIR',tmp_path)
    try:
        with pytest.raises(ValueError):learned.model()
    finally:learned.model.cache_clear()


def test_adaptive_ink_preserves_faint_lines_and_dark_background():
    gray=np.full((160,160),255,np.uint8);cv2.line(gray,(20,80),(140,80),225,2)
    assert np.count_nonzero(line_ink(gray)[79:82,30:130])>150
    assert np.array_equal(line_ink(gray),line_ink(255-gray))
    boxes=learned.proposals(gray)
    assert len(boxes)<=learned.MAX_PROPOSALS
    assert all(0<=a<c<=160 and 0<=b<d<=160 for a,b,c,d in boxes)


def test_learned_candidates_require_human_semantic_review(monkeypatch):
    image=Image.new('RGB',(220,150),'white');ImageDraw.Draw(image).rectangle((60,50,100,90),outline='black',width=2)
    monkeypatch.setattr('backend.geometry.legacy_candidates',lambda *args:([],[],False))
    monkeypatch.setattr('backend.symbols.refine_symbols',lambda *args:[])
    monkeypatch.setattr(learned,'detect',lambda *args:[([.25,.3,.5,.65],{'geometry':'learned_symbol','detector_score':.9,'requires_review':True})])
    elements,_=raster_candidates(image,[],with_ocr=False)
    assert len(elements)==1
    e=elements[0]
    assert e['source']=='learned_symbol_candidate' and e['status']=='needs_review' and e['kind']=='unknown'
    assert e['properties']['symbol_model_sha256']==learned.model()[1]['weights_sha256']
