"""Coverage, source-coordinate mapping, model integrity and review contracts."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np,pytest
from PIL import Image
from backend import neural_symbols as neural
from backend.extract import raster_candidates


def test_bounded_tiles_cover_source_edges_and_have_overlap():
 boxes=neural.tiles(4800,4800);assert len(boxes)==36
 coverage=np.zeros((4800,4800),np.uint8)
 for a,b,c,d in boxes:coverage[b:d,a:c]=1
 assert coverage.all()
 assert neural.tiles(300,200)==[(0,0,300,200)]
 with pytest.raises(ValueError):neural.tiles(5000,200)


def test_letterbox_and_decode_keep_native_source_coordinates():
 im=np.full((200,400,3),255,np.uint8);blob,ratio=neural.preprocess(im)
 assert blob.shape==(1,3,640,640) and ratio==1.6
 assert np.all(blob[0,:,320:]==114)
 raw=np.zeros((1,8400,6),np.float32);raw[0,0,:]=[2,3,0,0,.8,.5]
 result=neural.decode(raw)
 assert np.allclose(result[0],[12,20,20,28,.4])


def test_neural_weights_integrity_rejects_corruption(tmp_path,monkeypatch):
 (tmp_path/'symbol-neural.json').write_text(json.dumps({'weights_sha256':'0'*64}));(tmp_path/'symbol-neural.onnx').write_bytes(b'corrupt')
 neural.model.cache_clear();monkeypatch.setattr(neural,'MODEL_DIR',tmp_path)
 try:
  with pytest.raises(ValueError):neural.model()
 finally:neural.model.cache_clear()


def test_neural_candidates_preserve_unknown_and_human_review(monkeypatch):
 monkeypatch.setattr('backend.extract.geometric_candidates',lambda *args,**kwargs:([([.1,.2,.3,.4],{'geometry':'neural_symbol','requires_review':True})],[],False))
 els,_=raster_candidates(Image.new('RGB',(100,100),'white'),[],with_ocr=False)
 assert len(els)==1 and els[0]['source']=='neural_symbol_candidate'
 assert els[0]['kind']=='unknown' and els[0]['status']=='needs_review' and els[0]['concept_id'] is None


def test_uncertain_small_glyph_is_revisited_with_bounded_detail_crop(monkeypatch):
 class Net:
  calls=0
  def setInput(self,x):assert x.shape==(1,3,640,640)
  def forward(self):
   self.calls+=1
   return np.array([[200.,100.,220.,120.,.2 if self.calls==1 else .9]])
 net=Net();monkeypatch.setattr(neural,'model',lambda:(net,{'threshold':.6,'detail_tiles':6,'weights_sha256':'fixture'}));monkeypatch.setattr(neural,'decode',lambda raw,size:raw)
 im=Image.new('RGB',(1024,256),'white');__import__('PIL.ImageDraw',fromlist=['Draw']).Draw(im).line((0,120,1023,120),fill='black',width=2)
 found=neural.detect(im)
 assert net.calls==2 and found and all(p['inference_view']=='detail' and p['detail_tile_count']==1 and p['tile_count']<=37 and p['requires_review'] for b,p in found)
 assert all(0<=x<=1 for b,p in found for x in b)


def test_test_pages_disjoint_from_training_and_validation():
 root=Path(__file__).resolve().parents[1];p=json.loads((root/'evaluation/v6/PLAN.json').read_text());new={(g,i) for g,ids in p['new_page_test'].items() for i in ids}
 for role in ('training','validation'):
  used={(g,i) for g,ids in p[role].items() if g!='synthetic' for i in ids}
  assert new.isdisjoint(used)
 labels=root/'evaluation/v6/new-page-labels.json';assert hashlib.sha256(labels.read_bytes()).hexdigest()==json.loads((root/'evaluation/v6/LABELS-FROZEN.json').read_text())['sha256']


def test_release_candidate_set_matches_the_validated_selection():
 root=Path(__file__).resolve().parents[1]
 selected=json.loads((root/'evaluation/v9/RELEASE-SELECTION.json').read_text())
 deployed=json.loads((root/'backend/assets/symbol-neural.json').read_text())
 assert deployed['merge_mode']==selected['validated_candidate_set']=='neural_only'
 assert deployed['weights_sha256']==selected['symbol_configuration']['weights_sha256']
