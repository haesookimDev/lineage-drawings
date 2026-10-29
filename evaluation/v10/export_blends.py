"""One-pass detectors from already-trained endpoints; no held-out data access."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from evaluation.v6.train import model

torch.set_num_threads(2)
cv2.setNumThreads(2)
out=ROOT/'.benchmark-data/v10/models'
out.mkdir(parents=True,exist_ok=True)
paths=[ROOT/'.benchmark-data/v6-models/epoch-16.pth',ROOT/'.benchmark-data/v9b-models/epoch-8.pth']
states=[torch.load(p,map_location='cpu',weights_only=True)['model'] for p in paths]
rows=[]
for fraction in (.25,.5,.75):
    state={k:(states[0][k]*(1-fraction)+states[1][k]*fraction if states[0][k].is_floating_point() else states[1][k]) for k in states[0]}
    m=model();m.load_state_dict(state);m.eval();m.head.decode_in_inference=False
    path=out/f'blend-{fraction}.onnx'
    x=torch.full((1,3,640,640),255.)
    torch.onnx.export(m,x,str(path),input_names=['images'],output_names=['predictions'],opset_version=11,dynamo=False)
    with torch.no_grad():expected=m(x).numpy()
    net=cv2.dnn.readNetFromONNX(str(path));net.setInput(x.numpy());actual=net.forward()
    difference=float(np.max(np.abs(actual-expected)))
    assert np.allclose(actual,expected,rtol=.003,atol=.003),difference
    meta={'model':'yolox-nano-symbol-v10','weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'v9_fraction':fraction,'threshold':.6,'detail_tiles':0,'tile_size':1024,'overlap':.2,'input_size':640,'merge_mode':'neural_only','max_source_side':4800,'max_tiles':36,'opencv_torch_max_abs_difference':difference,'score_note':'Generic single-model candidate; uncalibrated, requires human review.'}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');rows.append(meta)
    print(fraction,meta['weights_sha256'],difference,flush=True)
(ROOT/'evaluation/v10/MODEL-CANDIDATES.json').write_text(json.dumps({'method':'Linear interpolation of all floating tensors, including BN statistics; integer buffers from v9. No gradient update, no ensemble/double inference. Parameters selected on explicit development only.','endpoints':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],'candidates':rows},indent=2)+'\n')
