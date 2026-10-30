import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v9.train import model
torch.set_num_threads(2)
for name in (sys.argv[1:] or ['initial','epoch-4','epoch-8','frozen-bn-epoch-4']):
 p=ROOT/'.benchmark-data/v12-models'/f'{name}.pth';m=model();m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['model']);m.eval();m.head.decode_in_inference=False;x=torch.full((1,3,640,640),255.);out=p.with_suffix('.onnx');torch.onnx.export(m,x,str(out),input_names=['images'],output_names=['predictions'],opset_version=11,dynamo=False)
 with torch.no_grad():expected=m(x).numpy()
 net=cv2.dnn.readNetFromONNX(str(out));net.setInput(x.numpy());actual=net.forward();delta=float(np.abs(actual-expected).max());assert np.allclose(actual,expected,atol=.003,rtol=.003),delta
 meta={**json.loads((ROOT/'backend/assets/symbol-neural.json').read_text()),'model':'yolox-nano-symbol-v12-'+name,'weights_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'opencv_torch_max_abs_difference':delta,'detail_tiles':0,'training':'former White Rose pilot explicitly transferred to training + replay; no new test-project inputs'};out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(name,meta['weights_sha256'],delta,flush=True)
