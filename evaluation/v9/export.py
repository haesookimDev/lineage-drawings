"""Export a training checkpoint to fixed-shape ONNX and check OpenCV parity."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v6.train import model
from backend.neural_symbols import decode
ap=argparse.ArgumentParser();ap.add_argument('--epoch',type=int,required=True);args=ap.parse_args();torch.set_num_threads(2)
p=ROOT/'.benchmark-data/v9-models'/f'epoch-{args.epoch}.pth';m=model();m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['model']);m.eval();m.head.decode_in_inference=False
out=ROOT/'.benchmark-data/v9-models'/f'epoch-{args.epoch}.onnx';x=torch.full((1,3,640,640),255.)
torch.onnx.export(m,x,str(out),input_names=['images'],output_names=['predictions'],opset_version=11,dynamo=False)
with torch.no_grad():expected=m(x).numpy()
net=cv2.dnn.readNetFromONNX(str(out));net.setInput(x.numpy());actual=net.forward();difference=float(np.max(np.abs(actual-expected)));assert np.allclose(actual,expected,rtol=.003,atol=.003),(difference,actual.shape)
metadata={'model':'yolox-nano-symbol-v9','weights_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'epoch':args.epoch,'threshold':.45,'detail_tiles':6,'detail_seed_max':.6,'tile_size':1024,'overlap':.2,'input_size':640,'merge_mode':'replace','max_source_side':4800,'max_tiles':36,'opencv_torch_max_abs_difference':difference,'score_note':'Objectness times single-class sigmoid; not calibrated probability. Generic candidate only, no class or standard approval.'}
out.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))
