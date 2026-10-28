from pathlib import Path
import argparse,hashlib,json,sys
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v7.train import Net
from evaluation.v7.procedural import CHANNELS
ap=argparse.ArgumentParser();ap.add_argument('--epoch',type=int,required=True);args=ap.parse_args();torch.set_num_threads(2);p=ROOT/'.benchmark-data/v7-models'/f'epoch-{args.epoch}.pth';m=Net();m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['model']);m.eval();x=torch.rand(1,3,384,384);out=p.with_suffix('.onnx');torch.onnx.export(m,x,str(out),opset_version=11,dynamo=False,input_names=['features'],output_names=['logits'])
with torch.no_grad():wanted=m(x).numpy()
net=cv2.dnn.readNetFromONNX(str(out));net.setInput(x.numpy());actual=net.forward();delta=float(np.abs(actual-wanted).max());assert np.allclose(actual,wanted,atol=.003,rtol=.003),delta
meta={'model':'tiny-unet-line-port-v1','channels':CHANNELS,'epoch':args.epoch,'weights_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'line_threshold':.2,'port_threshold':.25,'junction_threshold':.35,'dash_threshold':.6,'port_snap':6,'route_mode':'new','opencv_torch_max_abs_difference':delta,'score_note':'Heatmap scores uncalibrated; ports/junctions/paths require review. GraphML line projections are weak supervision.'};out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
