"""Exact procedural validation. Not evidence of real drawing accuracy."""
import sys,json,hashlib
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v7.procedural import scene,CHANNELS
cv2.setNumThreads(2)
rows=[]
for epoch in (4,8,16):
 p=ROOT/'.benchmark-data/v7-models'/f'epoch-{epoch}.onnx';net=cv2.dnn.readNetFromONNX(str(p));conf=np.zeros((6,3),np.int64)
 for seed in range(20271029,20271189):
  gray,prompt,truth=scene(seed);ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,7);x=np.zeros((1,3,384,384),np.float32);x[0,:,:256,:256]=np.stack([1-gray/255,ink/255,prompt]);net.setInput(x);raw=net.forward()[0,:,:256,:256];pred=(raw>0).astype(bool);truth=truth>0
  for k in range(6):conf[k]+=[np.count_nonzero(pred[k]&truth[k]),np.count_nonzero(pred[k]&~truth[k]),np.count_nonzero(~pred[k]&truth[k])]
 metrics={c:{'tp':int(t),'fp':int(f),'fn':int(n),'iou':float(t/max(1,t+f+n)),'f1':float(2*t/max(1,2*t+f+n))} for c,(t,f,n) in zip(CHANNELS,conf)};rows.append({'epoch':epoch,'weights_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'channels':metrics,'mean_iou':float(np.mean([m['iou'] for m in metrics.values()]))});print(epoch,rows[-1]['mean_iou'],flush=True)
(ROOT/'evaluation/v7/HEAD-VALIDATION.json').write_text(json.dumps({'seed_range':[20271029,20271188],'scenes':160,'threshold':.5,'metrics':'Pixel IoU/F1 of exact procedural masks, including radius4 point targets. No claim real pixel annotation or port object AP. Padding to384, no resizing.','rows':rows},indent=2)+'\n')
