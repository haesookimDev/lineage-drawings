#!/usr/bin/env python3
"""Build small CC BY-SA shape prototypes from OPEN100 development sheets only.

No regression or confirmation annotations are read. Shape family stays a model
hint; the production element kind remains unknown and requires human review.
"""
import hashlib,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_gt

def main():
    samples=[]
    for ident in ('0','1','2','3'):
        path=ROOT/f'.benchmark-data/open100/{ident}.png'
        with Image.open(path) as raw:
            original=raw.size;im=raw.convert('RGB');im.thumbnail((2400,2400))
        bw=(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2GRAY)<210).astype(np.float32)
        gt,_,_=load_gt(ROOT/f'.benchmark-data/open100/{ident}.graphml',original)
        for g in gt:
            if g['kind'] not in ('valve','pump','tank','inlet/outlet'):continue
            a,b,c,d=g['bbox'];x1,y1,x2,y2=round(a*im.width),round(b*im.height),round(c*im.width),round(d*im.height)
            crop=bw[max(0,y1):y2+1,max(0,x1):x2+1].copy();h,w=crop.shape
            if min(h,w)<5 or max(h,w)>280:continue
            norm=cv2.resize(crop,(40,40)).ravel();norm-=norm.mean();norm/=np.linalg.norm(norm)+1e-6
            if any(t['family']==g['kind'] and .8<w/t['image'].shape[1]<1.25 and .8<h/t['image'].shape[0]<1.25 and np.dot(norm,t['norm'])>.86 for t in samples):continue
            samples.append({'image':crop,'family':g['kind'],'norm':norm,'sheet':ident,'annotation':g['id'],'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    # Pointed continuation markers are recognized geometrically, without text
    # templates from the page. Retain valve/pump/tank prototypes only.
    samples=[s for s in samples if s['family']!='inlet/outlet']
    output=ROOT/'backend/assets';output.mkdir(exist_ok=True)
    path=output/'open100-dev-shapes.npz';np.savez_compressed(path,**{f't{i}':s['image'] for i,s in enumerate(samples)})
    meta={'version':'open100-dev-shapes-v1','source':'https://zenodo.org/records/14803338','license':'CC-BY-SA-4.0','development_ids':['0','1','2','3'],'weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'prototype_count':len(samples),'prototypes':[{k:v for k,v in s.items() if k not in ('image','norm')} for s in samples]}
    (output/'open100-dev-shapes.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Prototypes:',len(samples),'weights SHA256:',meta['weights_sha256'])

if __name__=='__main__':main()
