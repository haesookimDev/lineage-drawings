#!/usr/bin/env python3
"""Reproducible CPU training; no independent test-project pixels or labels read."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend.learned_symbols import proposals,features,probabilities,prepare_gray
from evaluation.benchmark import load_gt
from evaluation.v4.evaluate import within


def overlaps(boxes,truth):
    b=np.asarray(boxes,np.float32);g=np.asarray(truth,np.float32)
    lo=np.maximum(b[:,None,:2],g[None,:,:2]);hi=np.minimum(b[:,None,2:],g[None,:,2:]);inter=np.maximum(0,hi-lo).prod(axis=2)
    area=(b[:,2:]-b[:,:2]).prod(axis=1);ga=(g[:,2:]-g[:,:2]).prod(axis=1)
    return inter/np.maximum(1,area[:,None]+ga[None,:]-inter)


def training_sheets():
    plan=json.loads((ROOT/'evaluation/v5/PLAN.json').read_text())
    for ident in plan['training']['open100']:yield 'open100',ident,None
    labels=json.loads((ROOT/'evaluation/v4/pahala-labels.json').read_text())['sheets']
    for ident in plan['training']['pahala']:yield 'pahala',ident,labels[ident]
    for name in plan['training']['synthetic_members']:yield 'v5-synthetic',Path(name).stem,None


def train(variant='original'):
    cv2.setNumThreads(4);cv2.setRNGSeed(20261028);rng=np.random.default_rng(20261028);start=time.perf_counter();all_x=[];all_y=[];records=[]
    for group,ident,region in training_sheets():
        path=ROOT/'.benchmark-data'/group/(ident+'.png')
        with Image.open(path) as im:
            size=im.size;im=im.convert('RGB');im.thumbnail((2400,2400));gray=prepare_gray(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2GRAY));h,w=gray.shape
        gt=region['nodes'] if region else load_gt(path.with_suffix('.graphml'),size)[0]
        truth=np.array([[round(v*[w,h][n%2]) for n,v in enumerate(g['bbox'])] for g in gt],np.int32)
        boxes=proposals(gray,dilations=(1,3,7) if variant=='original' else (1,3,7,15))
        if region:boxes=[b for b in boxes if within([b[n]/[w,h][n%2] for n in range(4)],region['roi']) and not any(within([b[n]/[w,h][n%2] for n in range(4)],ig) for ig in region['ignore'])]
        if not boxes:
            boxes=[(int(x),int(y),int(x+24),int(y+24)) for x,y in rng.uniform([0,0],[w-24,h-24],(100,2))]
        iou=overlaps(boxes,truth);best=iou.max(axis=1);covered=(iou.max(axis=0)>=.5).sum()
        positives=[boxes[i] for i in np.flatnonzero(best>=.5)]
        # Exact and jittered target regions teach object extent; training targets
        # are read only here, never in runtime proposal generation.
        for a,b,c,d in truth:
            cw,ch=c-a,d-b
            for _ in range(9 if group!='v5-synthetic' else 5):
                dx,dy=rng.uniform(-.08,.08,2)*[cw,ch];sx,sy=rng.uniform(.88,1.12,2);cx,cy=(a+c)/2+dx,(b+d)/2+dy
                box=(max(0,round(cx-cw*sx/2)),max(0,round(cy-ch*sy/2)),min(w,round(cx+cw*sx/2)),min(h,round(cy+ch*sy/2)))
                if box[2]-box[0]>=5 and box[3]-box[1]>=5:positives.append(box)
        negative_indices=np.flatnonzero(best<.15);rng.shuffle(negative_indices);negatives=[boxes[i] for i in negative_indices[:2000]]
        chosen=positives+negatives;x=features(gray,chosen);y=np.array([1]*len(positives)+[0]*len(negatives),np.int32)
        # Rotate/reflect only training sheets. The exact same transformation is
        # applied to boxes, context and ports, without reading test projects.
        augmented=0
        for transform in (() if variant=='original' else ('flip_h','flip_v','rotate90')):
            pi=rng.choice(len(positives),min(400,len(positives)),replace=False)
            ni=rng.choice(len(negatives),min(400,len(negatives)),replace=False)
            selected=[positives[int(i)] for i in pi]+[negatives[int(i)] for i in ni]
            if transform=='flip_h':
                transformed=cv2.flip(gray,1);bb=[(w-c,b,w-a,d) for a,b,c,d in selected]
            elif transform=='flip_v':
                transformed=cv2.flip(gray,0);bb=[(a,h-d,c,h-b) for a,b,c,d in selected]
            else:
                transformed=cv2.rotate(gray,cv2.ROTATE_90_COUNTERCLOCKWISE);bb=[(b,w-c,d,w-a) for a,b,c,d in selected]
            all_x.append(features(transformed,bb));all_y.append(np.array([1]*len(pi)+[0]*len(ni),np.int32));augmented+=len(selected)
        all_x.append(x);all_y.append(y);records.append({'group':group,'id':ident,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'label_sha256':hashlib.sha256((ROOT/'evaluation/v4/pahala-labels.json').read_bytes()).hexdigest() if region else hashlib.sha256(path.with_suffix('.graphml').read_bytes()).hexdigest(),'symbols':len(gt),'proposals':len(boxes),'proposal_gt_recalled_at_iou50':int(covered),'positive_samples':len(positives),'negative_samples':len(negatives),'rotated_reflected_samples':augmented})
        print(group,ident,records[-1],flush=True)
    trees,depth,active=(128,18,40) if variant=='original' else (64,16,24)
    x=np.concatenate(all_x);y=np.concatenate(all_y);forest=cv2.ml.RTrees_create();forest.setMaxDepth(depth);forest.setMinSampleCount(4);forest.setRegressionAccuracy(0);forest.setUseSurrogates(False);forest.setMaxCategories(2);forest.setCalculateVarImportance(True);forest.setActiveVarCount(active);forest.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER,trees,0));forest.train(x,cv2.ml.ROW_SAMPLE,y)
    directory=ROOT/'backend/assets';directory.mkdir(exist_ok=True);path=directory/'symbol-objectness.xml.gz';forest.save(str(path));scores=probabilities(forest,x)
    metadata={'model':'hog-rtrees-symbol-v1','method':'OpenCV random-forest objectness over HOG/foreground/context features and geometry proposals. No pretrained neural weights. No semantic/standard approval.','opencv':cv2.__version__,'seed':20261028,'trees':trees,'max_depth':depth,'feature_dimensions':x.shape[1],'threshold':.72,'nms_iou':.35,'max_proposals':7000,'training_seconds':round(time.perf_counter()-start,3),'training_samples':len(y),'positive_samples':int(y.sum()),'negative_samples':int((y==0).sum()),'in_sample_positive_acceptance_at_072':float((scores[y==1]>=.72).mean()),'in_sample_negative_acceptance_at_072':float((scores[y==0]>=.72).mean()),'weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training_sheets':records,'training_sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'backend/learned_symbols.py',Path(__file__),ROOT/'evaluation/v5/PLAN.json']},'augmentation':'None; exact and jittered training boxes only.' if variant=='original' else 'Training-only horizontal/vertical reflection and 90-degree rotation; max 400 positives and 400 negatives per sheet per transform; not independent source drawings.','source_notice':'See backend/assets/NOTICE.md. OPEN100 and synthetic derive from PID2Graph CC BY-SA 4.0; Pahala is public third-party project data, not assumed open-licensed. No source images or OCR text redistributed.','role_note':'Pahala p23/24 training diagnostics are not generalization evidence. Virden/Ruttan independent projects never read here.'}
    (directory/'symbol-objectness.json').write_text(json.dumps(metadata,indent=2)+'\n');(ROOT/'evaluation/v5/TRAINING.json').write_text(json.dumps(metadata,indent=2)+'\n');print('TRAINED',len(y),'samples',x.shape,'seconds',metadata['training_seconds'],'model bytes',path.stat().st_size,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['original','augmented'],default='original');train(parser.parse_args().variant)
