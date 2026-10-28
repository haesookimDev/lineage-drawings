"""Weak real/benchmark masks projected from training GraphML only."""
from pathlib import Path
import hashlib,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.v7.graph_data import sources,maps

def main():
 rng=np.random.default_rng(20261029);out=ROOT/'.benchmark-data/v7-masks';out.mkdir(exist_ok=True);records=[]
 for path in sources('training'):
  rgb,prompt,target,points=maps(path);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=gray.shape;centers=[p for _,p in points];rng.shuffle(centers)
  for j,(x,y) in enumerate(centers[:40]):
   a=int(np.clip(x-rng.uniform(64,192),0,max(0,w-256)));b=int(np.clip(y-rng.uniform(64,192),0,max(0,h-256)));im=gray[b:b+256,a:a+256];box=prompt[b:b+256,a:a+256];m=target[:,b:b+256,a:a+256].astype(np.uint8)
   if im.shape!=(256,256):continue
   p=out/f'{path.parent.name}-{path.stem}-{j}.npz';np.savez_compressed(p,image=im,prompt=box,target=m);records.append({'file':str(p.relative_to(ROOT)),'source':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'graph_sha256':hashlib.sha256(path.with_suffix('.graphml').read_bytes()).hexdigest(),'crop':[a,b,a+256,b+256],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  print(path.parent.name,path.stem,'points',len(points),'patches',min(40,len(centers)),flush=True)
 (ROOT/'evaluation/v7/TRAINING-DATA.json').write_text(json.dumps({'seed':20261029,'real_weak_patches':len(records),'patches':records,'notes':'Training-only GraphML line projections are weak supervision, not certified pixel labels. Separate exact procedural masks generated during training.'},indent=2)+'\n')
if __name__=='__main__':main()
