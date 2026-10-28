"""GraphML topology for weak dense supervision, never runtime input."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
NS={'g':'http://graphml.graphdrawing.org/xmlns'}

def load(path,size):
 root=ET.parse(path).getroot();keys={k.get('id'):k.get('attr.name') for k in root.findall('g:key',NS)};nodes={};adj=defaultdict(set)
 for n in root.findall('.//g:node',NS):
  d={keys[v.get('key')]:v.text for v in n.findall('g:data',NS)};box=np.array([float(d[k]) for k in ('xmin','ymin','xmax','ymax')]);nodes[n.get('id')]={'label':d['label'],'box':box,'point':(box[:2]+box[2:])/2}
 edges=[]
 for e in root.findall('.//g:edge',NS):
  a,b=e.get('source'),e.get('target');d={keys[v.get('key')]:v.text for v in e.findall('g:data',NS)};edges.append((a,b,d.get('edge_label','solid')));adj[a].add(b);adj[b].add(a)
 return nodes,edges,adj


def maps(path,max_side=2400):
 im=Image.open(path).convert('RGB');original=im.size;im.thumbnail((max_side,max_side));w,h=im.size;sc=np.array([w/original[0],h/original[1]]);nodes,edges,adj=load(path.with_suffix('.graphml'),original);target=np.zeros((6,h,w),np.float32);boxes=np.zeros((h,w),np.uint8);points=[]
 objects={k for k,n in nodes.items() if n['label'] not in ('connector','crossing','background','arrow')}
 for k in objects:
  box=nodes[k]['box']*np.tile(sc,2);a,b,c,d=np.round(box).astype(int);cv2.rectangle(boxes,(a,b),(c,d),1,-1)
 for a,b,kind in edges:
  p=np.round(nodes[a]['point']*sc).astype(int);q=np.round(nodes[b]['point']*sc).astype(int);channel=1 if kind=='non-solid' else 0;cv2.line(target[channel],p,q,1,3)
 for k,n in nodes.items():
  p=np.round(n['point']*sc).astype(int)
  if n['label']=='connector' and any(x in objects for x in adj[k]):cv2.circle(target[2],p,4,1,-1);points.append(('port',p.tolist()))
  if n['label']=='crossing' and len(adj[k])>=3:
   channel=4 if len(adj[k])>=4 else 3;cv2.circle(target[channel],p,4,1,-1);points.append(('dot' if channel==4 else 'tee',p.tolist()))
 target[:2,boxes>0]=0
 # Coincident degree-two topology nodes indicate independent crossing paths.
 cross=[(k,n['point']*sc) for k,n in nodes.items() if n['label']=='crossing' and len(adj[k])==2]
 for i,(a,p) in enumerate(cross):
  for b,q in cross[i+1:]:
   if b not in adj[a] and np.linalg.norm(p-q)<4:
    cv2.circle(target[5],tuple(np.round((p+q)/2).astype(int)),4,1,-1)
 return np.asarray(im),boxes,target,points


def sources(role):
 p=json.loads((ROOT/'evaluation/v6/PLAN.json').read_text());old=json.loads((ROOT/'evaluation/v5/PLAN.json').read_text())
 for i in p[role]['open100']:yield ROOT/'.benchmark-data/open100'/(i+'.png')
 for n in old['training' if role=='training' else 'threshold_validation']['synthetic_members']:yield ROOT/'.benchmark-data/v5-synthetic'/Path(n).name
