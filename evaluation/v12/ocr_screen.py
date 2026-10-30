import sys,time,json
from pathlib import Path
from PIL import Image
from collections import Counter
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend.extract import ocr,clear_ocr_cache
from evaluation.v12.experimental_ocr import recognize
from evaluation.v10.data import images
cv2.setNumThreads(2);work=[('white-rose-25',ROOT/'.benchmark-data/v11-held/white-rose/p25-4800.png'),('white-rose-22',ROOT/'.benchmark-data/v11-held/white-rose/p22-4800.png'),('winkler-p142',ROOT/'.benchmark-data/v9-held/winkler/p142-4800.png'),('winnipeg-p7',ROOT/'.benchmark-data/v9-held/winnipeg/p7-4800.png'),('open100-6',ROOT/'.benchmark-data/open100/6.png')];rows=[]
for key,p in work:
 im,_=images(p);variants={};full=None
 for mode in ('full','auto','mask'):
  clear_ocr_cache();d={};t=time.perf_counter();words=recognize(im,[],ocr,d,mode);elapsed=time.perf_counter()-t
  if mode=='full':full=Counter(w['label'].casefold() for w in words)
  current=Counter(w['label'].casefold() for w in words);variants[mode]={'words':len(words),'seconds':elapsed,'diagnostics':d,'baseline_token_agreement':sum((full&current).values())/sum(full.values()) if full else 1.}
 rows.append({'id':key,'variants':variants});print(rows[-1],flush=True)
 (ROOT/'evaluation/v12/OCR-SCREEN.json').write_text(json.dumps({'role':'Development comparison, word agreement with baseline is not OCR ground-truth accuracy. No new-project sources opened.','rows':rows},indent=2)+'\n')
