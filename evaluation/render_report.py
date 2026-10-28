#!/usr/bin/env python3
"""Render measured benchmark results; no model selection or parameter changes."""
from pathlib import Path
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"lineage-mpl"))

import csv
import json
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_gt,match_boxes


def load(version,split):
    return json.loads((ROOT/f'evaluation/results/{version}-{split}/report.json').read_text())


def main():
    out=ROOT/'evaluation/figures';out.mkdir(exist_ok=True)
    baseline,improved=load('baseline','holdout'),load('improved','holdout')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    for axis,metric,title in zip(axes,['localization','connections','oracle_boxes'],['Symbol localization (IoU >= 0.50)','Connections: end to end','Connections: oracle symbol boxes']):
        for i,label in enumerate(['precision','recall','f1']):
            for off,report,color in [(-.18,baseline,'#9ca3af'),(.18,improved,'#0f766e')]:
                val=report['aggregate']['0.5'][metric][label]
                axis.bar(i+off,val,width=.33,color=color,label=('Baseline' if off<0 else 'Improved') if i==0 else None)
                axis.text(i+off,val+.018,f'{val*100:.1f}',ha='center',fontsize=9)
        axis.set(title=title,ylim=(0,1.05),xticks=[0,1,2],xticklabels=['Precision','Recall','F1'])
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.18);axis.set_axisbelow(True)
    axes[0].legend(frameon=False,loc='upper left');fig.suptitle('PID2Graph OPEN100 | fixed holdout: 8 real design sheets, 527 symbols, 1,393 adjacencies',fontsize=14)
    fig.text(.5,-.018,'Measured in this repository. Oracle boxes are diagnostic, not deployable accuracy. Data: Stuermer et al. / Energy Impact Center, CC BY-SA 4.0.',ha='center',fontsize=9)
    fig.savefig(out/'holdout-metrics.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Fixed sheet ID 6 is an illustration, not selected by best score.
    ident='6'
    with Image.open(ROOT/f'.benchmark-data/open100/{ident}.png') as source:
        original=source.size;im=source.convert('RGB');im.thumbnail((2400,2400))
    gt,_,_=load_gt(ROOT/f'.benchmark-data/open100/{ident}.graphml',original)
    fig,axes=plt.subplots(1,2,figsize=(18,8),layout='constrained')
    for axis,version in zip(axes,['baseline','improved']):
        pred=json.loads((ROOT/f'evaluation/results/{version}-holdout/{ident}-predictions.json').read_text())
        symbols=[e for e in pred['elements'] if e['kind']!='text'];matched=match_boxes(symbols,gt,.5)
        axis.imshow(im);axis.axis('off')
        for item in symbols:
            x1,y1,x2,y2=item['bbox'];axis.add_patch(Rectangle((x1*im.width,y1*im.height),(x2-x1)*im.width,(y2-y1)*im.height,fill=False,linewidth=.9,edgecolor='#16a34a' if item['id'] in matched else '#dc2626'))
        for item in gt:
            if item['id'] in matched.values():continue
            x1,y1,x2,y2=item['bbox'];axis.add_patch(Rectangle((x1*im.width,y1*im.height),(x2-x1)*im.width,(y2-y1)*im.height,fill=False,linewidth=.8,linestyle='--',edgecolor='#2563eb'))
        axis.set_title(f'{version.title()} | TP {len(matched)}, FP {len(symbols)-len(matched)}, FN {len(gt)-len(matched)}')
    fig.suptitle('OPEN100 sheet 6 | green: matched proposal | red: false positive | blue dashed: missed symbol',fontsize=14)
    fig.text(.5,-.015,'Source: PID2Graph / Energy Impact Center OPEN100. Adapted overlays: LINEAGE. CC BY-SA 4.0. Class-agnostic localization only.',ha='center',fontsize=10)
    fig.savefig(out/'sheet6-comparison.png',dpi=150,bbox_inches='tight');plt.close(fig)
    # Deterministic first unmatched valve/instrumentation + false positive crops.
    ident='4'
    with Image.open(ROOT/f'.benchmark-data/open100/{ident}.png') as source:
        original=source.size;im=source.convert('RGB')
    gt,_,_=load_gt(ROOT/f'.benchmark-data/open100/{ident}.graphml',original)
    pred=json.loads((ROOT/f'evaluation/results/improved-holdout/{ident}-predictions.json').read_text())
    symbols=[e for e in pred['elements'] if e['kind']!='text'];matched=match_boxes(symbols,gt,.5)
    examples=[]
    for kind in ['valve','instrumentation','tank']:
        item=next((e for e in gt if e['kind']==kind and e['id'] not in matched.values()),None)
        if item:examples.append((f'Missed {kind}',item['bbox'],'#2563eb'))
    for item in [e for e in symbols if e['id'] not in matched][:3]:examples.append(('False positive proposal',item['bbox'],'#dc2626'))
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for axis,(title,box,color) in zip(axes.flat,examples):
        a,b,c,d=box;x1,y1,x2,y2=a*im.width,b*im.height,c*im.width,d*im.height;pad=45
        crop=im.crop((int(x1-pad),int(y1-pad),int(x2+pad),int(y2+pad)))
        axis.imshow(crop);axis.add_patch(Rectangle((pad,pad),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=2));axis.set_title(title);axis.axis('off')
    for axis in list(axes.flat)[len(examples):]:axis.axis('off')
    fig.suptitle('Remaining errors | fixed sheet 4 | no tuning after holdout evaluation',fontsize=14)
    fig.text(.5,-.015,'Data: Stuermer et al. / Energy Impact Center OPEN100, CC BY-SA 4.0. Crops are derived from public design drawings.',ha='center',fontsize=9)
    fig.savefig(out/'remaining-errors.png',dpi=150,bbox_inches='tight');plt.close(fig)
    with (ROOT/'evaluation/results/per-sheet.csv').open('w') as f:
        columns=['version','split','sheet','gt_symbols','pred_symbols','localization_precision','localization_recall','localization_f1','gt_connections','pred_connections','connection_precision','connection_recall','connection_f1','seconds']
        writer=csv.DictWriter(f,columns);writer.writeheader()
        for split in ['development','holdout']:
            for version in ['baseline','improved']:
                for s in load(version,split)['sheets']:
                    loc=s['scores']['0.5']['localization'];edge=s['scores']['0.5']['connections']
                    writer.writerow(dict(version=version,split=split,sheet=s['id'],gt_symbols=s['gt_symbols'],pred_symbols=s['pred_symbols'],localization_precision=loc['precision'],localization_recall=loc['recall'],localization_f1=loc['f1'],gt_connections=s['gt_connections'],pred_connections=s['pred_connections'],connection_precision=edge['precision'],connection_recall=edge['recall'],connection_f1=edge['f1'],seconds=s['elapsed_seconds']))
    for version in ['baseline','improved']:
        r=load(version,'holdout');print(version,'mean seconds',statistics.mean(s['elapsed_seconds'] for s in r['sheets']))

if __name__=='__main__':main()
