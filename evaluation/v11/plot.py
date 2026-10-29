import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent;dev=json.loads((root/'DEVELOPMENT.json').read_text());held=json.loads((root/'HELD-OUT.json').read_text());fig,ax=plt.subplots(1,2,figsize=(12,4.8));groups=['Expanded\ndevelopment','OPEN100\ndevelopment','White Rose\nnew-project pilot'];a=[dev['aggregate'][g]['baseline']['connections']['recall']*100 for g in ('expanded-development','open100-repair-development')]+[held['aggregate']['v10']['connections']['recall']*100];b=[dev['aggregate'][g]['short']['connections']['recall']*100 for g in ('expanded-development','open100-repair-development')]+[held['aggregate']['v11']['connections']['recall']*100]
for x,values,color,label in [(-.18,a,'#9bacb7','v10'),(.18,b,'#1a806b','v11')]:
 bars=ax[0].bar([i+x for i in range(3)],values,width=.34,label=label,color=color);ax[0].bar_label(bars,fmt='%.2f%%',padding=3,fontsize=9)
ax[0].set_xticks(range(3),groups);ax[0].set_ylim(0,100);ax[0].set_ylabel('Connection recall (%)');ax[0].legend();ax[0].set_title('Development gains; new pilot unchanged')
agg=held['aggregate'];values=[agg['v10']['mean_page_cold_median'],agg['v11']['mean_page_cold_median'],agg['v11']['mean_identical_image_warm_median']];bars=ax[1].bar(['v10 cold','v11 cold','v11 same-image\nOCR cache hit'],values,color=['#9bacb7','#1a806b','#7ab3a4']);ax[1].bar_label(bars,fmt='%.3fs',padding=4);ax[1].set_ylim(0,6);ax[1].set_ylabel('Seconds / page');ax[1].set_title('2 pages, 3 paired repeats; cache hit is separate')
fig.suptitle('LINEAGE v11 - limited AI-annotated public drawing pilot',fontsize=13);fig.text(.5,.01,'White Rose: 45 glyphs / 56 edges in 2 ROIs. Not expert-certified; no full-project accuracy or service SLA claim.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(root/'comparison.svg');plt.close(fig)
