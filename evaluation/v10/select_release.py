"""Select on explicit development only; do not load expanded test labels."""
import copy,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];folder=ROOT/'evaluation/v10'
baseline=json.loads((ROOT/'evaluation/v9/results/report.json').read_text())
required={(s['group']+'-preservation',s['id']):s['variants']['v9']['scores']['connections']['tp'] for s in baseline['sheets']}
ranked=[]
for filename in ['ROUTE-VALIDATION.json','STYLE-VALIDATION.json','THRESHOLD-VALIDATION.json']:
    data=json.loads((folder/filename).read_text());configs={c['name']:c for c in data['configs']}
    for name,cs in data['aggregate'].items():
        for config,groups in cs.items():
            old=groups['open100-repair']['connections']
            selected=[r for r in data['rows'] if r['model']==name and r['config']==config]
            preservation={(r['group'],r['id']):r['scores']['connections']['tp'] for r in selected if r['group'].endswith('preservation')}
            gates={'open100_recall_at_least_v8':old['tp']>=343,'open100_f1_at_least_v8':old['f1']>=686/1304,
                   'each_previous_roi_recall_retained':all(preservation.get(k,-1)>=tp for k,tp in required.items())}
            objective=sum(v['connections']['f1']+.25*v['localization']['f1'] for v in groups.values())/len(groups)
            ranked.append({'file':filename,'model':name,'config':configs[config],'objective':objective,'gates':gates,'eligible':all(gates.values()),'groups':groups,'preservation_tp':{f'{g}-{i}':tp for (g,i),tp in preservation.items()}})
ranked.sort(key=lambda r:(r['eligible'],r['objective']),reverse=True)
selected=ranked[0];assert selected['eligible'], 'No release candidate satisfies preservation/repair gates'
name=selected['model'];fraction=float(name.split('-')[1])
source=ROOT/'.benchmark-data/v10/models'/f'blend-{fraction}.onnx'
meta=json.loads(source.with_suffix('.json').read_text())
if '-cut' in name:meta['threshold']=float(name.split('-cut')[1])
meta.update(local_style_gate=True,selection='Explicit regression repair selection on previous20 development pages/ROIs. New enlarged labels withheld; prior35 labels and OPEN1006-11 are now development. No gradient training/ensemble. See evaluation/v10/RELEASE-SELECTION.json.')
shutil.copyfile(source,ROOT/'backend/assets/symbol-neural.onnx')
(ROOT/'backend/assets/symbol-neural.json').write_text(json.dumps(meta,indent=2)+'\n')
context=json.loads((ROOT/'backend/assets/topology-context.json').read_text());context.update({k:v for k,v in selected['config'].items() if k!='name'})
for k in ('extra_solid_threshold','extra_junction_threshold'):
    if not context.get(k):context.pop(k,None)
context.update(routing_revision='v10-local-style-and-replay',release_policy='Single-model parameter interpolation with development repair/preservation gates; same image-supported periodic recovery and human review. Local style search replaces unused full-page template proposals.')
(ROOT/'backend/assets/topology-context.json').write_text(json.dumps(context,indent=2)+'\n')
record={'role':'Development selection, no enlarged test label access. Source groups are7 development strata, not7 independent datasets/projects. Prior4 test pages/35 labels now preservation development.','validated_candidate_set':'neural_only','symbol_configuration':meta,'topology_configuration':context,'selected':selected,'ranked':ranked}
(folder/'RELEASE-SELECTION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in selected.items() if k!='groups'},indent=2))
