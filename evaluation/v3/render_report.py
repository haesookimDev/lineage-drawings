#!/usr/bin/env python3
"""Render final measurements without changing models, thresholds, or scoring."""
from pathlib import Path
import csv,json,os,statistics,sys,tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'lineage-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import load_gt,match_boxes
from backend.extract import infer_connections
BASE=ROOT/'evaluation/v3';FIG=BASE/'figures';FIG.mkdir(exist_ok=True)
def load(name):return json.loads((BASE/'results'/name/'report.json').read_text())
def pct(n):return f'{n*100:.2f}%'
def main():
    old,new=load('v2-regression'),load('v3-regression')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    sets=[('Real regression (8)','regression'),('Synthetic stress (8)','stress'),('Fresh synthetic (8)','confirmation')]
    for axis,metric,title in zip(axes,['localization','connections'],['Symbol localization F1 (IoU >= .50)','Equipment adjacency F1']):
        for i,(label,role) in enumerate(sets):
            for off,version,color in [(-.18,'v2','#9ca3af'),(.18,'v3','#0f766e')]:
                value=load(version+'-'+role)['aggregate']['0.5'][metric]['f1']
                axis.bar(i+off,value,width=.34,color=color,label=version if i==0 else None)
                axis.text(i+off,value+.012,f'{value*100:.2f}%',ha='center',fontsize=9)
        axis.set(title=title,ylim=(0,.92),xticks=range(3),xticklabels=[x[0] for x in sets]);axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    axes[0].legend(frameon=False);fig.suptitle('Final gated v3 | Real-drawing gain; unsupported synthetic conventions retain v2',fontsize=14)
    fig.text(.5,-.02,'Real set is historical regression, not a new blind test. Data: PID2Graph / OPEN100 / Digitize-PID, CC BY-SA 4.0.',ha='center',fontsize=9)
    fig.savefig(FIG/'final-metrics.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Same illustration sheet as the live deployment test; no best-sheet picking.
    ident='4';source=ROOT/f'.benchmark-data/open100/{ident}.png'
    with Image.open(source) as raw:original=raw.size;im=raw.convert('RGB');im.thumbnail((2400,2400))
    gt,_,_=load_gt(ROOT/f'.benchmark-data/open100/{ident}.graphml',original)
    fig,axes=plt.subplots(1,2,figsize=(18,8),layout='constrained')
    for axis,version in zip(axes,['v2','v3']):
        p=json.loads((BASE/f'results/{version}-regression/{ident}-predictions.json').read_text());symbols=[e for e in p['elements'] if e['kind']!='text'];mapping=match_boxes(symbols,gt,.5)
        axis.imshow(im);axis.axis('off')
        for e in symbols:
            a,b,c,d=e['bbox'];axis.add_patch(Rectangle((a*im.width,b*im.height),(c-a)*im.width,(d-b)*im.height,fill=False,linewidth=.9,edgecolor='#16a34a' if e['id'] in mapping else '#dc2626'))
        for e in gt:
            if e['id'] in mapping.values():continue
            a,b,c,d=e['bbox'];axis.add_patch(Rectangle((a*im.width,b*im.height),(c-a)*im.width,(d-b)*im.height,fill=False,linewidth=.8,linestyle='--',edgecolor='#2563eb'))
        axis.set_title(f'{version} | TP {len(mapping)}, FP {len(symbols)-len(mapping)}, FN {len(gt)-len(mapping)}')
    fig.suptitle('OPEN100 sheet 4 | green: matched | red: false proposal | blue dashed: missed symbol',fontsize=14)
    fig.text(.5,-.02,'Source: Stuermer et al., Energy Impact Center OPEN100; adapted overlays: LINEAGE. CC BY-SA 4.0.',ha='center',fontsize=10)
    fig.savefig(FIG/'sheet4-symbols.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for axis,kind in zip(axes,['tee','plain','dot']):
        im=Image.new('RGB',(100,100),'white');d=ImageDraw.Draw(im);lines=[[15,50,85,50],[50,15,50,50 if kind=='tee' else 85]]
        for line in lines:d.line(line,fill='black',width=2)
        if kind=='dot':d.ellipse((46,46,54,54),fill='black')
        els=[{'id':name,'kind':'unknown','bbox':box} for name,box in [('L',[.05,.4,.15,.6]),('R',[.85,.4,.95,.6]),('T',[.4,.05,.6,.15]),('B',[.4,.85,.6,.95])]]
        if kind=='tee':els=els[:3]
        edges=infer_connections(els,lines,100,100,image=im);axis.imshow(im);axis.axis('off');axis.set_title({'tee':'T-junction','plain':'Plain crossing','dot':'Dotted junction'}[kind]+f'\n{len(edges)} adjacent pairs')
        for e in els:
            a,b,c,dd=e['bbox'];axis.add_patch(Rectangle((a*100,b*100),(c-a)*100,(dd-b)*100,fill=False,edgecolor='#0f766e',linewidth=2))
    fig.suptitle('Tested routing convention inside supported-style v3 (not a universal drafting rule)',fontsize=12)
    fig.savefig(FIG/'junction-policy.png',dpi=160,bbox_inches='tight');plt.close(fig)
    with (BASE/'per-sheet.csv').open('w') as f:
        cols=['role','version','sheet','symbols_tp','symbols_fp','symbols_fn','symbols_f1','edge_tp','edge_fp','edge_fn','edge_f1','seconds']
        writer=csv.DictWriter(f,cols,lineterminator='\n');writer.writeheader()
        for role in ['development','regression','stress','confirmation']:
            for version in ['v2','v3']:
                for s in load(version+'-'+role)['sheets']:
                    loc=s['scores']['0.5']['localization'];edge=s['scores']['0.5']['connections'];writer.writerow(dict(role=role,version=version,sheet=s['id'],symbols_tp=loc['tp'],symbols_fp=loc['fp'],symbols_fn=loc['fn'],symbols_f1=loc['f1'],edge_tp=edge['tp'],edge_fp=edge['fp'],edge_fn=edge['fn'],edge_f1=edge['f1'],seconds=s['elapsed_seconds']))
    mean=lambda r:statistics.mean(s['elapsed_seconds'] for s in r['sheets'])
    summary={role:{version:load(version+'-'+role)['aggregate']['0.5'] for version in ['v2','v3']} for role in ['development','regression','stress','confirmation']}
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table='| 지표 | v2 P / R / F1 | 최종 v3 P / R / F1 |\n|---|---|---|\n'
    for key,label in [('localization','기호 위치 · IoU 0.50'),('connections','연결 · end-to-end'),('oracle_boxes','정답 영역 연결 진단')]:
        a,b=old['aggregate']['0.5'][key],new['aggregate']['0.5'][key];table+=f"| {label} | {' / '.join(pct(a[k]) for k in ['precision','recall','f1'])} | {' / '.join(pct(b[k]) for k in ['precision','recall','f1'])} |\n"
    locold,locnew=old['aggregate']['0.5']['localization'],new['aggregate']['0.5']['localization'];eold,enew=old['aggregate']['0.5']['connections'],new['aggregate']['0.5']['connections']
    per='| 실제 도면 ID | v2 기호 F1 | v3 기호 F1 | v2 연결 F1 | v3 연결 F1 |\n|---|---:|---:|---:|---:|\n'
    for a,b in zip(old['sheets'],new['sheets']):per+=f"| {a['id']} | {pct(a['scores']['0.5']['localization']['f1'])} | {pct(b['scores']['0.5']['localization']['f1'])} | {pct(a['scores']['0.5']['connections']['f1'])} | {pct(b['scores']['0.5']['connections']['f1'])} |\n"
    report=f'''# 기호 검출·접속점·교차 판별 v3 평가와 운영 반영

평가일: 2026-10-28 · 배포 앱: `2e111c42f33a02fb253b82a49e993c661e690ef3`

**기존 공개 실제 도면 8장에서 연결 F1은 {pct(eold['f1'])}→{pct(enew['f1'])}, 기호 위치 F1은 {pct(locold['f1'])}→{pct(locnew['f1'])}로 개선됐다.** 연결 정밀도는 {pct(eold['precision'])}→{pct(enew['precision'])}로 올랐지만 재현율은 {pct(enew['recall'])}다. 자동 영향 분석에 충분한 연결 완전성은 확보하지 못했다.

지원되는 기호 형상과 연속 표지가 함께 확인된 스타일에 새 경로를 사용한다. 그 외에는 기호·선분·연결까지 v2 경로를 유지한다. **모든 도면에서 새 모델이 향상됐다는 결과는 아니다.**

## 데이터 분리와 실패 결과 보존

원본은 [PID2Graph v1](https://zenodo.org/records/14803338)이다. OPEN100은 Energy Impact Center의 공개 실제 설계 문서이고 Dataset PID는 [Digitize-PID](https://arxiv.org/abs/2109.03794)의 합성 도면이다. 합성을 실제 도면으로 표현하지 않았다.

- **개발·모델 구축:** OPEN100 0–3. 이 네 장의 밸브·펌프·탱크 주석 영역에서 73개 형상 프로토타입을 구성했다. 개발 점수는 학습에 사용한 도면의 결과다.
- **역사적 실제 회귀:** OPEN100 4–11, 기호 527개·인접 관계 1,393개. 이전 작업에서 결과를 이미 본 도면이므로 이번에는 새 블라인드 테스트라고 부르지 않는다.
- **초기 합성 스트레스 8장:** `48,303,277,24,173,95,313,320`. 무조건 v3를 적용했을 때 연결 F1이 2.23%→0.72%, 위치 F1이 25.09%→23.08%로 떨어져 해당 안을 폐기했다. [폐기한 결과](results/rejected-v3-confirmation/report.json)와 [초기 고정본](FROZEN.initial.json)을 남겼다. 이 집합은 이후 대체 경로의 개발·회귀 검증에 사용했으며 더 이상 보류셋이 아니다.
- **최종 새 합성 확인 8장:** `9,34,385,285,317,281,82,382`. 앞의 실패 집합을 제외하고 사전 SHA-256 순서의 다음 8장을 선정했다. [선정 계획](confirmation-plan.json)과 [최종 고정본](FROZEN.json)을 저장한 뒤 내려받아 평가했다.

최종 스트레스 셋은 v2와 같은 위치 F1 25.09%·연결 F1 2.23%, 새 확인 셋도 v2와 같은 위치 F1 31.36%·연결 F1 2.75%였다. 미지원 스타일이 정확히 기존 방식으로 돌아가 성능 저하를 막은 결과이며, 이 두 합성 집합에서 인식 성능 자체를 향상시킨 것은 아니다. 새로운 회사·프로젝트의 실제 도면에 대한 독립 일반화 검증은 남아 있다.

## 같은 기준의 실제 도면 회귀 결과

기존과 같은 5종 대상(`valve`, `instrumentation`, `tank`, `pump`, `inlet/outlet`), 클래스 무관 IoU 0.50 일대일 매칭, 무방향 기호 인접 관계를 사용했다. 정답의 connector·crossing·arrow·general 경로는 축약하되 중간 대상 기호를 통과하지 않는다. 매칭되지 않은 예측 끝점도 연결 FP로 센다. 기준을 완화해 점수를 올리지 않았다.

{table}
기호 TP/FP/FN은 {locold['tp']}/{locold['fp']}/{locold['fn']}→{locnew['tp']}/{locnew['fp']}/{locnew['fn']}, 연결은 {eold['tp']}/{eold['fp']}/{eold['fn']}→{enew['tp']}/{enew['fp']}/{enew['fn']}이다. 연결 FP가 약 {(1-enew['fp']/eold['fp'])*100:.1f}% 감소했다.

정답 영역을 준 진단에서는 **연결 F1이 61.05%→58.53%로 낮아졌다.** 대신 정밀도가 78.62%→92.97%로 올랐다. 보수적인 교차 분리가 재현율을 희생하는 교환 관계가 있으므로, 연결기만 모든 지표에서 향상됐다고 해석하면 안 된다. 실제 end-to-end 개선은 기호 검출 보강과 연결 오검출 억제의 조합이다. 이 진단에 정답 영역을 쓰는 것은 운영 인식률이 아니다.

![최종 지표](figures/final-metrics.png)

{per}
도면마다 편차가 크며 일부 연결 F1은 낮아질 수 있다. 원시 결과는 [results](results/), 도면별 수치는 [CSV](per-sheet.csv), 집계 수치는 [summary.json](summary.json)에 있다.

![고정 4번 도면의 기호 검출 비교](figures/sheet4-symbols.png)

## 실제 변경

- [형상 보강](../../backend/symbols.py): 개발 4장만 사용한 73개 작은 템플릿, 3개 크기에서 정규화 상관 매칭, 외부 선 근거와 중복 억제. 원 후보의 내부 문자 조각 근거와 끝이 뾰족한 연속 표지의 기하 조건도 사용한다. 거대한 닫힌 배관 영역이나 안내문을 기호로 받아 선망을 끊는 경우를 줄였다.
- [포트 그래프](../../backend/network.py): 기호의 선 접점을 그래프 노드로 분리하고, 다른 기호를 만나면 탐색을 멈춘다. T자·선 끝점 접속은 잇고, 4방 교차는 주변 선 두께에 비해 중심 잉크가 두꺼운 접속점 근거가 있을 때 잇는다. 점 없는 X는 두 경로를 분리한다.
- **스타일 보호:** 3개 이상의 형상 매칭과 2개 이상의 상자형 연속 표지가 함께 있어야 새 형상·래스터 연결 경로를 활성화한다. 파일명·데이터셋 ID로 분기하지 않는다. 부족하면 [v2 선분/연결 처리](../../backend/legacy_network.py)를 그대로 사용한다. 이는 표기 관례를 확정한 판별기가 아니라 보수적인 지원 조건이다.
- **검토 가능한 출처:** 페이지 변환 정보에 적용 알고리즘·교차 정책·제한 도달 여부를 저장하고, UI에 포트·접속점 기반 연결 출처를 표시한다. 무점 교차의 비접속 가정과 일반 경로 사용 사실을 안내한다.
- **자원 상한:** OpenCV 스레드 2개, 후보·선분·교차 비교 예산 유지. 연결 500개 상한의 선택 순서가 무작위 요소 ID에 따라 달라지지 않게 고쳤다.

![지원 스타일에서 사용하는 접속 규칙 테스트](figures/junction-policy.png)

이 규칙은 모든 도면 표준에 공통인 법칙이 아니다. 점 없는 4방 교차를 접속으로 쓰는 회사 관례, 점선·점프 아크·가려진 선·사선 래스터 배관은 추가 모델이나 프로젝트 설정이 필요하다. 지원 조건에 우연히 들어오는 다른 표기 관례도 있을 수 있다. 모든 자동 결과는 계속 `needs_review`, 요소 종류는 `unknown`이며, `shape_hint`는 표준 기호 식별자가 아니다. 의미/표준 자동 분류 성공률을 주장하지 않는다.

## 검증의 신뢰성·재현

이전 배포 v2는 `3bb6ff7`의 **전체 backend와 자산을 격리해 로드**했다. extract.py만 과거 파일로 읽고 현재 geometry.py를 가져오는 실수를 방지하는 회귀 테스트를 추가했다. v2의 기존 연결 F1 11.36%를 그대로 재현했다. 정답 영역 진단도 실제 영상에서 선택한 지원/대체 경로를 그대로 따른다.

후속 확인 결과를 본 뒤 최종 파라미터를 변경하지 않았다. 초기 실패 후 바꾼 지원 조건과 ID 독립적인 연결 상한 수정은 `FROZEN.json`에 명시했다. 모델과 런타임·평가기 SHA-256도 그 파일과 각 결과 JSON에 있다. 개발 이외 도면의 주석이나 기업 운영 도면으로 템플릿을 만들지 않았다.

Tesseract 5.3.0과 기존 eng/kor 데이터·TSV 설정, OpenCV 4.12.0, Pillow 12.0.0, Python 3.12를 사용했다. 각 평가 도면에서 OCR 결과가 양수임을 검사했다. 최대 변 2400px의 전체 이미지를 사용했고 새 16장 원본은 모두 운영 50MP 제한 이내다. 라이선스·원본 해시는 [데이터 NOTICE](../NOTICE.md), [모델 NOTICE](../../backend/assets/NOTICE.md), [스트레스 출처](stress-data.json), [새 확인 출처](confirmation-data.json)에 있다. 파생 결과/그림과 모델 자산은 CC BY-SA 4.0이다.

회귀 테스트 **24개**가 통과했다. 무점 X의 두 경로 유지, 점이 있는 접속, T자, 끊긴 간격, 중간 기호 장벽, 미지원 스타일의 v2 동작, 모델 출처, 역사적 버전 격리, ID를 바꿔도 같은 연결 500개 선택을 포함한다. [같은 앱 커밋 CI](https://github.com/haesookimDev/lineage-drawings/actions/runs/40633332847)도 성공했다.

## 처리 시간의 교환 관계

실제 회귀 8장의 로컬 OCR+추출 평균은 v2 {mean(old):.3f}초→v3 {mean(new):.3f}초였다. 형상 탐색 때문에 지원 스타일의 처리 시간이 늘었다. 새 합성 셋의 대체 경로는 v2 {mean(load('v2-confirmation')):.3f}초, 최종 v3 {mean(load('v3-confirmation')):.3f}초였다. 미지원 스타일은 비싼 템플릿 탐색 전에 빠져나간다. 이 값은 단일 로컬 실행 관찰치이며 서비스 SLA나 부하 시험이 아니다.

## 운영 반영

`2e111c42f33a02fb253b82a49e993c661e690ef3`의 CI 컨테이너를 실제 k3s에 배포했다. 실행 Pod에서 코드·모델 10개 파일의 해시가 최종 고정본과 같고, T자/무점 X/점 있는 교차의 규칙 테스트도 통과했다. 외부 서비스의 공개 도면에서 평가와 같은 기호 영역·연결 쌍 및 대체 경로를 확인했다. 상세 기능 검증은 [배포 검증 기록](../../DEPLOYED_V3_VERIFICATION.ko.md)에 있다.

```bash
export OMP_NUM_THREADS=1
.venv/bin/pip install -r requirements-eval.txt
.venv/bin/python scripts/fetch_public_benchmark.py
.venv/bin/python scripts/build_shape_prototypes.py
.venv/bin/python scripts/fetch_v3_confirmation.py
.venv/bin/python evaluation/benchmark.py --extractor-ref 3bb6ff7 --split holdout --role historical-regression --output evaluation/v3/results/v2-regression
.venv/bin/python evaluation/benchmark.py --split holdout --role historical-regression --output evaluation/v3/results/v3-regression
.venv/bin/python evaluation/benchmark.py --extractor-ref 3bb6ff7 --split all --ids 9,34,385,285,317,281,82,382 --role unseen-synthetic-confirmation --data .benchmark-data/pid-confirmation --output evaluation/v3/results/v2-confirmation
.venv/bin/python evaluation/benchmark.py --split all --ids 9,34,385,285,317,281,82,382 --role unseen-synthetic-confirmation --data .benchmark-data/pid-confirmation --output evaluation/v3/results/v3-confirmation
.venv/bin/python evaluation/v3/render_report.py
```

스트레스 8장은 `stress-plan.json`의 ID와 원본 경로로 받아 같은 명령에 넣을 수 있다. 입력 PNG·GraphML은 Git에서 제외하며 결과 JSON과 파생 그림만 배포한다. 위 명령은 이미 공개한 확인 집합의 재현을 위한 것이고, 재실행을 새 독립 실험으로 부르지 않는다.

다음 우선순위는 다른 실제 프로젝트의 고정 평가셋, 더 넓은 기호 검출 학습 데이터, 기호 포트·접속/비접속 교차의 별도 주석과 학습 모델이다. 이번 지원 조건을 광범위한 관례 판별기로 확대해 주장하지 않는다.
'''
    (BASE/'REPORT.ko.md').write_text(report)
    print('Rendered final v3 report and three figures.')

if __name__=='__main__':main()
