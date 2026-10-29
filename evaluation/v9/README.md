# v9 재현과 데이터 역할

최종 평가·차트·실패는 [REPORT.ko.md](REPORT.ko.md), 실행 바이트는
[FROZEN-RELEASE.json](FROZEN-RELEASE.json), 배포 증거는
[DEPLOYED.json](DEPLOYED.json)을 따른다.

기준선은 실제 운영 v8-R2 앱 커밋 c9cef5afc6363e30c86e9ab4a5fe17dd68313e11다.
문서만 추가한 뒤의 HEAD를 기준선 실행으로 쓰지 않는다. 역사적 backend 전체를
Git에서 별도 Python 패키지로 로드해 현재 모듈이 섞이지 않도록 한다.

## 공개 원본과 역할

기존 데이터는 evaluation/v6/REPRODUCE.ko.md, v7/fetch_pages.py,
v8r2/fetch_pages.py를 따른다. 새 자료는 다음 명령으로 고정 SHA를 확인하며 받는다.
정부 PDF·페이지·잘라낸 이미지·전체 OCR을 Git이나 이미지에 재배포하지 않는다.

```bash
.venv/bin/python evaluation/v9/fetch_pages.py
```

새 자료집2개는 실제3개 사업을 포함한다. Winkler WWTP project60430450의
05-N001/05-N009와 Winnipeg 자료집의 NEWPCC BNR/WEWPCC WEP-91이다.
학습은 OPEN100/Pahala/Virden/Ruttan만 사용한다. 파일 SHA·프로젝트 역할은
SPLIT-AUDIT.json에 있다. 기존 프로젝트의 이전 테스트를 학습으로 옮긴 이력도
TRAINING-DATA.json에 명시하며 새 프로젝트 평가로 세지 않는다.

새 정답은 단일 AI 주석자가 원본을 보고 옮긴4개 작은 ROI의35기호·35연결이다.
전문가 인증 정답이나 전체 페이지 주석이 아니다. 명시적으로 제외한 모호한 조립체,
태그 타원·관로 축소부·지지 브래킷은 new-project-labels.json을 따른다.
Winnipeg7은 복잡한 제어 조립체의 내부 경계가 모호해 추론 전에3기호의 작은
점선 계기 연쇄를 선택했다. 정확도 전체화를 피하고 장별 결과를 제공한다.

## 최종 고정 평가

정답·가중치·문턱값·추적 규칙은 새 자료 추론 전에 선택했다. 최초 실행에서
sidecar의 merge_mode=replace가 그대로 복사되어, 신경망 후보만 쓴 검증과
실행 후보 집합이 달랐다. 합성 회귀에서 발견했고 아직 배포하지 않은 후보였다.
원래 선택을 구현하도록 neural_only로 교정하고 동일 정답·가중치·문턱값으로
다시 실행했다. 초기 결과는 invalid-merge-run/에 보존하며 최종 수치로 쓰지 않는다.
이 경위 때문에 수정 후 첫-ever 노출이라고 표현하지 않는다. 새 프로젝트는 여전히
학습·문턱값 선택에 사용하지 않았다. VALIDATION-CONTRACT-FIX.json 참조.

동일 Tesseract 바이너리·라이브러리·eng/kor 데이터 지문은
../ocr_fingerprint.json을 따른다. 아래 래퍼는 이 호스트에 준비된 경로이며
다른 호스트에서 자동 제공되는 도구가 아니다. 버전 이름만 같아도 충분하지 않다.

```bash
PATH=/tmp/lineage-ocr-bookworm/bin:$PATH OMP_NUM_THREADS=1 \
  .venv/bin/python evaluation/v9/evaluate_release.py --scope new --runs 3
PATH=/tmp/lineage-ocr-bookworm/bin:$PATH OMP_NUM_THREADS=1 \
  .venv/bin/python evaluation/v9/evaluate_release.py --scope regression --runs 1
# 7168px 합성 입력의 제품과 같은 독립 미리보기 축소는
# evaluate_regression.py와 REGRESSION-PROTOCOL.json을 따른다.
PATH=/tmp/lineage-ocr-bookworm/bin:$PATH OMP_NUM_THREADS=1 \
  .venv/bin/python evaluation/v9/evaluate_regression.py --scope regression --runs 1
.venv/bin/python -m evaluation.v9.merge_regression
```

새4장×기존/기호만 교체/최종3버전×3반복=36회다. 세 버전의 순서를 반복마다
회전시키고 박스·연결·포트·접점·관찰 경로의 결정성을 검사한다. 모든 입력은
전체 페이지이며 지정 ROI 내 모든 후보를 센다. IoU0.5 일대일 기호 대응,
중간 기호에서 정지하는 무방향 연결을 평가한다. 기호 누락도 연결 FN에 포함한다.
점선·가지 부분집합은 재현율만 별도 집계하며 독립 표본 수로 더하지 않는다.

시간은 OCR+최대4800px 기호+2400px 추적/인식을 포함하고 디코딩·PDF렌더·네트워크·
저장은 제외한다. CPU OpenCV2threads/OMP1, 장별3회 중앙값의 평균이다.
운영 SLA나 동시 처리량 측정이 아니다. 기존17장 회귀는 각1회로 품질 점검하며,
그 시간 한 번을 새4장3반복 처리 시간과 섞지 않는다.

## 학습과 선택

학습 환경은 training-environment.json. 운영에는 Torch를 설치하지 않는다.
기존 v6 epoch16을 먼저 재현하거나 로컬 보존 체크포인트를 사용한다.

```bash
.venv/bin/python evaluation/v9/prepare.py
OMP_NUM_THREADS=4 .venv-ml/bin/python evaluation/v9/train.py --epochs 12 --steps-per-epoch 64 --batch 4
# first-pass4를 내보낸 뒤 scale 학습은 그 체크포인트에서 시작한다.
OMP_NUM_THREADS=4 .venv-ml/bin/python evaluation/v9/train_scale.py --epochs 8 --steps-per-epoch 64 --batch 4
.venv-ml/bin/python evaluation/v9/export_scale.py --epoch 8
```

기존607타일에 이전 v6/v7/v8-R1 테스트73기호의162타일을 추가했다. 실제20장과
합성12장의 원천1,332기호·파생769타일이며 타일 수는 독립 도면 수가 아니다.
20장은 일부 ROI 주석과 상관된 페이지/개정판을 포함한다. 학습에 운영 업로드를
쓰지 않았다. 초기12단계 후4체크포인트에서 축소0.35–1.0·무작위 위치 증강으로
8단계 학습했다. 실제 작은 기호의 픽셀 크기와 과도하게 확대된 ROI의 차이를 줄인다.

원래10개 검증 페이지·4개 출처 그룹만으로 두 단계 선택을 했다. 기호 macro F1로
두 체크포인트 계열을 줄인 후 macro(connectionF1+.25symbolF1)를 비교했다.
이전 후보와 기준선을 계속 포함했고 세 개발 반복 결과를 모두 보존했다.
최종 scale8/확대0/문턱0.6 + recover-recall이다. 확대 재검출·형상 포트 힌트·짧은
두 점선 직접 연결·추적기 대체 실험도 기록했지만 최종 선택에 쓰이지 않은
기능을 배포 효과로 주장하지 않는다. 최종 파이프라인은 neural_only 계약을 따른다.

재학습은 CPU 병렬 부동소수점 차이로 바이트 일치를 보장하지 않는다.
제공된 ONNX와 고정 해시가 이번 평가·배포의 기준이다. 원본/코드/가중치 출처는
backend/assets/NOTICE.md를 따른다. 차트 생성은 개발 환경 matplotlib3.9.4를 사용한다.

합성 원본7168px의 미리보기는 제품처럼 원본에서2400px로 독립 축소한다.
초기 공통4800px 축소 진단은 PREBOUNDED-REGRESSION.json으로 분리하고
새4장(원본4800px)의 입력 픽셀이 변하지 않음을 검증했다. 최종 합성8장은
evaluate_regression.py로 다시 실행해 기존 실제6/검증3행과 병합했다.
