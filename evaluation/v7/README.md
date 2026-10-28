# v7: 선·포트·접속점 인식

새로운 여섯 채널 모델은 실선·점선, 기호 경계 포트, T·4방 접속·비접속 교차의
관찰 근거를 생성한다. **배포 모드는 기존 v6 연결 판정을 유지한다.** 검증에서
새 추적기로의 대체·합집합이 기존 평균 연결 F1을 넘지 못했기 때문이다.
인식은 모두 `needs_review`이며 기호·연결의 승인이 포트·접점을 승인하지 않는다.

- 기호: 기존 v6 YOLOX-Nano 가중치·추론 해상도 그대로.
- 토폴로지 학습: OPEN100 0–3 + 기존 학습용 합성 12장, GraphML 투영으로 만든
  약한 정답 640패치와 직접 생성한 1,200장. 실제 도면 픽셀 정답이 아니다.
- 검증: OPEN100 4–5, Pahala20, 합성4장 + 절차적 장면160개. epoch4/8/16 비교.
- 새 평가: Virden9/11, Pahala19의 소규모 영역. 10기호·8연결·14포트·5접점.
  단일 주석자이며 기존 프로젝트의 새 페이지다. 새 프로젝트 일반화 증거가 아니다.
- 회귀: v6의 기존19장. 별도로 표시하며 블라인드 평가라고 부르지 않는다.
- 입력: OCR/뷰어2400px, 기호4800px, CPU OpenCV ONNX. PyTorch는 학습 전용.

`PLAN.json`, `LABELS-FROZEN.json`, `MODEL-SELECTION.json`, `FROZEN.json`이
데이터 역할·레이블·선택·평가 코드를 기록한다. `HEAD-VALIDATION.json`은
**생성 장면 픽셀** IoU/F1이다. port AP 또는 실제 도면 정확도로 해석하면 안 된다.
비접속 교차 head는 해당 검증 임계값0.5에서 F1=0이므로 기하학 규칙을 함께 쓰며,
점 없는 X의 비접속 관례는 사용자가 확인해야 한다.

## 재현

소스 준비 경로는 이전 v5/v6 문서와 `new-page-sources.json`을 따른다. 원본 엔지니어링
PDF·페이지·잘라낸 이미지는 재배포하지 않는다. 도면 저작권은 공개 호스팅과 별개다.

```bash
.venv/bin/python evaluation/v7/prepare.py
.venv-ml/bin/python evaluation/v7/train.py --epochs 16 --steps 100
.venv-ml/bin/python evaluation/v7/export.py --epoch 16
.venv/bin/python evaluation/v7/validate_heads.py
# epoch4/8/16 내보내기 후 각 검증
PATH=/tmp/lineage-ocr-bookworm/bin:$PATH OMP_NUM_THREADS=1 .venv/bin/python evaluation/v7/validate.py --epoch 16
# 고정 가중치·코드·레이블로 최종 평가
PATH=/tmp/lineage-ocr-bookworm/bin:$PATH OMP_NUM_THREADS=1 .venv/bin/python evaluation/v7/evaluate.py --runs 3
```

훈련 환경은 PyTorch2.9.0+cpu/ONNX1.19.1, 실행 환경은 requirements.txt 참조.
CPU 재학습의 완전한 비트 일치는 보장하지 않는다. `TRAINING.json`의 원본 코드 해시와
export SHA로 실행 가중치를 식별한다. 평가는 같은 OCR 바이너리/언어 데이터 해시가 필요하다.

## 저장과 화면

`elements[].properties.ports`에 정규화 포트 좌표·변·점수·검토 상태를 저장한다.
`topology_bbox`는 추출 당시 영역이다. `page.transform.connection_inference`에는
정규화 선·접속/교차 후보·기존 연결과 일치하는 관찰 경로를 저장한다.
DB 마이그레이션 없이 기존 JSON 컬럼으로 저장·조회·내보내기·검토 이력을 유지한다.

뷰어의 **선·포트·접점** 토글에서 실선/점선 후보·파란 포트·초록 접속·보라 비접속 교차를
확인한다. 연결은 실제 경로가 있는 경우 그 경로를, 그 외에는 기존 중심선을 표시한다.
영역을 수정하면 해당 요소의 이전 포트·연결 경로를 숨긴다. 별도 포트/접점 편집·승인 UI,
흐름 방향 확정·표준 의미 부여·설비 영향 분석은 이번 단계에 포함되지 않는다.
