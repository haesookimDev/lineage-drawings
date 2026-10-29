# v10 재현과 데이터 역할

[평가 보고서](REPORT.ko.md) · [모델 카드](MODEL-CARD.ko.md) · [고정 런타임](FROZEN.json) · [선택](RELEASE-SELECTION.json)

고정 모델은 `backend/assets/symbol-neural.onnx`에 포함한다. SHA-256은
`64aa437392cc5d02134611bdae950dd1814e09d1f8c50b8baa79f1da59a300b5`다.
운영에는 PyTorch·학습 자료가 필요 없다.

## 역할

- 추가 경사 학습 없음. v6 epoch16과 v9 scale epoch8의 부동소수 텐서를0.5로 결합한다. 원천 경사 학습은 이전20실제 페이지/부분 ROI와12합성장,1,332원천 기호·769파생 타일로 동일하다. 파생 타일을 독립 도면 수로 세지 않는다.
- 기존10검증 페이지와 OPEN1006–11 및 이전 새 도면35정답은 명시적인 개발/보존 선택 자료다. OPEN1006–11을 새 테스트로 부르지 않는다.
- 확대 정답78기호·102연결은 선택에 사용하지 않았다. 이전 전체 이미지·프로젝트와35정답 노출은 존재한다. 확장 정답만 보류했으며 새 프로젝트 블라인드 테스트가 아니다.
- 정부 원본·오버레이·OCR·운영 데이터는 Git/이미지에 없다. 원본 URL/해시는 `../v9/NEW-SOURCES.json`, 렌더 해시는 `../v9/SOURCE-PAGES.json`, 학습 분리는 `SPLIT-AUDIT.json`에 있다.

## 최종 평가

저장소 루트에서 Python3.12와 `requirements-dev.txt` 및 운영과 같은 Tesseract5.3/eng/kor를 사용한다. 이번 측정은 `.venv`의 OpenCV4.12/NumPy2.2.6/Pillow12/PyMuPDF1.26.5를 썼다. 상세 버전은 `FROZEN.json`과 `../ocr_fingerprint.json`에 있다. 원본은 무시된 `.benchmark-data/`에 있어야 한다.

```bash
PATH=/tmp/lineage-ocr-bookworm/bin:$PATH OMP_NUM_THREADS=1 .venv/bin/python evaluation/v10/evaluate.py --scope expanded --runs 3
PATH=/tmp/lineage-ocr-bookworm/bin:$PATH OMP_NUM_THREADS=1 .venv/bin/python evaluation/v10/evaluate.py --scope repair --runs 3
PATH=/tmp/lineage-ocr-bookworm/bin:$PATH OMP_NUM_THREADS=1 .venv/bin/python evaluation/v10/evaluate.py --scope synthetic --runs 1
MPLCONFIGDIR=/tmp/lineage-mpl .venv-ml/bin/python evaluation/v10/render_report.py
.venv/bin/python -m pytest -q
```

각 평가 실행은 현재 backend/모델/정답 지문을 먼저 확인한다. 런타임을 바꾸고 고정 검사를 우회해 같은 릴리스 결과라고 부르면 안 된다. 원본에서 보기2400과기호4800 영상을 독립 축소한다. 실제10장의60회만3회 반복 속도 비교이며 합성16회는 품질 회귀다. 네트워크·PDF렌더·저장은 측정 밖이다.

결과: `results/`, `repair/`, `synthetic/`, 장별 `per-page.csv`. 정부 페이지의 공개 예측에는 문자/라벨을 제거하고 후보 박스·관찰 근거만 둔다. 기존 v8 확장 참고는 `V8-EXPANDED-REFERENCE.json`의 고정 v9 실행 예측을 재채점한 것이며 새 시간 측정이 아니다.

## 개발 실행 이력

모든 개발 실행은 확대 정답을 읽지 않는다. `data.py`의 `development()`는 이전20개 개발 페이지/ROI만 공급한다. 신경망 체크포인트 재생성에는 기존 v6/v9 학습 스크립트·원천 파일·기존 체크포인트가 필요하다. 실험용 `.venv-ml`의 Torch/ONNX는 운영 이미지에 없다.

1. `export_blends.py`: 기존 체크포인트 텐서 결합과 Torch/OpenCV 출력 비교. `MODEL-CANDIDATES.json`.
2. `validate_symbols.py`:0/0.25/0.5/0.75/1 계수의 기호 검출 비교. `SYMBOL-VALIDATION.json`.
3. `validate_routes.py`: 최초 로컬 NN 게이트 결과는 `ROUTE-VALIDATION-ITERATION1.json`. 형상 힌트와 실제 통과 접점 점수를 추가한 비교는 `ROUTE-VALIDATION.json`. 원시 캐시는 무시된 `.benchmark-data/v10/route-screen/`에 둔다.
4. `validate_line_styles.py`: 실제 반복 점선 증거에 따른 문턱 분리 비교. `STYLE-VALIDATION.json`.
5. `validate_thresholds.py`: 마지막 제한된 추가 실험으로0.25/0.5 계수에0.45 기호 문턱. `THRESHOLD-VALIDATION.json`.
6. `select_release.py`: OPEN100 v8 이상, 이전4영역 각각의TP유지 조건을 만족한 후보 중7개 개발 층의 평균 목적값으로 선정. `RELEASE-SELECTION.json`.7개 층은7개 독립 출처가 아니다.
7. `annotate_expanded.py`: 모델 출력을 읽지 않고 원본에서 주석, 원래35노드/35연결 보존. 확대 채점 전 `LABELS-FROZEN.json`으로 고정. 확대 결과를 보고 재선정하지 않았다.

사후 결과에서는 확대 R34.31→33.33%, 합성 F148.17→48.00%의 하락도 그대로 공개한다. 모든 출력은 검토 후보이며 공학적 접속·물리 흐름·표준 ID를 자동 확정하지 않는다.
