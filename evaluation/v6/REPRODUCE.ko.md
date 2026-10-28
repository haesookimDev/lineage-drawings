# v6 재현 안내

평가 재실행에는 저장소의 고정 ONNX를 사용한다. 학습은 별도 환경에서 실행하며, 운영 이미지에 PyTorch를 추가할 필요가 없다. 원본 데이터는 Git에 포함하지 않는다.

## 공개 데이터 준비

저장소 루트에서 Python3.12 기반 기존 개발 환경과 requirements-dev.txt를 준비한 뒤 실행한다.

```bash
.venv/bin/python scripts/fetch_public_benchmark.py
.venv/bin/python scripts/fetch_v3_confirmation.py
.venv/bin/python evaluation/v6/fetch_data.py
```

PDF 해시는 v5 원본 메타데이터로 검증하며, 새 정답의2400px 기준 PNG 해시도 검사한다. 같은 PyMuPDF1.26.5로4800px 입력을 재구성한다. 정부 기관의 제3자 PDF 접근 가능성이 재배포 허가를 뜻하지 않는다.

## 고정 평가

원본 OCR은 Tesseract5.3.0/eng+kor/TSV다. 정확한 바이너리·라이브러리·학습 데이터 지문은 evaluation/ocr_fingerprint.json을 따른다. 이 작업 호스트의 `/tmp/lineage-ocr-bookworm/bin`은 운영과 같은 OCR을 실행하기 위해 준비했던 로컬 래퍼이며 다른 호스트에 자동으로 존재하지 않는다. 버전 이름만 같아도 지문이 다르면 결과가 달라질 수 있다.

```bash
export PATH=/tmp/lineage-ocr-bookworm/bin:$PATH  # 같은 OCR을 준비한 호스트에서만
export OMP_NUM_THREADS=1
.venv/bin/python evaluation/v6/evaluate.py --runs 3
.venv/bin/python evaluation/v6/render_report.py
```

평가기에는 backend·모델·주석·선정 설정 해시 검사가 있다. 테스트는 기존 프로젝트의 새 페이지4개 ROI와 역사적 회귀15장이다. 선택된 모델은16단계로,24단계 최종 모델과 다르다. 재학습 또는 runtime 수정 후 기존 FROZEN을 조용히 갱신해 같은 시험으로 표현하지 말고 별도 실험·고정 기록을 만든다.

## 학습 재현

별도 checkout을 권장한다. 아래 초기 가중치와 소스는 공식 YOLOX 배포다. 체크포인트는 로컬 `.benchmark-data`에 저장한다.

```bash
python3.12 -m venv .venv-ml
.venv-ml/bin/pip install -r evaluation/v6/requirements-training.txt --extra-index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/Megvii-BaseDetection/YOLOX.git .benchmark-data/yolox-source
git -C .benchmark-data/yolox-source checkout 6ddff4824372906469a7fae2dc3206c7aa4bbaee
curl -fL https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth -o .benchmark-data/yolox_nano_coco.pth
.venv/bin/python evaluation/v6/prepare.py
.venv-ml/bin/python evaluation/v6/train.py --epochs 24 --steps-per-epoch 96 --batch 4
.venv-ml/bin/python evaluation/v6/export.py --epoch 8
.venv-ml/bin/python evaluation/v6/export.py --epoch 16
.venv-ml/bin/python evaluation/v6/export.py --epoch 24
.venv/bin/python evaluation/v6/validate.py --epoch 8
.venv/bin/python evaluation/v6/validate.py --epoch 16
.venv/bin/python evaluation/v6/validate.py --epoch 24
```

초기 가중치 SHA는 TRAINING.json에 있고 실제 패키지 버전은 training-environment.json에 있다. 기본 prepare는607타일을 만들며 원천 주석1,259개가 모두 독립 학습 예제라는 뜻은 아니다. CPU 병렬 부동소수점 연산으로 재학습 가중치가 바이트까지 같을 것을 보장하지 않는다. 제공된 고정 ONNX가 이번 배포·평가의 기준이다.

새 모델을 선택할 때에는 MODEL-SELECTION.json의 출처별 목적함수와 데이터 역할을 유지한다. 기존 v5 외부 평가 페이지를 이번 학습에 편입했으므로 그 페이지를 새 보류셋으로 재사용하지 않는다. 포트·접점·Relationformer 실험은 별도 후속 범위다.
