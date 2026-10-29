# v8-R2: 실제 계기 버스 정답으로 보강한 최종 배포 후보

첫 후보v8-R1은 포트 오검출과 속도를 개선했지만, 처음 평가한 실제3장에서는
연결 재현율이 개선되지 않았다. 이 실패를 숨기지 않고 해당3장을 명시적으로
학습에 전환했다. R2는 실제 Pahala18 계기 버스의11포트·10검은 접속점을 포함해
학습하며, 새 평가는 Pahala14·Virden13/14의 별도 영역으로 분리했다.

학습에 운영 사용자 파일을 쓰지 않았다. 실제 수동 학습은 정부 문서8영역
(71포트·31개 양성/음성 접점) + OPEN1000–3에서 시각 검토한60점이다.
기존 OPEN1000–3/합성12장의 약한 GraphML 포트와 절차적300장도 사용한다.
모델2개는 CPU OpenCV RTrees160trees·depth18, 794차원 점 주변 특징이다.
기호 YOLOX-Nano는 이전 가중치·해상도 그대로이며 bbox를 바꾸지 않는다.

검증은 기존9장·5출처 그룹이고 정답/평가 범위도 유지했다. 최종 선택:
초기 R2: 포트0.65, 경로 접속0.25, 추가 경로 끝점0.75, 표시 접점0.45.
평균 연결 F1 기준선33.05% → 선택 후보39.66%. 동률은 수동 포트/접점 F1로 결정.
`VALIDATION.json`, `MODEL-SELECTION.json`에 모든 조합과 기준선이 있다.

`FROZEN.json`은 첫 R2 새 페이지 추론 전 코드/가중치/임계값/정답/평가기 해시다.
릴리스에 덧붙인 설명용 NOTICE 한 파일은 `RELEASE-NOTICES.json`으로 따로 검증한다.
그 추가는 실행·정답을 변경하지 않는다. R1 코드/가중치는 Git 커밋6836d0a에
그대로 남겼으며 새3장에서는 이 중간 후보도 함께 비교한다.

새 정답은 단일 주석자가 원본을 보며 전사한 작은 영역18기호·61연결·24포트·14접점이다.
Hough 원형 좌표는 주석 보조로만 사용하고 원본 위에서 확인했다. 공학 인증 GT가 아니다.
Pahala14의55연결은 한 공통 계기 버스의 상관된 쌍이며55개 독립 네트워크가 아니다.
기존 프로젝트의 새 페이지라서 독립 프로젝트 성능이나 전체 도면 정확도로 확장하지 않는다.

추론 전 제외: Pahala17의 겹쳐 붙은 이중 계기 기호는 부품 경계/기호 내부 연결을
안정적으로 주석하기 어려워 정량 영역에서 제외했다. OPEN10012/13도 검토하려 했지만
공식 고정 아카이브의 실제 OPEN100은0–11의12쌍뿐이어서 존재하지 않는 자료를 사용하지 않았다.
이 결정에는 새 모델 결과를 쓰지 않았고 PLAN.json에 남겼다.

검출 후보와 추적 접속 가설은 구분한다. 검출은 기호/문자 근접 제외, 팔 지지,
꺾임 제외, 점수순 좌표 억제·표시 임계값을 적용한다. 초기 R2의 낮은 점수 연결 가설은
`routing_contacts`에 별도 기록했다. 최종 릴리스는 실제 팔/문자 제외와0.45 임계값을 표시 접점과 통일하며 새 연결에 기호0.8·경로4점 상한을 적용한다. 긴 사슬의 오접속을 본 뒤 개발한 규칙이므로 최종 연결 결과는 사후 회귀다. `GRAPH-AUDIT.json`과 `STRICT-RELEASE-SELECTION.json`에 실패·선택 과정을 남겼다.

## 재현

이전 공개 자료 준비는 evaluation/v5/v6 문서, 새 페이지는 fetch_pages.py를 따른다.
공개 원본 PDF·페이지·잘라낸 이미지·전체 OCR은 Git과 Docker에서 제외한다.

```bash
OMP_NUM_THREADS=1 .venv/bin/python evaluation/v8r2/train.py
# 생성 가중치를 backend/assets로 복사하고 검증/선택
PATH=/tmp/lineage-ocr-bookworm/bin:$PATH OMP_NUM_THREADS=1 .venv/bin/python evaluation/v8r2/validate.py
# 재학습 비트 일치 대신 제공된 고정 가중치로 평가 권장
# 위 원래 전체 평가를 재현할 때는 a70bffd를 별도 checkout한다.
# 현재 최종 릴리스에서 원래 결과를 준비한 뒤:
PATH=/tmp/lineage-ocr-bookworm/bin:$PATH OMP_NUM_THREADS=1 .venv/bin/python evaluation/v8r2/replay_guard.py
PATH=/tmp/lineage-ocr-bookworm/bin:$PATH OMP_NUM_THREADS=1 .venv/bin/python evaluation/v8r2/measure_release.py
```

환경은 requirements.txt, 데이터/코드/가중치 해시는 TRAINING.json. PyTorch/GPU는
이 모델 학습·실행에 필요하지 않다. OpenCV 버전에 따른 재학습 비트 일치는 보장하지 않는다.
최대4800px 기호·2400px뷰어, OpenCV2threads/OMP1로 버전 순서를 교대하며 각장3회.
시간은 OCR·기호·추적/인식이며 디코딩·PDF 렌더링·네트워크·저장을 제외한다.
출력 연결 한도500은 양쪽 동일하고 기존 후보를 보존한다. DXF 원시 기하 연결도 유지한다.
새3장 이후에는 이전25장을 회귀/훈련 진단으로 나눠 비교하며 이를 새 테스트라고 부르지 않는다.

최종 고정은 FROZEN-RELEASE.json이며 원래 FROZEN.json을 덮어쓰지 않았다.
최종18회 실행은 전체 추론과 그래프 재생·점 출력의 일치를 검증한다.
차트 생성은 개발 환경의 matplotlib==3.9.4로 render_report.py를 실행한다.
[보고서/차트](REPORT.ko.md) · [최종 전체 시간](RELEASE-TIMING.json) · [모델 카드](MODEL-CARD.ko.md)
