# MVP 검증 기록

검증일: 2026-10-28. 로컬 검증과 이후 외부 서비스 검증을 구분해 기록합니다.

**외부 배포 검증 완료:** https://lineage.haesoo.dev 에서 HTTPS·로그인·PNG/JPG/PDF/DXF 업로드/분석·검토/사내 의미 저장·재로그인 후 보존·브라우저 다운로드를 확인했습니다. 상세 결과는 [외부 배포 검증 기록](DEPLOYMENT_VERIFICATION.ko.md)을 참고하세요. 외부 API 46개 항목과 실제 브라우저 흐름이 통과했습니다.

## 완료

- API 통합 테스트 8개 통과: PNG/JPEG 업로드·분석, 회전된 다중 페이지 PDF, DXF 블록/문자·연결 추출, 요소·관계 수정과 삭제, 사내 의미 등록·중복 방지·연결, 원본/JSON 반환, 잘못된 입력 차단, 실패 작업 재시도, 로그인/CSRF, 서버 재시작 후 보존.
- TypeScript 검사와 Vite production build 통과.
- 실제 브라우저에서 DXF 예제 → VALVE 선택 → V-201/밸브로 수정 → 해석 승인 → 사내 의미 등록 → 선택 요소에 연결 → 새로고침 후 보존 확인.
- 브라우저에서 연결 검토·승인, 사내 의미 사전 목록 표시 확인.
- 배포 manifest 구조 확인: 앱 전용 PVC·Deployment·Service·HTTPRoute, replica 0, 기존 boot gate 유지, 비루트·읽기 전용 rootfs, 인증 Secret 참조.
- 배포 Python 문법, 명령 도움말 및 shell 스크립트 문법 확인.

## 배포 상태와 검증 범위

- Docker 컨테이너 빌드는 GitHub CI에서 성공했고, 배포된 앱의 외부 기능 동작도 확인했습니다. 내부 클러스터 rollout 조건과 Pod 재시작은 직접 확인하지 않았습니다.
- GitHub 업로드는 권한 갱신 후 성공했습니다. 저장소: https://github.com/haesookimDev/lineage-drawings · 검증한 앱 커밋: `6847bdf9b929965aae0328101b42f3c0ea9d4e1e`.
- GitHub CI 테스트·프런트 빌드·컨테이너 artifact 생성이 모두 성공했습니다. 실행 기록: https://github.com/haesookimDev/lineage-drawings/actions/runs/40618535622 .
- k3s kubeconfig가 관리자 전용이므로 내부 리소스 상태를 직접 재검사하지 못했습니다. 사용자가 배포 명령을 실행한 뒤 외부 HTTPS 응답과 실제 서비스 흐름은 검증했습니다.
- 대규모 실제 도면의 요소/연결 정확도는 측정하지 않았습니다. 정량 모델 성능을 주장하지 않습니다.

서비스는 사용자가 `deploy/README.md`의 명령을 실행해 배포했습니다. 향후 갱신에는 `bash scripts/release.sh --deploy`가 GitHub push → 동일 SHA의 CI 성공 확인 → 이미지 다운로드 → k3s 배포·시작을 이어갑니다. 기존 앱은 변경하지 않습니다.

브라우저 검증용 synthetic 예제 데이터와 화면 캡처는 `.gitignore` 대상이며 실제 사용자 도면은 저장소에 포함되지 않습니다.


## 공개 실제 도면 개선 평가 · 2026-10-28

[평가 보고서](evaluation/REPORT.ko.md): OPEN100 공개 실제 설계 도면 12장 중 개발 4장·고정 보류 8장으로 나눴습니다. 보류 기호 527개·무방향 인접 관계 1,393개에 대해 위치 F1(IoU 0.50) 6.01%→60.27%, 연결 F1 0.23%→11.36%를 측정했습니다. 정답 영역을 준 연결 진단은 9.87%→61.05%입니다. 의미 분류·표준 식별 성공률이 아니며 자동 영향 분석에는 부족합니다.

새 기하 추출기·연결 추적기와 평가기를 포함해 회귀 테스트 14개를 통과했습니다. 이 수치는 로컬에서 동일한 OCR 조건으로 비교한 결과입니다. 이후 같은 커밋의 CI 이미지로 운영 갱신을 완료했습니다. [개선 버전 운영 배포 검증](DEPLOYED_IMPROVEMENT_VERIFICATION.ko.md)에서 실행 중인 코드 해시, 외부 API 56개 항목, 공개 도면 브라우저 검토 저장 및 기존 데이터 보존을 확인했습니다.
