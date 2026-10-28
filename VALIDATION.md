# MVP 검증 기록

검증일: 2026-10-28. 로컬 실제 API·프로덕션 프런트 빌드 기준입니다.

## 완료

- API 통합 테스트 8개 통과: PNG/JPEG 업로드·분석, 회전된 다중 페이지 PDF, DXF 블록/문자·연결 추출, 요소·관계 수정과 삭제, 사내 의미 등록·중복 방지·연결, 원본/JSON 반환, 잘못된 입력 차단, 실패 작업 재시도, 로그인/CSRF, 서버 재시작 후 보존.
- TypeScript 검사와 Vite production build 통과.
- 실제 브라우저에서 DXF 예제 → VALVE 선택 → V-201/밸브로 수정 → 해석 승인 → 사내 의미 등록 → 선택 요소에 연결 → 새로고침 후 보존 확인.
- 브라우저에서 연결 검토·승인, 사내 의미 사전 목록 표시 확인.
- 배포 manifest 구조 확인: 앱 전용 PVC·Deployment·Service·HTTPRoute, replica 0, 기존 boot gate 유지, 비루트·읽기 전용 rootfs, 인증 Secret 참조.
- 배포 Python 문법, 명령 도움말 및 shell 스크립트 문법 확인.

## 아직 검증하지 않은 항목

- Docker 컨테이너 빌드는 GitHub CI에서 성공했습니다. 컨테이너 자체의 실행·클러스터 rollout은 아직 확인하지 않았습니다. 로컬 호스트에는 Docker가 없습니다.
- GitHub 업로드는 권한 갱신 후 성공했습니다. 저장소: https://github.com/haesookimDev/lineage-drawings · 검증한 앱 커밋: `6847bdf9b929965aae0328101b42f3c0ea9d4e1e`.
- GitHub CI 테스트·프런트 빌드·컨테이너 artifact 생성이 모두 성공했습니다. 실행 기록: https://github.com/haesookimDev/lineage-drawings/actions/runs/40618535622 .
- k3s kubeconfig가 관리자 전용이며 현재 sudo 인증이 없어 클러스터 적용·서비스 Ready·외부 HTTPS 응답은 아직 확인하지 못했습니다.
- 대규모 실제 도면의 요소/연결 정확도는 측정하지 않았습니다. 정량 모델 성능을 주장하지 않습니다.

Kubernetes 관리자 접근이 준비되면 `deploy/README.md`의 서비스 전용 명령으로 검증한 이미지를 배포할 수 있습니다. 향후 갱신에는 `bash scripts/release.sh --deploy`가 GitHub push → 동일 SHA의 CI 성공 확인 → 이미지 다운로드 → k3s 배포·시작을 이어갑니다. 기존 앱은 변경하지 않습니다.

브라우저 검증용 synthetic 예제 데이터와 화면 캡처는 `.gitignore` 대상이며 실제 사용자 도면은 저장소에 포함되지 않습니다.
