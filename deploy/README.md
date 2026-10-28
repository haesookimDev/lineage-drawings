# 기존 Haesoo k3s에 LINEAGE 배포

이 배포 구성은 `/home/khs/workspace/k8s/SERVICE-OPERATIONS.md`와 기존 boot gate/HTTPRoute 구조를 참고했습니다. 기존 XGEN 및 공유 인프라를 재설치하지 않습니다. 새 서비스만 적용하며, 단일 노드 클러스터인지 확인합니다.

## 1. CI 컨테이너 준비

GitHub의 `Test and build` workflow가 성공해야 합니다. 앱 저장소에 Contents·Workflows 쓰기 권한이 있는 인증으로 push하세요. 코드만 올렸다고 이미지나 실행 서비스가 생기는 것은 아닙니다.

```bash
gh run list --repo haesookimDev/lineage-drawings --workflow ci.yml --limit 5
mkdir -p .artifacts/RUN_ID
gh run download RUN_ID --repo haesookimDev/lineage-drawings \
  --name container-image --dir .artifacts/RUN_ID
```

`RUN_ID`는 성공한 실제 실행 ID로 바꾸고, 그 실행의 전체 commit SHA를 확인합니다. artifact의 이미지는 `docker.io/library/lineage-drawings:FULL_SHA`로 태그됩니다.

## 2. 변경 계획 검토

```bash
python3 deploy/k3s_deploy.py \
  --artifact .artifacts/RUN_ID/lineage-image.tar \
  --sha FULL_SHA --dry-run
```

기본 외부 주소는 `https://lineage.haesoo.dev`입니다. 배포 전 다른 HTTPRoute와 hostname이 겹치지 않는지 검사합니다. 기존 와일드카드 DNS·Tunnel 경로가 실제로 이 서버에 연결되는지는 외부 요청으로 별도 검증해야 합니다.

## 3. 관리자 권한으로 배포

`k3s kubectl`을 사용할 수 있는 계정에서 실행합니다. 이 PC의 기본 설정은 관리자 전용이므로 필요 시 **사용자 터미널에서** sudo 인증을 완료합니다. 비밀번호나 kubeconfig 내용을 채팅·Git으로 보내지 마세요.

```bash
sudo python3 /home/khs/workspace/ontology/mvp/deploy/k3s_deploy.py \
  --artifact /home/khs/workspace/ontology/mvp/.artifacts/RUN_ID/lineage-image.tar \
  --sha FULL_SHA --start
```

명령은 먼저 namespace·gateway·boot ConfigMap·도메인 충돌을 확인하고, 로컬 containerd에 이미지를 import합니다. `lineage-auth` Secret이 없을 때만 랜덤 초기 로그인 비밀번호와 PBKDF2 해시를 생성합니다. 비밀번호는 표준 출력으로 내보내지 않고 `.secrets/login.txt`에 0600으로 저장합니다. sudo로 실행했다면 해당 파일의 소유권을 실행한 일반 사용자로 설정합니다.

다음에 전용 PVC 10Gi, Deployment, Service, HTTPRoute를 적용합니다. 기존 LINEAGE를 갱신하면 그 서비스만 중지합니다. `--start`가 없으면 replica 0을 유지합니다. `--start`는 기존 플랫폼과 같은 방식으로 LINEAGE의 현재 boot ID만 승인한 후 Ready 상태를 기다립니다.

## 4. 배포 후 확인

```bash
sudo k3s kubectl -n apps get deploy,svc,pvc -l app=lineage
sudo k3s kubectl -n apps get httproute lineage -o yaml
sudo k3s kubectl -n apps logs deploy/lineage --tail=60
curl -i https://lineage.haesoo.dev/healthz
```

외부 주소에서 로그인 → 예제/실제 도면 업로드 → 분석 완료 → 요소 승인 → 사내 정의 등록 → 새로고침 후 보존 → JSON 내보내기를 확인합니다. 이미지 artifact import와 클러스터 rollout을 실제로 수행하기 전에는 배포 완료로 간주하지 않습니다.

## 이미지·데이터·복구

이 구성은 레지스트리 없이 단일 노드 containerd의 imported image를 사용하고 `imagePullPolicy: Never`입니다. 다른 노드나 이미지 GC 이후에는 같은 artifact를 다시 import하거나 동일 소스로 재빌드해야 합니다. 다중 노드 운영은 전용 레지스트리와 pull 정책으로 전환합니다.

SQLite·원본·미리보기는 `lineage-data` PVC에 저장됩니다. Deployment 중지·이미지 교체는 PVC를 삭제하지 않습니다. 백업·이전 이미지 복구 시 데이터 스키마 호환성을 먼저 확인합니다.

Jenkins/Argo 연결은 이 첫 배포 명령이 자동으로 구성하지 않습니다. 향후 배포를 Jenkins 전용 작업으로 옮기려면 같은 이미지 SHA와 생성 manifest를 전용 배포 저장소에서 관리하고, PVC/Secret provisioning은 관리자 작업으로 분리합니다. 기존 일반 서비스 배포 스크립트는 PVC·Secret을 자동 처리하지 않습니다.
