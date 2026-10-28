#!/usr/bin/env bash
# Push the reviewed source, wait for its CI image, download, optionally deploy.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain)" ]; then
  echo 'Commit the reviewed source before releasing.' >&2
  exit 1
fi
sha="$(git rev-parse HEAD)"
git push -u origin main
run=''
for attempt in $(seq 1 20); do
  run="$(gh run list --repo haesookimDev/lineage-drawings --workflow ci.yml --commit "$sha" --json databaseId --jq '.[0].databaseId // empty')"
  [ -n "$run" ] && break
  sleep 3
done
if [ -z "$run" ]; then echo 'No CI run found. Confirm GitHub Actions is enabled.' >&2; exit 1; fi
gh run watch "$run" --repo haesookimDev/lineage-drawings --exit-status
artifact_dir="$PWD/.artifacts/$run"
mkdir -p "$artifact_dir"
gh run download "$run" --repo haesookimDev/lineage-drawings --name container-image --dir "$artifact_dir"
if [ "${1:-}" = '--deploy' ]; then
  sudo python3 deploy/k3s_deploy.py --artifact "$artifact_dir/lineage-image.tar" --sha "$sha" --start
else
  echo "Container artifact ready: $artifact_dir/lineage-image.tar"
  echo "Commit: $sha"
  echo 'Run this script with --deploy from your terminal to deploy and start LINEAGE.'
fi
