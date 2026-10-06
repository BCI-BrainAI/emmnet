#!/usr/bin/env bash
# 서버의 학습/평가 결과를 로컬(Mac)로 가져온다. 로컬에서 실행. *.pth(수백 MB)는 기본 제외.
#   bash scripts/pull_results.sh user@server              # 서버 repo 경로 기본값 /mnt/disk2/projects/emmnet
#   bash scripts/pull_results.sh user@server base2 lr3e-4 # 특정 run만
#   bash scripts/pull_results.sh user@server --with-best   # best.pth도 포함(재평가/CAM 용)
# 환경변수: EMMNET_REMOTE_DIR(서버 repo 경로), EMMNET_RESULTS(로컬 저장 폴더, 기본 <repo>/results)
# 받는 것: history.json, meta.json, config.yaml, train.log, *_metrics.json, predictions_*.csv, analysis.json, curves.png
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 1; }
HOST="$1"; shift
REMOTE_DIR="${EMMNET_REMOTE_DIR:-/mnt/disk2/projects/emmnet}"
DEST="${EMMNET_RESULTS:-results}"
WITH_BEST=0; RUNS=()
for a in "$@"; do
  case "$a" in
    --with-best) WITH_BEST=1 ;;
    *) RUNS+=("$a") ;;
  esac
done
command -v rsync >/dev/null || { echo "[ERR] rsync 필요"; exit 1; }
mkdir -p "$DEST"

FILTERS=(--include='*/')
[ "$WITH_BEST" = 1 ] && FILTERS+=(--include='best.pth')
FILTERS+=(--include='history.json' --include='meta.json' --include='config.yaml' --include='train.log'
          --include='nohup.out' --include='_queue_gpu*.out' --include='.test_used.json'
          --include='*_metrics.json' --include='predictions_*.csv' --include='analysis.json' --include='curves.png'
          --exclude='*')

if [ ${#RUNS[@]} -eq 0 ]; then
  rsync -av --prune-empty-dirs "${FILTERS[@]}" "$HOST:$REMOTE_DIR/checkpoints/" "$DEST/"
else
  for r in "${RUNS[@]}"; do
    rsync -av --prune-empty-dirs "${FILTERS[@]}" "$HOST:$REMOTE_DIR/checkpoints/$r/" "$DEST/$r/"
  done
fi
echo "[DONE] 저장 위치: $(pwd)/$DEST"
echo "분석: python scripts/analyze_results.py 는 서버에서 이미 실행됨 -> $DEST/<run>/analysis.json, curves.png 확인"
