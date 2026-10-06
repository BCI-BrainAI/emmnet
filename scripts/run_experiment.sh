#!/usr/bin/env bash
# 학습 + val/train 평가를 한 번에. 로그/설정/git sha는 checkpoints/<name>/ 에 남는다. test는 자동 실행하지 않는다.
#
#   scripts/run_experiment.sh <name> [--gpu 0] [--bg] -- [train_mri_encoder.py 인자: --set k=v ... --processed-dir P]
#   예) EMMNET_DATA=/path/to/processed scripts/run_experiment.sh lr3e-4 --gpu 0 --bg -- --set train.lr=3e-4
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

[[ $# -ge 1 ]] || { sed -n '2,6p' "$0"; exit 1; }
NAME="$1"; shift
GPU=0; BG=0; RESUME=0
while [[ $# -gt 0 && "$1" != "--" ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --bg) BG=1; shift ;;
    --resume) RESUME=1; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")

RUN="${CKPT_ROOT:-checkpoints}/${NAME}"
if [[ -e "$RUN" && -n "$(ls -A "$RUN" 2>/dev/null)" && $RESUME -eq 0 ]]; then
  echo "이미 존재/비어있지 않음: $RUN (덮어쓰기 방지). 이어서 하려면 --resume, 아니면 다른 이름을 쓰세요."; exit 1
fi
mkdir -p "$RUN"
RESUME_ARG=(); TEE=(tee)
[[ $RESUME -eq 1 ]] && { RESUME_ARG=(--resume); TEE=(tee -a); }

job() {
  export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1
  python scripts/train_mri_encoder.py --checkpoint-dir "$RUN" ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | "${TEE[@]}" "$RUN/train.log"
  python scripts/evaluate_mri_encoder.py --checkpoint "$RUN/best.pth" --splits train val 2>&1 | tee "$RUN/eval_trainval.log"
  echo "[RUN-DONE] $RUN  (test는 미실행: python scripts/evaluate_mri_encoder.py --checkpoint $RUN/best.pth --splits test)"
}

if [[ $BG -eq 1 ]]; then
  # 터미널/SSH가 끊겨도 계속 실행되도록 새 세션에서 분리(setsid + nohup)
  ARGS=("$NAME" --gpu "$GPU"); [[ $RESUME -eq 1 ]] && ARGS+=(--resume)
  LAUNCH=(nohup); command -v setsid >/dev/null 2>&1 && LAUNCH=(setsid nohup)
  "${LAUNCH[@]}" bash "$SELF" "${ARGS[@]}" -- ${EXTRA[@]+"${EXTRA[@]}"} > "$RUN/nohup.out" 2>&1 < /dev/null &
  echo "background PID $! , log: $RUN/nohup.out, $RUN/train.log (끊기면: 같은 명령에 --resume)"
else
  job
fi
