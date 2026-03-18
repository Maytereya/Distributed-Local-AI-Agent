#!/usr/bin/env bash
set -uo pipefail

# Единый запуск eval-стадий для удаленного endpoint.
# Запускать из корня репозитория:
#   bash messengers_router/eval_suite/run_remote_eval.sh --url http://172.16.0.16/api/messenger-generate-once

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

URL="http://localhost:8000/api/messenger-generate-once"
SESSION_PREFIX="s_eval_remote"
GOLDEN_VERSION=""
HOST_HEADER=""
LLM_MODE="hybrid"
CASES_PATH="${SCRIPT_DIR}/critical_cases.jsonl"

if command -v python >/dev/null 2>&1; then
  PY_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
else
  echo "[error] python interpreter not found (python/python3)" >&2
  exit 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      URL="$2"; shift 2;;
    --session-prefix)
      SESSION_PREFIX="$2"; shift 2;;
    --golden-version)
      GOLDEN_VERSION="$2"; shift 2;;
    --host-header)
      HOST_HEADER="$2"; shift 2;;
    --llm-mode)
      LLM_MODE="$2"; shift 2;;
    --cases)
      CASES_PATH="$2"; shift 2;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash messengers_router/eval_suite/run_remote_eval.sh [options]

Options:
  --url <url>                Endpoint URL (default: http://localhost:8000/api/messenger-generate-once)
  --session-prefix <prefix>  Session prefix for all eval runs
  --golden-version <vN>      Stage5 golden version, e.g. v2
  --host-header <host>       Optional Host header for critical eval script
  --llm-mode <mode>          strict|hybrid|rich (used by critical eval script)
  --cases <path>             Critical cases JSONL path (default: messengers_router/eval_suite/critical_cases.jsonl)
EOF
      exit 0;;
    *)
      echo "[error] unknown arg: $1" >&2
      exit 2;;
  esac
done

RUN_ID="$(date +%s)"
LOG_DIR="${SCRIPT_DIR}/logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

echo "== Remote Eval =="
echo "URL: ${URL}"
echo "SESSION_PREFIX: ${SESSION_PREFIX}"
echo "RUN_ID: ${RUN_ID}"
echo "LOG_DIR: ${LOG_DIR}"
echo "CASES_PATH: ${CASES_PATH}"
echo

OVERALL=0

run_stage() {
  local name="$1"; shift
  local log_file="${LOG_DIR}/${name}.log"
  echo "---- ${name} ----"
  echo "CMD: $*"
  if "$@" 2>&1 | tee "${log_file}"; then
    echo "[ok] ${name}"
  else
    local code=${PIPESTATUS[0]}
    echo "[fail] ${name} (exit=${code})"
    OVERALL=1
  fi
  echo
}

cd "${REPO_ROOT}" || exit 2

run_stage stage1 \
  "${PY_BIN}" messengers_router/scripts/eval_stage1_cases.py \
    --url "${URL}" \
    --session-prefix "${SESSION_PREFIX}_stage1_${RUN_ID}"

run_stage stage3 \
  "${PY_BIN}" messengers_router/scripts/eval_stage3_appointment_flow.py \
    --url "${URL}" \
    --session-prefix "${SESSION_PREFIX}_stage3_${RUN_ID}" \
    --run-id "${RUN_ID}"

run_stage stage4 \
  "${PY_BIN}" messengers_router/scripts/eval_stage4_reliability.py \
    --url "${URL}" \
    --session-prefix "${SESSION_PREFIX}_stage4_${RUN_ID}" \
    --run-id "${RUN_ID}"

if [[ -n "${GOLDEN_VERSION}" ]]; then
  run_stage stage5 \
    "${PY_BIN}" messengers_router/scripts/eval_stage5_corpus.py \
      --url "${URL}" \
      --session-prefix "${SESSION_PREFIX}_stage5_${RUN_ID}" \
      --run-id "${RUN_ID}" \
      --golden-version "${GOLDEN_VERSION}"
else
  run_stage stage5 \
    "${PY_BIN}" messengers_router/scripts/eval_stage5_corpus.py \
      --url "${URL}" \
      --session-prefix "${SESSION_PREFIX}_stage5_${RUN_ID}" \
      --run-id "${RUN_ID}"
fi

CRIT_CMD=(
  "${PY_BIN}" messengers_router/eval_suite/eval_critical_cases.py
  --url "${URL}"
  --cases "${CASES_PATH}"
  --session-prefix "${SESSION_PREFIX}_critical_${RUN_ID}"
  --run-id "${RUN_ID}"
  --llm-mode "${LLM_MODE}"
)
if [[ -n "${HOST_HEADER}" ]]; then
  CRIT_CMD+=(--host-header "${HOST_HEADER}")
fi
run_stage critical "${CRIT_CMD[@]}"

echo "== Summary =="
if [[ ${OVERALL} -eq 0 ]]; then
  echo "All stages passed."
else
  echo "Some stage failed. See logs in: ${LOG_DIR}"
fi

exit ${OVERALL}
