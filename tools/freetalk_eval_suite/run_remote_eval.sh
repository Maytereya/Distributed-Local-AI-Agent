#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

URL="http://172.16.0.16/v1/freetalk/generate-once"
API_KEY="${AGENT_API_KEY:-}"
HOST_HEADER=""
TIMEOUT_SEC="60"
SESSION_PREFIX="ft_remote_eval"
CASES_PATH="${SCRIPT_DIR}/remote_cases.jsonl"
THRESHOLDS_PATH="${SCRIPT_DIR}/thresholds.json"

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
    --api-key)
      API_KEY="$2"; shift 2;;
    --host-header)
      HOST_HEADER="$2"; shift 2;;
    --timeout-sec)
      TIMEOUT_SEC="$2"; shift 2;;
    --session-prefix)
      SESSION_PREFIX="$2"; shift 2;;
    --cases)
      CASES_PATH="$2"; shift 2;;
    --thresholds)
      THRESHOLDS_PATH="$2"; shift 2;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash tools/freetalk_eval_suite/run_remote_eval.sh [options]

Options:
  --url <url>                FT debug endpoint URL
  --api-key <key>            X-API-Key value (or set AGENT_API_KEY env)
  --host-header <host>       Optional Host header
  --timeout-sec <sec>        HTTP timeout per turn
  --session-prefix <prefix>  Prefix for generated session ids
  --cases <path>             JSONL with FT eval dialogs
  --thresholds <path>        Quality gate thresholds JSON
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

echo "== FT Remote Eval =="
echo "URL: ${URL}"
echo "SESSION_PREFIX: ${SESSION_PREFIX}"
echo "RUN_ID: ${RUN_ID}"
echo "LOG_DIR: ${LOG_DIR}"
echo "CASES_PATH: ${CASES_PATH}"
echo "THRESHOLDS_PATH: ${THRESHOLDS_PATH}"
echo "TIMEOUT_SEC: ${TIMEOUT_SEC}"
echo

cd "${REPO_ROOT}" || exit 2

LOG_FILE="${LOG_DIR}/freetalk_remote_eval.log"
CMD=(
  "${PY_BIN}" tools/freetalk_eval_suite/eval_remote_cases.py
  --url "${URL}"
  --cases "${CASES_PATH}"
  --thresholds "${THRESHOLDS_PATH}"
  --session-prefix "${SESSION_PREFIX}"
  --run-id "${RUN_ID}"
  --timeout-sec "${TIMEOUT_SEC}"
)
if [[ -n "${API_KEY}" ]]; then
  CMD+=(--api-key "${API_KEY}")
fi
if [[ -n "${HOST_HEADER}" ]]; then
  CMD+=(--host-header "${HOST_HEADER}")
fi

echo "CMD: ${CMD[*]}"
if "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"; then
  echo "[ok] FT remote eval passed"
  exit 0
fi

CODE=${PIPESTATUS[0]}
echo "[fail] FT remote eval failed (exit=${CODE})"
exit "${CODE}"
