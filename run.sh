#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${METHOD:-irc-gs}"

run_method() {
  local method="$1"
  case "${method}" in
    irc-gs)
      MODEL_NAME=irc-gs bash "${REPO_DIR}/train.sh"
      ;;
    cl-splats)
      bash "${REPO_DIR}/run_cl_splats.sh"
      ;;
    scaffold-gs)
      MODEL_NAME=scaffold-gs bash "${REPO_DIR}/train.sh"
      ;;
    3dgs)
      MODEL_NAME=3dgs bash "${REPO_DIR}/train.sh"
      ;;
    4dgs)
      MODEL_NAME=4dgs bash "${REPO_DIR}/train.sh"
      ;;
    *)
      echo "[ERROR] Unknown METHOD=${method}" >&2
      echo "[INFO] Choose irc-gs, cl-splats, scaffold-gs, 3dgs, 4dgs, or all." >&2
      return 2
      ;;
  esac
}

if [[ "${METHOD}" == "all" ]]; then
  for method in irc-gs cl-splats scaffold-gs 3dgs 4dgs; do
    run_method "${method}"
  done
else
  run_method "${METHOD}"
fi
