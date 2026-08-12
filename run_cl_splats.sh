#!/usr/bin/env bash

# Standalone batch runner for the public CL-Splats implementation.
# This file contains the complete training, resume, evaluation, and history workflow.
#
# Example:
#   CL_SPLATS_MAIN_DIR=/path/to/cl-splats-main \
#   DATA_ROOT=/path/to/WAT ONLY_SCENES=breville,spa bash run_cl_splats.sh

set -u
set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_DIR}"
fi
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TORCH_CUDA_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
GPU_COMPUTE_CAPABILITY="$("${PYTHON_BIN}" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "")' 2>/dev/null || true)"
CUDA_REQUIRED_ARCH=""
if [[ -n "${GPU_COMPUTE_CAPABILITY}" ]]; then
  CUDA_REQUIRED_ARCH="compute_${GPU_COMPUTE_CAPABILITY//./}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${GPU_COMPUTE_CAPABILITY}}"
fi

# gsplat is JIT-compiled on first use. A directory containing nvcc alone is
# not a complete CUDA_HOME: torch's host C++ compilation also needs the CUDA
# runtime headers under CUDA_HOME/include.
cuda_toolkit_include_dir() {
  local root="$1"
  local candidate=""

  for candidate in \
    "${root}/include" \
    "${root}/targets/x86_64-linux/include" \
    "${root}"/targets/*/include; do
    if [[ -f "${candidate}/cuda_runtime_api.h" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

cuda_toolkit_library_dir() {
  local root="$1"
  local candidate=""

  for candidate in \
    "${root}/lib64" \
    "${root}/lib" \
    "${root}/targets/x86_64-linux/lib" \
    "${root}"/targets/*/lib; do
    if [[ -d "${candidate}" ]] \
      && compgen -G "${candidate}/libcudart.so*" >/dev/null; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

cuda_toolkit_is_complete() {
  local root="$1"
  [[ -x "${root}/bin/nvcc" ]] && cuda_toolkit_include_dir "${root}" >/dev/null
}

cuda_toolkit_version() {
  local root="$1"
  "${root}/bin/nvcc" --version 2>/dev/null \
    | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' \
    | head -n 1
}

cuda_toolkit_is_compatible() {
  local root="$1"
  local version=""

  cuda_toolkit_is_complete "${root}" || return 1
  version="$(cuda_toolkit_version "${root}")"
  if [[ -n "${TORCH_CUDA_VERSION}" ]] \
    && [[ "${version%%.*}" != "${TORCH_CUDA_VERSION%%.*}" ]]; then
    return 1
  fi
  if [[ -n "${CUDA_REQUIRED_ARCH}" ]] \
    && ! "${root}/bin/nvcc" --list-gpu-arch 2>/dev/null | grep -qx "${CUDA_REQUIRED_ARCH}"; then
    return 1
  fi
  return 0
}

REQUESTED_CUDA_HOME="${CUDA_HOME:-}"
CUDA_CANDIDATES=()
if [[ -n "${REQUESTED_CUDA_HOME}" ]]; then
  CUDA_CANDIDATES+=("${REQUESTED_CUDA_HOME}")
fi
if [[ -n "${TORCH_CUDA_VERSION}" ]]; then
  CUDA_CANDIDATES+=("/usr/local/cuda-${TORCH_CUDA_VERSION}")
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  CUDA_CANDIDATES+=("${CONDA_PREFIX}")
fi
CUDA_CANDIDATES+=("/usr/local/cuda" "/usr/local/cuda-12.6")
for CUDA_VERSIONED_DIR in /usr/local/cuda-[0-9]*; do
  if [[ -d "${CUDA_VERSIONED_DIR}" ]]; then
    CUDA_CANDIDATES+=("${CUDA_VERSIONED_DIR}")
  fi
done
NVCC_ON_PATH="$(command -v nvcc 2>/dev/null || true)"
if [[ -n "${NVCC_ON_PATH}" ]]; then
  NVCC_ROOT="$(cd "$(dirname "${NVCC_ON_PATH}")/.." 2>/dev/null && pwd || true)"
  if [[ -n "${NVCC_ROOT}" ]]; then
    CUDA_CANDIDATES+=("${NVCC_ROOT}")
  fi
fi

SELECTED_CUDA_HOME=""
FOUND_COMPLETE_CUDA=0
for CUDA_CANDIDATE in "${CUDA_CANDIDATES[@]}"; do
  if cuda_toolkit_is_complete "${CUDA_CANDIDATE}"; then
    FOUND_COMPLETE_CUDA=1
  fi
  if cuda_toolkit_is_compatible "${CUDA_CANDIDATE}"; then
    SELECTED_CUDA_HOME="$(cd "${CUDA_CANDIDATE}" && pwd)"
    break
  fi
done

if [[ -z "${SELECTED_CUDA_HOME}" ]]; then
  echo "[ERROR] A CUDA toolkit compatible with PyTorch and the current GPU was not found."
  echo "[ERROR] gsplat needs <CUDA_HOME>/bin/nvcc and cuda_runtime_api.h under include/ or targets/*/include/."
  echo "[ERROR] PyTorch CUDA=${TORCH_CUDA_VERSION:-unknown}, required GPU arch=${CUDA_REQUIRED_ARCH:-unknown}."
  if [[ "${FOUND_COMPLETE_CUDA}" == "1" ]]; then
    echo "[ERROR] A complete toolkit exists, but its nvcc version does not support this configuration."
  fi
  if [[ -n "${NVCC_ON_PATH}" ]]; then
    echo "[ERROR] nvcc on PATH: ${NVCC_ON_PATH}"
  fi
  echo "[ERROR] Locate the header with:"
  echo "        find /usr/local \"${CONDA_PREFIX:-/nonexistent}\" -name cuda_runtime_api.h 2>/dev/null"
  if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/nvcc" ]] && [[ -n "${TORCH_CUDA_VERSION}" ]]; then
    echo "[ERROR] The Conda nvcc is present; install its matching runtime headers with:"
    echo "        conda install -y -c nvidia cuda-cudart-dev=${TORCH_CUDA_VERSION}"
  fi
  echo "[ERROR] Then rerun with CUDA_HOME set to the directory containing bin/ and include/."
  exit 2
fi

if [[ -n "${REQUESTED_CUDA_HOME}" ]] && ! cuda_toolkit_is_compatible "${REQUESTED_CUDA_HOME}"; then
  echo "[WARN] Ignoring incomplete or incompatible CUDA_HOME=${REQUESTED_CUDA_HOME}"
  echo "[WARN] Using complete CUDA toolkit at ${SELECTED_CUDA_HOME}"
fi
export CUDA_HOME="${SELECTED_CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
CUDA_INCLUDE_DIR="$(cuda_toolkit_include_dir "${CUDA_HOME}")"
CUDA_LIBRARY_DIR="$(cuda_toolkit_library_dir "${CUDA_HOME}" || true)"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_INCLUDE_DIR}${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${CUDA_INCLUDE_DIR}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
if [[ -n "${CUDA_LIBRARY_DIR}" ]]; then
  export LIBRARY_PATH="${CUDA_LIBRARY_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CUDA_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Use a CUDA-versioned cache so a failed build from another AutoDL image is not
# reused. Limit Ninja parallelism to keep the first gsplat build predictable.
CUDA_TOOLKIT_VERSION="$(cuda_toolkit_version "${CUDA_HOME}")"
CUDA_CACHE_TAG="cu${CUDA_TOOLKIT_VERSION//./}"
if [[ "${CUDA_CACHE_TAG}" == "cu" ]]; then
  CUDA_CACHE_TAG="cuda_unknown"
fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_DIR}/.torch_extensions/${CUDA_CACHE_TAG}}"
export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

if [[ "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]] && [[ "${OMP_NUM_THREADS}" -ge 1 ]]; then
  export OMP_NUM_THREADS
else
  export OMP_NUM_THREADS=8
fi

DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/cl-splats/WAT}"
NUM_TIMES="${NUM_TIMES:-5}"
ITERS_PER_TIMESTEP="${ITERS_PER_TIMESTEP:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
MODEL_NAME="cl-splats"
MODEL_SLUG="${MODEL_NAME//-/_}"
# Keep official CL-Splats artifacts inside this checkout by default. Both
# variables remain overridable for experiments that intentionally use another
# storage location.
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_DIR}/outputs/${MODEL_SLUG}_official_30k}"
IMPORTANT_ROOT="${IMPORTANT_ROOT:-${REPO_DIR}/important/${MODEL_SLUG}_official_30k}"
CL_SPLATS_PYTHON_BIN="${CL_SPLATS_PYTHON_BIN:-${PYTHON_BIN}}"
CL_SPLATS_MAIN_DIR="${CL_SPLATS_MAIN_DIR:-}"
CL_SPLATS_OFFLINE="${CL_SPLATS_OFFLINE:-1}"
CL_SPLATS_PREFER_HARDLINKS="${CL_SPLATS_PREFER_HARDLINKS:-1}"
CL_SPLATS_SH_DEGREE="${CL_SPLATS_SH_DEGREE:-0}"
SKIP_FINISHED_SCENES="${SKIP_FINISHED_SCENES:-1}"
RESUME_PARTIAL_SCENES="${RESUME_PARTIAL_SCENES:-1}"
EVAL_ONLY="${EVAL_ONLY:-0}"
EVAL_TRAIN_SPLIT="${EVAL_TRAIN_SPLIT:-0}"
SPLIT_SEED="${SPLIT_SEED:-42}"
OFFICIAL_CHANGE_THRESHOLD="${OFFICIAL_CHANGE_THRESHOLD:-0.8}"
OFFICIAL_DILATE_MASK="${OFFICIAL_DILATE_MASK:-0}"
OFFICIAL_DILATE_KERNEL_SIZE="${OFFICIAL_DILATE_KERNEL_SIZE:-31}"
OFFICIAL_UPSAMPLE="${OFFICIAL_UPSAMPLE:-1}"
OFFICIAL_LAMBDA_BOUND="${OFFICIAL_LAMBDA_BOUND:-0.0}"
OFFICIAL_PRUNE_EVERY="${OFFICIAL_PRUNE_EVERY:-50}"
OFFICIAL_PRUNE_DIST_THRESH="${OFFICIAL_PRUNE_DIST_THRESH:-0.02}"
OFFICIAL_PRUNE_CONSECUTIVE="${OFFICIAL_PRUNE_CONSECUTIVE:-3}"
ONLY_SCENES="${ONLY_SCENES:-}"

if [[ -z "${CL_SPLATS_MAIN_DIR}" ]]; then
  for CANDIDATE in "${REPO_DIR}/cl-splats-main" "$(dirname "${REPO_DIR}")/cl-splats-main"; do
    if [[ -f "${CANDIDATE}/clsplats/train.py" ]]; then
      CL_SPLATS_MAIN_DIR="${CANDIDATE}"
      break
    fi
  done
fi

SCENES=(
  breville
  car_resized
  community
  grill_resized
  kitchen
  living_room
  mac
  ninja
  spa
  street
)

get_scene_num_times() {
  case "$1" in
    community)
      echo 10
      ;;
    mac)
      echo 6
      ;;
    *)
      echo "${NUM_TIMES}"
      ;;
  esac
}

summary_has_expected_timesteps() {
  local summary_path="$1"
  local expected_count="$2"

  "${PYTHON_BIN}" - "${summary_path}" "${expected_count}" >/dev/null 2>&1 <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_count = int(sys.argv[2])

try:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("rows", []) or []
    observed = {int(row["timestep"]) for row in rows}
except Exception:
    raise SystemExit(1)

raise SystemExit(0 if set(range(expected_count)).issubset(observed) else 1)
PY
}

official_run_is_complete() {
  local run_root="$1"
  local expected_count="$2"

  "${PYTHON_BIN}" - "${run_root}" "${expected_count}" >/dev/null 2>&1 <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
expected_count = int(sys.argv[2])
official_run = run_root / "_official_cl_splats" / "run"
manifest_path = official_run / "bridge_run.json"

try:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if int(manifest.get("num_times", -1)) != expected_count:
        raise SystemExit(1)
except Exception:
    raise SystemExit(1)

missing = [
    timestep
    for timestep in range(expected_count)
    if not (official_run / "outputs" / f"gaussians_time_{timestep:04d}.ply").is_file()
]
raise SystemExit(0 if not missing else 1)
PY
}

if [[ -n "${ONLY_SCENES:-}" ]]; then
  SCENES_FILTER="${ONLY_SCENES//,/ }"
  read -r -a SCENES <<< "${SCENES_FILTER}"
fi

find_latest_checkpoint() {
  local output_dir="$1"
  local latest_path=""
  local latest_t=-1
  local path name timestep ply_path

  shopt -s nullglob
  for path in "${output_dir}"/checkpoint_t*.pt; do
    name="$(basename "${path}")"
    timestep="${name#checkpoint_t}"
    timestep="${timestep%.pt}"
    ply_path="${path%.pt}.ply"
    if [[ "${timestep}" =~ ^[0-9]+$ ]] \
      && [[ -f "${ply_path}" ]] \
      && "${PYTHON_BIN}" - "${path}" >/dev/null 2>&1 <<'PY'
import sys

import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
raise SystemExit(
    0 if str(payload.get("trainer_type", "")).lower() == "official_cl_splats" else 1
)
PY
    then
      if (( timestep <= latest_t )); then
        continue
      fi
      latest_t="${timestep}"
      latest_path="${path}"
    fi
  done
  shopt -u nullglob

  if [[ -n "${latest_path}" ]]; then
    echo "${latest_path}"
  fi
}

collect_important_results() {
  local scene_name="$1"
  local dataset_path="$2"
  local run_root="$3"
  local csv_path="$4"
  local json_path="$5"
  local jsonl_path="$6"
  local important_root="$7"
  local split_seed="$8"
  local model_slug="$9"

  "${PYTHON_BIN}" - "${scene_name}" "${dataset_path}" "${run_root}" "${csv_path}" "${json_path}" "${jsonl_path}" "${important_root}" "${split_seed}" "${model_slug}" <<'PY'
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

scene_name = sys.argv[1]
dataset_path = Path(sys.argv[2]).resolve()
run_root = Path(sys.argv[3]).resolve()
csv_path = Path(sys.argv[4]).resolve()
json_path = Path(sys.argv[5]).resolve()
jsonl_path = Path(sys.argv[6]).resolve()
important_root = Path(sys.argv[7]).resolve()
split_seed = int(sys.argv[8])
model_slug = sys.argv[9]
important_dir = important_root / f"{model_slug}_{scene_name}"

important_dir.mkdir(parents=True, exist_ok=True)

for src_path in (csv_path, json_path, jsonl_path):
    if src_path.is_file():
        shutil.copy2(src_path, important_dir / src_path.name)

history_metrics_dir = important_dir / "history_metrics"
history_metrics_dir.mkdir(parents=True, exist_ok=True)
for src_path in sorted((run_root / "history").glob("*_metrics.*")):
    if src_path.is_file():
        shutil.copy2(src_path, history_metrics_dir / src_path.name)

rows: list[dict[str, Any]] = []
if json_path.is_file():
    with open(json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    rows = summary_data.get("rows", []) or []
else:
    summary_data = {}

copied = 0
skipped = 0
timesteps = 0
test_images_root = important_dir / "test_images"
test_images_root.mkdir(parents=True, exist_ok=True)

try:
    from ircgs.dataset import CLSplatsDataset  # type: ignore

    dataset = CLSplatsDataset(
        path=str(dataset_path),
        resolution_scale=1.0,
        white_background=False,
        eval_mode=True,
        split_seed=split_seed,
        prefer_undist=True,
    )
    timesteps = int(dataset.get_num_timesteps())
    for t in range(timesteps):
        t_dir = test_images_root / f"t{t}"
        if t_dir.exists():
            shutil.rmtree(t_dir)
        t_dir.mkdir(parents=True, exist_ok=True)
        scene_info = dataset.get_scene_info(t)
        cam_infos = getattr(scene_info, "test_cameras", []) or []
        for cam_info in cam_infos:
            src = getattr(cam_info, "image_path", None)
            if not src:
                skipped += 1
                continue
            src_path = Path(src)
            if not src_path.is_file():
                skipped += 1
                continue
            dst_path = t_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            copied += 1
    copy_status = "ok"
except Exception as exc:
    copy_status = f"failed:{exc}"

scene_summary = {
    "scene": scene_name,
    "dataset_path": str(dataset_path),
    "run_root": str(run_root),
    "important_dir": str(important_dir),
    "num_timesteps": summary_data.get("num_timesteps"),
    "avg_psnr": summary_data.get("avg_psnr"),
    "avg_ssim": summary_data.get("avg_ssim"),
    "rows": rows,
    "copied_test_images": copied,
    "skipped_test_images": skipped,
    "test_image_timesteps": timesteps,
    "test_copy_status": copy_status,
    "test_images_dir": str(test_images_root.resolve()),
}

with open(important_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(scene_summary, f, indent=2, ensure_ascii=False)

with open(important_dir / "summary.txt", "w", encoding="utf-8") as f:
    f.write(f"scene: {scene_name}\n")
    f.write(f"dataset_path: {dataset_path}\n")
    f.write(f"run_root: {run_root}\n")
    f.write(f"num_timesteps: {scene_summary['num_timesteps']}\n")
    f.write(f"avg_psnr: {scene_summary['avg_psnr']}\n")
    f.write(f"avg_ssim: {scene_summary['avg_ssim']}\n")
    f.write(f"copied_test_images: {copied}\n")
    f.write(f"skipped_test_images: {skipped}\n")
    f.write(f"test_image_timesteps: {timesteps}\n")
    f.write(f"test_copy_status: {copy_status}\n")
    f.write(f"test_images_dir: {scene_summary['test_images_dir']}\n")

summary_csv_path = important_root / f"{model_slug}_summary.csv"
fieldnames = [
    "scene",
    "num_timesteps",
    "avg_psnr",
    "avg_ssim",
    "copied_test_images",
    "skipped_test_images",
    "test_image_timesteps",
    "test_copy_status",
    "run_root",
    "important_dir",
]

existing_rows: list[dict[str, Any]] = []
if summary_csv_path.is_file():
    with open(summary_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("scene") == scene_name:
                continue
            existing_rows.append(row)

existing_rows.append(
    {
        "scene": scene_name,
        "num_timesteps": scene_summary["num_timesteps"],
        "avg_psnr": scene_summary["avg_psnr"],
        "avg_ssim": scene_summary["avg_ssim"],
        "copied_test_images": copied,
        "skipped_test_images": skipped,
        "test_image_timesteps": timesteps,
        "test_copy_status": copy_status,
        "run_root": str(run_root),
        "important_dir": str(important_dir),
    }
)

with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(existing_rows)

print(f"[IMPORTANT] scene={scene_name} copied_test_images={copied} status={copy_status} dir={important_dir}")
PY
}

INTEGER_NAMES=(NUM_TIMES ITERS_PER_TIMESTEP SAVE_INTERVAL OFFICIAL_DILATE_KERNEL_SIZE OFFICIAL_PRUNE_EVERY OFFICIAL_PRUNE_CONSECUTIVE)
for INTEGER_NAME in "${INTEGER_NAMES[@]}"; do
  INTEGER_VALUE="${!INTEGER_NAME}"
  if [[ ! "${INTEGER_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ${INTEGER_NAME} must be a positive integer, got '${INTEGER_VALUE}'."
    exit 2
  fi
done
if [[ ! "${SPLIT_SEED}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] SPLIT_SEED must be a non-negative integer, got '${SPLIT_SEED}'."
  exit 2
fi
if [[ ! "${CL_SPLATS_SH_DEGREE}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] CL_SPLATS_SH_DEGREE must be a non-negative integer, got '${CL_SPLATS_SH_DEGREE}'."
  exit 2
fi
for BOOLEAN_NAME in CL_SPLATS_OFFLINE CL_SPLATS_PREFER_HARDLINKS OFFICIAL_DILATE_MASK OFFICIAL_UPSAMPLE SKIP_FINISHED_SCENES RESUME_PARTIAL_SCENES EVAL_ONLY EVAL_TRAIN_SPLIT; do
  BOOLEAN_VALUE="${!BOOLEAN_NAME}"
  if [[ "${BOOLEAN_VALUE}" != "0" && "${BOOLEAN_VALUE}" != "1" ]]; then
    echo "[ERROR] ${BOOLEAN_NAME} must be 0 or 1, got '${BOOLEAN_VALUE}'."
    exit 2
  fi
done
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] Dataset root does not exist: ${DATA_ROOT}"
  echo "[ERROR] Set DATA_ROOT to the directory containing the WAT scene folders."
  exit 2
fi
if [[ -z "${CL_SPLATS_MAIN_DIR}" ]] || [[ ! -f "${CL_SPLATS_MAIN_DIR}/clsplats/train.py" ]] || [[ ! -f "${CL_SPLATS_MAIN_DIR}/configs/cl-splats.yaml" ]]; then
  echo "[ERROR] Public CL-Splats checkout not found: ${CL_SPLATS_MAIN_DIR:-<unset>}"
  echo "[ERROR] Put cl-splats-main beside this repository or set CL_SPLATS_MAIN_DIR."
  exit 2
fi
if ! command -v "${CL_SPLATS_PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] CL-Splats Python executable not found: ${CL_SPLATS_PYTHON_BIN}"
  exit 2
fi

echo "[INFO] Repo       : ${REPO_DIR}"
echo "[INFO] Official   : ${CL_SPLATS_MAIN_DIR}"
echo "[INFO] Official py: ${CL_SPLATS_PYTHON_BIN}"
echo "[INFO] Data root  : ${DATA_ROOT}"
echo "[INFO] Num times  : default=${NUM_TIMES}, community=10, mac=6"
echo "[INFO] Iters/time : ${ITERS_PER_TIMESTEP}"
echo "[INFO] SH degree  : ${CL_SPLATS_SH_DEGREE}"
echo "[INFO] Offline    : ${CL_SPLATS_OFFLINE}"
echo "[INFO] Output base : ${OUTPUT_BASE}"
echo "[INFO] Model       : ${MODEL_NAME}"
echo "[INFO] Important  : ${IMPORTANT_ROOT}"
echo "[INFO] CUDA home   : ${CUDA_HOME:-<auto>}"
echo "[INFO] CUDA include: ${CUDA_INCLUDE_DIR:-<unknown>}"
echo "[INFO] CUDA library: ${CUDA_LIBRARY_DIR:-<runtime-provided>}"
echo "[INFO] Ext cache   : ${TORCH_EXTENSIONS_DIR}"
echo "[INFO] Build jobs  : ${MAX_JOBS}"
echo "[INFO] Skip done   : ${SKIP_FINISHED_SCENES}"
echo "[INFO] Resume      : ${RESUME_PARTIAL_SCENES}"
echo "[INFO] Eval only   : ${EVAL_ONLY}"
echo "[INFO] Eval train  : ${EVAL_TRAIN_SPLIT}"
echo "[INFO] Split seed  : ${SPLIT_SEED}"
echo "[INFO] DINO thresh : ${OFFICIAL_CHANGE_THRESHOLD}"
echo "[INFO] DINO dilate : ${OFFICIAL_DILATE_MASK} (kernel=${OFFICIAL_DILATE_KERNEL_SIZE})"
echo "[INFO] Bound lambda: ${OFFICIAL_LAMBDA_BOUND}"
echo "[INFO] Prune       : every=${OFFICIAL_PRUNE_EVERY}, dist=${OFFICIAL_PRUNE_DIST_THRESH}, consecutive=${OFFICIAL_PRUNE_CONSECUTIVE}"
echo "[INFO] Scenes     : ${SCENES[*]}"

FOUND_SCENES=0
for SCENE in "${SCENES[@]}"; do
  DATASET_PATH="${DATA_ROOT}/${SCENE}"
  SCENE_NAME="${SCENE}"
  if [[ ! -d "${DATASET_PATH}" ]] && [[ "${SCENE}" == "car_resized" ]] && [[ -d "${DATA_ROOT}/car" ]]; then
    DATASET_PATH="${DATA_ROOT}/car"
    SCENE_NAME="car"
  fi
  if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "[WARN] Skip missing dataset folder: ${DATASET_PATH}"
    continue
  fi
  FOUND_SCENES=$((FOUND_SCENES + 1))

  RUN_ROOT="${OUTPUT_BASE}/${SCENE_NAME}"
  SCENE_NUM_TIMES="$(get_scene_num_times "${SCENE}")"
  TRAIN_LOG_DIR="${RUN_ROOT}/logs"
  METRICS_DIR="${RUN_ROOT}/metrics"
  TRAIN_LOG="${TRAIN_LOG_DIR}/train.log"
  CSV_PATH="${RUN_ROOT}/per_timestep_summary.csv"
  JSON_PATH="${RUN_ROOT}/per_timestep_summary.json"
  SUMMARY_ROWS_JSONL="${RUN_ROOT}/per_timestep_summary.jsonl"

  if [[ "${EVAL_ONLY}" != "1" ]] \
    && [[ "${SKIP_FINISHED_SCENES}" == "1" ]] \
    && [[ -f "${JSON_PATH}" ]]; then
    if summary_has_expected_timesteps "${JSON_PATH}" "${SCENE_NUM_TIMES}"; then
      echo "[SKIP] Already finished (${SCENE_NUM_TIMES} timesteps): ${JSON_PATH}"
      continue
    fi
    echo "[RESUME] Existing summary is incomplete; expected t0-t$((SCENE_NUM_TIMES - 1)): ${JSON_PATH}"
  fi

  mkdir -p "${RUN_ROOT}" "${TRAIN_LOG_DIR}" "${METRICS_DIR}"
  : > "${SUMMARY_ROWS_JSONL}"

  echo
  echo "################################################################"
  echo "[SCENE] ${SCENE_NAME}"
  echo "[DATA ] dataset.path=${DATASET_PATH}"
  echo "[OUT  ] ${RUN_ROOT}"
  echo "[TIMES] ${SCENE_NUM_TIMES} (t0-t$((SCENE_NUM_TIMES - 1)))"
  echo "################################################################"

  if [[ "${EVAL_ONLY}" != "1" ]]; then
    RESUME_ARGS=()
    if [[ "${RESUME_PARTIAL_SCENES}" == "1" ]]; then
      LATEST_CHECKPOINT="$(find_latest_checkpoint "${RUN_ROOT}")"
      if [[ -n "${LATEST_CHECKPOINT}" ]]; then
        if official_run_is_complete "${RUN_ROOT}" "${SCENE_NUM_TIMES}"; then
          RESUME_ARGS=("++resume_from=${LATEST_CHECKPOINT}")
          echo "[RESUME] ${LATEST_CHECKPOINT}"
        else
          echo "[RESTART] Existing official run does not contain all ${SCENE_NUM_TIMES} timesteps."
          echo "[RESTART] Rebuilding ${SCENE_NAME} from t0 so all checkpoints use one consistent run."
        fi
      fi
    fi

    TRAIN_CMD=(
      "${PYTHON_BIN}" -m ircgs.train
      "++dataset.path=${DATASET_PATH}"
      "++dataset.type=colmap"
      "++dataset.resolution=1.0"
      "++dataset.white_background=false"
      "++dataset.eval=true"
      "++dataset.split_seed=${SPLIT_SEED}"
      "++dataset.prefer_undist=true"
      "++model.name=${MODEL_NAME}"
      "++model.official_cl_splats.root=${CL_SPLATS_MAIN_DIR}"
      "++model.official_cl_splats.python=${CL_SPLATS_PYTHON_BIN}"
      "++model.official_cl_splats.iters_per_timestep=${ITERS_PER_TIMESTEP}"
      "++model.official_cl_splats.sh_degree=${CL_SPLATS_SH_DEGREE}"
      "++model.official_cl_splats.offline=$([[ "${CL_SPLATS_OFFLINE}" == "1" ]] && echo true || echo false)"
      "++model.official_cl_splats.prefer_hardlinks=$([[ "${CL_SPLATS_PREFER_HARDLINKS}" == "1" ]] && echo true || echo false)"
      "++model.official_cl_splats.change_threshold=${OFFICIAL_CHANGE_THRESHOLD}"
      "++model.official_cl_splats.dilate_mask=$([[ "${OFFICIAL_DILATE_MASK}" == "1" ]] && echo true || echo false)"
      "++model.official_cl_splats.dilate_kernel_size=${OFFICIAL_DILATE_KERNEL_SIZE}"
      "++model.official_cl_splats.upsample=$([[ "${OFFICIAL_UPSAMPLE}" == "1" ]] && echo true || echo false)"
      "++model.official_cl_splats.lambda_bound=${OFFICIAL_LAMBDA_BOUND}"
      "++model.official_cl_splats.prune_every=${OFFICIAL_PRUNE_EVERY}"
      "++model.official_cl_splats.prune_dist_thresh=${OFFICIAL_PRUNE_DIST_THRESH}"
      "++model.official_cl_splats.prune_consecutive=${OFFICIAL_PRUNE_CONSECUTIVE}"
      "++train.start_time=0"
      "++train.num_times=${SCENE_NUM_TIMES}"
    )
    TRAIN_CMD+=(
      "++save_interval=${SAVE_INTERVAL}"
      "++runtime.experiment_tag=${SCENE_NAME}"
      "++output_dir=${RUN_ROOT}"
      "++wandb_mode=disabled"
      "++history.log_history=true"
      "${RESUME_ARGS[@]}"
    )

    echo "[TRAIN] Running all timesteps for scene=${SCENE_NAME}. Log: ${TRAIN_LOG}"
    "${TRAIN_CMD[@]}" 2>&1 | tee "${TRAIN_LOG}"
    TRAIN_STATUS=${PIPESTATUS[0]}
    if [[ ${TRAIN_STATUS} -ne 0 ]]; then
      echo "[ERROR] Training failed for scene=${SCENE_NAME}. See ${TRAIN_LOG}"
      exit ${TRAIN_STATUS}
    fi

    HISTORY_STORAGE_CSV="${RUN_ROOT}/history/storage_metrics.csv"
    HISTORY_RECOVERY_CSV="${RUN_ROOT}/history/recovery_metrics.csv"
    if [[ -f "${HISTORY_STORAGE_CSV}" ]]; then
      echo
      echo "[HISTORY STORAGE] ${HISTORY_STORAGE_CSV}"
      cat "${HISTORY_STORAGE_CSV}"
    fi
    if [[ -f "${HISTORY_RECOVERY_CSV}" ]]; then
      echo
      echo "[HISTORY RECOVERY] ${HISTORY_RECOVERY_CSV}"
      cat "${HISTORY_RECOVERY_CSV}"
    fi
  else
    echo "[EVAL ONLY] Using existing checkpoints under ${RUN_ROOT}"
  fi

  EVALUATED_COUNT=0
  for ((t=0; t<SCENE_NUM_TIMES; t++)); do
    CKPT_PATH="${RUN_ROOT}/checkpoint_t${t}.pt"
    PLY_PATH="${RUN_ROOT}/checkpoint_t${t}.ply"
    T_EVAL_JSON="${METRICS_DIR}/eval_t${t}.json"

    if [[ ! -f "${CKPT_PATH}" ]] && [[ ! -f "${PLY_PATH}" ]]; then
      echo "[INFO] No artifacts for timestep ${t}; expected ${SCENE_NUM_TIMES} timesteps for ${SCENE_NAME}."
      continue
    fi
    if [[ ! -f "${CKPT_PATH}" ]]; then
      echo "[ERROR] Missing checkpoint: ${CKPT_PATH}"
      exit 1
    fi
    if [[ ! -f "${PLY_PATH}" ]]; then
      echo "[ERROR] Missing ply: ${PLY_PATH}"
      exit 1
    fi

    EVAL_EXTRA_ARGS=()
    if [[ "${EVAL_TRAIN_SPLIT}" == "1" ]]; then
      EVAL_EXTRA_ARGS+=(--eval-train-split)
    fi

    "${PYTHON_BIN}" -m ircgs.eval \
      --dataset-path "${DATASET_PATH}" \
      --checkpoint "${CKPT_PATH}" \
      --timestep "${t}" \
      --split-seed "${SPLIT_SEED}" \
      --save-images \
      --output-json "${T_EVAL_JSON}" \
      "${EVAL_EXTRA_ARGS[@]}" \
      2>&1 | tee "${METRICS_DIR}/eval_t${t}.log"
    EVAL_STATUS=${PIPESTATUS[0]}
    if [[ ${EVAL_STATUS} -ne 0 ]]; then
      echo "[ERROR] Eval failed at scene=${SCENE_NAME} timestep=${t}. See ${METRICS_DIR}/eval_t${t}.log"
      exit ${EVAL_STATUS}
    fi

    "${PYTHON_BIN}" - "${SCENE_NAME}" "${DATASET_PATH}" "${CKPT_PATH}" "${PLY_PATH}" "${T_EVAL_JSON}" "${t}" "${SUMMARY_ROWS_JSONL}" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

scene = sys.argv[1]
dataset_path = sys.argv[2]
ckpt_path = Path(sys.argv[3])
ply_path = Path(sys.argv[4])
eval_json_path = Path(sys.argv[5])
timestep = int(sys.argv[6])
summary_jsonl_path = Path(sys.argv[7])

checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
metrics = {}
with open(eval_json_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

num_gaussians = None
scene_counts = checkpoint.get("scene_final_gaussian_counts", {}) or {}
if isinstance(scene_counts, dict):
    scene_key = f"t{timestep}"
    if scene_key in scene_counts:
        num_gaussians = int(scene_counts[scene_key])
if num_gaussians is None:
    history = checkpoint.get("history", {}) or {}
    hist_counts = history.get("num_gaussians", []) if isinstance(history, dict) else []
    if hist_counts:
        try:
            num_gaussians = int(hist_counts[-1])
        except Exception:
            num_gaussians = None

row = {
    "scene": scene,
    "dataset_path": dataset_path,
    "timestep": timestep,
    "checkpoint_path": str(ckpt_path.resolve()),
    "ply_path": str(ply_path.resolve()),
    "checkpoint_size_bytes": int(ckpt_path.stat().st_size),
    "ply_size_bytes": int(ply_path.stat().st_size),
    "num_gaussians": num_gaussians,
    "psnr": float(metrics["metrics"]["psnr"]),
    "ssim": float(metrics["metrics"]["ssim"]),
    "l1": float(metrics["metrics"]["l1"]),
    "num_evaluated_views": int(metrics.get("num_evaluated_views", 0)),
    "saved_test_images": int(metrics.get("saved_test_images", 0)),
}
with open(summary_jsonl_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(row, ensure_ascii=False))
PY
    EVALUATED_COUNT=$((EVALUATED_COUNT + 1))
  done

  if [[ "${EVALUATED_COUNT}" -eq 0 ]]; then
    if [[ "${EVAL_ONLY}" == "1" ]]; then
      echo "[WARN] No timestep checkpoints were evaluated under ${RUN_ROOT}; continuing with the remaining scenes."
      continue
    fi
    echo "[ERROR] Training completed but no timestep checkpoints were found under ${RUN_ROOT}."
    exit 1
  fi

  "${PYTHON_BIN}" - "${SUMMARY_ROWS_JSONL}" "${CSV_PATH}" "${JSON_PATH}" <<'PY'
import csv
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
json_path = Path(sys.argv[3])

rows = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

rows.sort(key=lambda x: int(x["timestep"]))
avg_psnr = sum(float(r["psnr"]) for r in rows) / len(rows) if rows else None
avg_ssim = sum(float(r["ssim"]) for r in rows) / len(rows) if rows else None

summary = {
    "num_timesteps": len(rows),
    "avg_psnr": avg_psnr,
    "avg_ssim": avg_ssim,
    "rows": rows,
}

fieldnames = [
    "scene",
    "dataset_path",
    "timestep",
    "checkpoint_path",
    "ply_path",
    "checkpoint_size_bytes",
    "ply_size_bytes",
    "num_gaussians",
    "psnr",
    "ssim",
    "l1",
    "num_evaluated_views",
    "saved_test_images",
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Saved csv : {csv_path}")
print(f"Saved json: {json_path}")
print(f"avg_psnr  : {avg_psnr}")
print(f"avg_ssim  : {avg_ssim}")
PY

  collect_important_results \
    "${SCENE_NAME}" \
    "${DATASET_PATH}" \
    "${RUN_ROOT}" \
    "${CSV_PATH}" \
    "${JSON_PATH}" \
    "${SUMMARY_ROWS_JSONL}" \
    "${IMPORTANT_ROOT}" \
    "${SPLIT_SEED}" \
    "${MODEL_SLUG}"

  echo
  echo "[DONE] Results saved under: ${RUN_ROOT}"
done

if [[ "${FOUND_SCENES}" -eq 0 ]]; then
  echo "[ERROR] None of the requested scenes were found under ${DATA_ROOT}."
  exit 2
fi
