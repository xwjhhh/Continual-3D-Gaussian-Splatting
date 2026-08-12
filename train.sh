#!/usr/bin/env bash

# Standalone WAT batch runner for IRC-GS and the bundled baselines.
# This file contains the complete training, resume, evaluation, and history workflow.
#
# Example:
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

if [[ "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]] && [[ "${OMP_NUM_THREADS}" -ge 1 ]]; then
  export OMP_NUM_THREADS
else
  export OMP_NUM_THREADS=8
fi

DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/cl-splats/WAT}"
NUM_TIMES="${NUM_TIMES:-10}"
BASE_ITERS="${BASE_ITERS:-30000}"
INC_ITERS="${INC_ITERS:-30000}"
T0_IMPORTANCE_ADAPT_ITERS="${T0_IMPORTANCE_ADAPT_ITERS:-5000}"
TEMPORAL_STAGE1_UNTIL="${TEMPORAL_STAGE1_UNTIL:-10000}"
TEMPORAL_STAGE2_UNTIL="${TEMPORAL_STAGE2_UNTIL:-25000}"
TEMPORAL_DENSIFICATION_ENABLED="${TEMPORAL_DENSIFICATION_ENABLED:-0}"
TEMPORAL_CLONE_VOTE_THRESHOLD="${TEMPORAL_CLONE_VOTE_THRESHOLD:-3}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
MODEL_NAME="${MODEL_NAME:-irc-gs}"
MODEL_SLUG="${MODEL_NAME//-/_}"
FOURDGS_IMAGE_SCALE="${FOURDGS_IMAGE_SCALE:-1.0}"
FOURDGS_SKIP_IF_OUTPUT_EXISTS="${FOURDGS_SKIP_IF_OUTPUT_EXISTS:-1}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_DIR}/outputs/${MODEL_SLUG}}"
IMPORTANT_ROOT="${IMPORTANT_ROOT:-${REPO_DIR}/important}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_FINISHED_SCENES="${SKIP_FINISHED_SCENES:-0}"
RESUME_PARTIAL_SCENES="${RESUME_PARTIAL_SCENES:-1}"
SPLIT_SEED="${SPLIT_SEED:-42}"
CHANGE_THRESHOLD="${CHANGE_THRESHOLD:-0.5}"
CHANGE_DILATE_KERNEL="${CHANGE_DILATE_KERNEL:-13}"
LIFTER_VOTE_THRESHOLD="${LIFTER_VOTE_THRESHOLD:-3}"
PRUNING_MIN_CLUSTER_SIZE="${PRUNING_MIN_CLUSTER_SIZE:-50}"
HISTORY_RECOVERY_EVAL="${HISTORY_RECOVERY_EVAL:-1}"
SAVE_RECOVERY_IMAGES="${SAVE_RECOVERY_IMAGES:-1}"
KEEP_RECOVERED_PLY="${KEEP_RECOVERED_PLY:-1}"
ONLY_SCENES="${ONLY_SCENES:-}"

USES_IRCGS_TEMPORAL_STAGES=0
case "${MODEL_NAME}" in
  irc-gs|irc_gs)
    USES_IRCGS_TEMPORAL_STAGES=1
    ;;
esac

SCENES=(
  breville
  kitchen
  living_room
  community
  spa
  street
  car_resized
  grill_resized
  mac
  ninja
)

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
    if [[ "${timestep}" =~ ^[0-9]+$ ]] && [[ -f "${ply_path}" ]] && (( timestep > latest_t )); then
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

INTEGER_NAMES=(NUM_TIMES BASE_ITERS INC_ITERS T0_IMPORTANCE_ADAPT_ITERS SAVE_INTERVAL CHANGE_DILATE_KERNEL LIFTER_VOTE_THRESHOLD PRUNING_MIN_CLUSTER_SIZE)
if [[ "${USES_IRCGS_TEMPORAL_STAGES}" == "1" ]]; then
  INTEGER_NAMES+=(TEMPORAL_STAGE1_UNTIL TEMPORAL_STAGE2_UNTIL)
fi
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
if [[ "${TEMPORAL_DENSIFICATION_ENABLED}" != "0" && "${TEMPORAL_DENSIFICATION_ENABLED}" != "1" ]]; then
  echo "[ERROR] TEMPORAL_DENSIFICATION_ENABLED must be 0 or 1, got '${TEMPORAL_DENSIFICATION_ENABLED}'."
  exit 2
fi
if [[ "${MODEL_NAME}" == "4dgs" ]]; then
  if [[ ! "${FOURDGS_IMAGE_SCALE}" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "${FOURDGS_IMAGE_SCALE}" == "0"* && "${FOURDGS_IMAGE_SCALE}" != "0."* ]]; then
    echo "[ERROR] FOURDGS_IMAGE_SCALE must be a positive number, got '${FOURDGS_IMAGE_SCALE}'."
    exit 2
  fi
  if [[ "${FOURDGS_SKIP_IF_OUTPUT_EXISTS}" != "0" && "${FOURDGS_SKIP_IF_OUTPUT_EXISTS}" != "1" ]]; then
    echo "[ERROR] FOURDGS_SKIP_IF_OUTPUT_EXISTS must be 0 or 1, got '${FOURDGS_SKIP_IF_OUTPUT_EXISTS}'."
    exit 2
  fi
fi
if [[ ! "${TEMPORAL_CLONE_VOTE_THRESHOLD}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ERROR] TEMPORAL_CLONE_VOTE_THRESHOLD must be a non-negative number, got '${TEMPORAL_CLONE_VOTE_THRESHOLD}'."
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] Dataset root does not exist: ${DATA_ROOT}"
  echo "[ERROR] Set DATA_ROOT to the directory containing the WAT scene folders."
  exit 2
fi

echo "[INFO] Repo       : ${REPO_DIR}"
echo "[INFO] Data root  : ${DATA_ROOT}"
echo "[INFO] Num times  : ${NUM_TIMES}"
echo "[INFO] Base iters : ${BASE_ITERS}"
echo "[INFO] Inc iters  : ${INC_ITERS}"
echo "[INFO] T0 schedule: train=1..${BASE_ITERS}, consolidate=$((BASE_ITERS + 1))..$((BASE_ITERS + T0_IMPORTANCE_ADAPT_ITERS)) (${T0_IMPORTANCE_ADAPT_ITERS} extra iters)"
echo "[INFO] T1 densify : ${TEMPORAL_DENSIFICATION_ENABLED} (0 keeps only the one-time DINO clones)"
if [[ "${USES_IRCGS_TEMPORAL_STAGES}" == "1" ]]; then
  echo "[INFO] IRC-GS stages: inherit=1..${TEMPORAL_STAGE1_UNTIL}, revise=$((TEMPORAL_STAGE1_UNTIL + 1))..${TEMPORAL_STAGE2_UNTIL}, consolidate=${TEMPORAL_STAGE2_UNTIL}, adapt=$((TEMPORAL_STAGE2_UNTIL + 1))..${INC_ITERS}"
  echo "[INFO] Resid vote : ${TEMPORAL_CLONE_VOTE_THRESHOLD}"
else
  echo "[INFO] IRC-GS stages: disabled for model=${MODEL_NAME}"
fi
echo "[INFO] Output base : ${OUTPUT_BASE}"
echo "[INFO] Model       : ${MODEL_NAME}"
echo "[INFO] Important  : ${IMPORTANT_ROOT}"
echo "[INFO] Skip done   : ${SKIP_FINISHED_SCENES}"
echo "[INFO] Resume      : ${RESUME_PARTIAL_SCENES}"
echo "[INFO] Split seed  : ${SPLIT_SEED}"
echo "[INFO] DINO thresh : ${CHANGE_THRESHOLD}"
echo "[INFO] DINO dilate : ${CHANGE_DILATE_KERNEL}"
echo "[INFO] Vote thresh : ${LIFTER_VOTE_THRESHOLD}"
echo "[INFO] HDBSCAN min : ${PRUNING_MIN_CLUSTER_SIZE}"
echo "[INFO] History eval: ${HISTORY_RECOVERY_EVAL}"
echo "[INFO] History PNG : ${SAVE_RECOVERY_IMAGES}"
echo "[INFO] History PLY : ${KEEP_RECOVERED_PLY}"
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
  TRAIN_LOG_DIR="${RUN_ROOT}/logs"
  METRICS_DIR="${RUN_ROOT}/metrics"
  TRAIN_LOG="${TRAIN_LOG_DIR}/train.log"
  CSV_PATH="${RUN_ROOT}/per_timestep_summary.csv"
  JSON_PATH="${RUN_ROOT}/per_timestep_summary.json"
  SUMMARY_ROWS_JSONL="${RUN_ROOT}/per_timestep_summary.jsonl"

  if [[ "${SKIP_FINISHED_SCENES}" == "1" ]] && [[ -f "${RUN_ROOT}/per_timestep_summary.json" ]]; then
    echo "[SKIP] Already finished: ${RUN_ROOT}/per_timestep_summary.json"
    continue
  fi

  mkdir -p "${RUN_ROOT}" "${TRAIN_LOG_DIR}" "${METRICS_DIR}"
  : > "${SUMMARY_ROWS_JSONL}"

  echo
  echo "################################################################"
  echo "[SCENE] ${SCENE_NAME}"
  echo "[DATA ] dataset.path=${DATASET_PATH}"
  echo "[OUT  ] ${RUN_ROOT}"
  echo "################################################################"

  RESUME_ARGS=()
  if [[ "${RESUME_PARTIAL_SCENES}" == "1" ]]; then
    LATEST_CHECKPOINT="$(find_latest_checkpoint "${RUN_ROOT}")"
    if [[ -n "${LATEST_CHECKPOINT}" ]]; then
      RESUME_ARGS=("++resume_from=${LATEST_CHECKPOINT}")
      echo "[RESUME] ${LATEST_CHECKPOINT}"
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
    "++model.representation=gaussian"
    "++model.optimizer_type=sparse_adam"
    "++train.start_time=0"
    "++train.num_times=${NUM_TIMES}"
    "++train.iterations=${BASE_ITERS}"
    "++train.incremental_iterations=${INC_ITERS}"
    "++train.t0_importance_prune_enabled=true"
    "++train.t0_importance_adapt_iters=${T0_IMPORTANCE_ADAPT_ITERS}"
    "++change_detection.threshold=${CHANGE_THRESHOLD}"
    "++change_detection.dilate_mask=true"
    "++change_detection.dilate_kernel_size=${CHANGE_DILATE_KERNEL}"
    "++change_detection.upsample=true"
    "++lifter.vote_threshold=${LIFTER_VOTE_THRESHOLD}"
    "++pruning.min_cluster_size=${PRUNING_MIN_CLUSTER_SIZE}"
  )
  if [[ "${MODEL_NAME}" == "4dgs" ]]; then
    TRAIN_CMD+=("++model.fourdgs.image_scale=${FOURDGS_IMAGE_SCALE}")
    TRAIN_CMD+=("++model.fourdgs.skip_if_output_exists=$([[ "${FOURDGS_SKIP_IF_OUTPUT_EXISTS}" == "1" ]] && echo true || echo false)")
  fi
  if [[ "${USES_IRCGS_TEMPORAL_STAGES}" == "1" ]]; then
    TRAIN_CMD+=(
      "++train.temporal_stage1_until=${TEMPORAL_STAGE1_UNTIL}"
      "++train.temporal_stage2_until=${TEMPORAL_STAGE2_UNTIL}"
      "++train.temporal_densification_enabled=$([[ "${TEMPORAL_DENSIFICATION_ENABLED}" == "1" ]] && echo true || echo false)"
      "++train.temporal_clone_vote_threshold=${TEMPORAL_CLONE_VOTE_THRESHOLD}"
    )
  fi
  TRAIN_CMD+=(
    "++save_interval=${SAVE_INTERVAL}"
    "++runtime.experiment_tag=${SCENE_NAME}"
    "++output_dir=${RUN_ROOT}"
    "++wandb_mode=disabled"
    "++history.log_history=true"
    "++history.eval_recovered=$([[ "${HISTORY_RECOVERY_EVAL}" == "1" ]] && echo true || echo false)"
    "++history.save_recovery_images=$([[ "${SAVE_RECOVERY_IMAGES}" == "1" ]] && echo true || echo false)"
    "++history.keep_recovered_ply=$([[ "${KEEP_RECOVERED_PLY}" == "1" ]] && echo true || echo false)"
    "++history.keep_full_checkpoints=true"
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

  EVALUATED_COUNT=0
  for ((t=0; t<NUM_TIMES; t++)); do
    CKPT_PATH="${RUN_ROOT}/checkpoint_t${t}.pt"
    PLY_PATH="${RUN_ROOT}/checkpoint_t${t}.ply"
    T_EVAL_JSON="${METRICS_DIR}/eval_t${t}.json"

    if [[ ! -f "${CKPT_PATH}" ]] && [[ ! -f "${PLY_PATH}" ]]; then
      echo "[INFO] No artifacts for timestep ${t}; the dataset may contain fewer than ${NUM_TIMES} timesteps."
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

    "${PYTHON_BIN}" -m ircgs.eval \
      --dataset-path "${DATASET_PATH}" \
      --checkpoint "${CKPT_PATH}" \
      --timestep "${t}" \
      --split-seed "${SPLIT_SEED}" \
      --output-json "${T_EVAL_JSON}" \
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
    echo "[ERROR] No timestep checkpoints were evaluated under ${RUN_ROOT}."
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
