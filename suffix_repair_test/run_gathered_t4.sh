#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model_dir=${1:?Pass the local Qwen2.5-1.5B-Instruct model directory}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENABLE_V1_MULTIPROCESSING=0

.venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("The T4 test requires a visible CUDA device.")
name = torch.cuda.get_device_name(0)
if "T4" not in name.split() or torch.cuda.get_device_capability(0) != (7, 5):
    raise SystemExit(f"Select an NVIDIA T4 with CUDA_VISIBLE_DEVICES. Found {name}.")
print(f"Running gathered replay on {name} with PyTorch {torch.__version__}")
PY

.venv/bin/python -m unittest -v \
    suffix_repair_test.test_gathered_replay.GatheredReplayTest.test_triton_attention_against_dense_prefixes
.venv/bin/python -m suffix_repair_test.verify_selective_edit "$model_dir" \
    --tp 1 --q-format int4 --repeats 7
