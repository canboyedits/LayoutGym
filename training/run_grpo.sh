#!/usr/bin/env bash
# DesignGym GRPO — HuggingFace Job launcher
#
# Usage:
#   SMOKE TEST (v3_anchored schedule + best-of-4 eval, ~60–80 min on T4):
#       bash training/run_grpo.sh --smoke
#
#   The --smoke flag in training/grpo_train.py expands to:
#       reward_version    = v3_anchored   (anchored to instruction levers)
#       num_train_states  = 240           (with bad_state_ratio = 0.30)
#       max_steps         = 100
#       num_generations   = 8
#       batch / grad_acc  = 2 / 4
#       learning_rate     = 2e-5
#       temperature/top_p = 1.10 / 0.95
#       beta              = 0.04          (canonical KL anchor)
#       eval_seeds        = 4             (12 rollouts per policy)
#       eval_best_of      = 4             (adds sft@4 + grpo@4 to the table)
#       output_repo       = …-smoke-v3
#
#   SMOKE TEST with explicit args (equivalent override):
#       bash training/run_grpo.sh \
#           --reward_version v3_anchored \
#           --output_dir /workspace/grpo_designgym2_qwen05_smoke_v3 \
#           --hub_model_id yashvyasop/designgym2-grpo-qwen05-lora-smoke-v3 \
#           --num_train_states 240 --max_steps 100 \
#           --num_generations 8 \
#           --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
#           --learning_rate 2e-5 --temperature 1.10 --top_p 0.95 --beta 0.04 \
#           --max_completion_length 128 \
#           --eval_seeds 4 --eval_best_of 4 \
#           --bon_temperature 0.9 --bon_top_p 0.95 \
#           --bad_state_ratio 0.30 --reward_debug_samples 40 \
#           --smoke_report_json /workspace/grpo_designgym2_qwen05_smoke_v3/smoke_report.json
#
#   FULL TRAIN (defaults: 400 states / 200 steps / 3 eval seeds, no BoN):
#       bash training/run_grpo.sh
#
#   FULL TRAIN with best-of-4 highlight eval:
#       bash training/run_grpo.sh --eval_best_of 4
#
#   CUSTOM:  any argparse flag in training/grpo_train.py works here, e.g.
#       bash training/run_grpo.sh --max_steps 300 --temperature 1.10
set -euo pipefail

EXTRA_ARGS="${@}"

echo "============================================"
echo " DesignGym GRPO Training"
echo "============================================"

# ── GPU check ──────────────────────────────────────────────────────────
python -c "
import torch
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    try:
        print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
    except Exception:
        pass
" || true

# ── Install deps (pinned to unsloth-compatible ranges) ─────────────────
echo "[SETUP] Installing dependencies…"
pip install -U pip -q

pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q \
    "trl>=0.22.0,<=0.24.0" \
    "peft>=0.15.0,<1.0" \
    "transformers>=4.51.3,<=5.5.0" \
    "datasets>=3.4.1,<4.4.0" \
    accelerate bitsandbytes mergekit llm-blender weave \
    huggingface_hub matplotlib pandas tqdm \
    fastapi uvicorn pydantic 

pip uninstall -y torchao 2>/dev/null || true

# ── Sanity check ───────────────────────────────────────────────────────
echo "[SETUP] Import sanity check…"
python -c "
import torch, transformers, peft, trl
print(f'torch={torch.__version__}  transformers={transformers.__version__}')
print(f'peft={peft.__version__}  trl={trl.__version__}')

# Patch missing constant that llm_blender expects from older transformers
import transformers.utils.hub as _h
if not hasattr(_h, 'TRANSFORMERS_CACHE'):
    from huggingface_hub.constants import HF_HUB_CACHE
    _h.TRANSFORMERS_CACHE = HF_HUB_CACHE

from trl import GRPOTrainer, GRPOConfig
print('GRPOTrainer OK')
"

# ── Install DesignGym ──────────────────────────────────────────────────
echo "[SETUP] Installing DesignGym…"
pip install -e . -q

# mergekit 0.1.4 needs pydantic ~2.10; the editable install above may
# upgrade pydantic past that, breaking mergekit's torch.Tensor schemas.
pip install -q "pydantic>=2.10.0,<2.11.0"

# ── Run training ───────────────────────────────────────────────────────
echo "[TRAIN] Starting GRPO training…"
python training/grpo_train.py ${EXTRA_ARGS}

echo "[DONE] Job complete."
