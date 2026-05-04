#!/bin/bash
export PATH="/venv/t2/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
cd /workspace/transformer-squared

BASE_EVAL="base_model@_global_=qwen25_1b5 mode@_global_=eval wandb_log=false"
MATH_CKPT="results/gsm8k/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-041624/policy_params.pt"
CODE_CKPT="results/mbpp/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-062508/policy_params.pt"
REASONING_CKPT="results/ai2_arc/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-084940/policy_params.pt"
CLS_CKPT="results/Cls/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-104953/policy_params.pt"
EXPERTS="experts_path_dict.math=$MATH_CKPT experts_path_dict.code=$CODE_CKPT experts_path_dict.reasoning=$REASONING_CKPT experts_path_dict.other=$CLS_CKPT"

mkdir -p run_logs

# ── MMLU ──────────────────────────────────────────────────────────
echo ">>> [1/4] MMLU - Base model"
python svd_reinforce_hydra.py $BASE_EVAL task@_global_=mmlu prompt_based_eval=false \
    2>&1 | tee run_logs/mmlu_base.log

echo ">>> [2/4] MMLU - T2 (Cls-expert / prompt-based)"
python svd_reinforce_hydra.py $BASE_EVAL task@_global_=mmlu prompt_based_eval=true \
    load_ckpt=$CLS_CKPT $EXPERTS \
    2>&1 | tee run_logs/mmlu_t2.log

# ── HellaSwag ─────────────────────────────────────────────────────
echo ">>> [3/4] HellaSwag - Base model"
python svd_reinforce_hydra.py $BASE_EVAL task@_global_=hellaswag prompt_based_eval=false \
    2>&1 | tee run_logs/hellaswag_base.log

echo ">>> [4/4] HellaSwag - T2 (Cls-expert / prompt-based)"
python svd_reinforce_hydra.py $BASE_EVAL task@_global_=hellaswag prompt_based_eval=true \
    load_ckpt=$CLS_CKPT $EXPERTS \
    2>&1 | tee run_logs/hellaswag_t2.log

echo "=== Novel datasets done ==="
