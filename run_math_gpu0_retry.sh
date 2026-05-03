#!/bin/bash
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

echo ">>> [9/12] Math - Base"
python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_math prompt_based_eval=false \
    2>&1 | tee run_logs/math_base.log

echo ">>> [10/12] Math - Prompt-based"
python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_math prompt_based_eval=true $EXPERTS \
    2>&1 | tee run_logs/math_prompt.log

echo ">>> [11/12] Math - Cls-expert"
python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_math prompt_based_eval=false load_ckpt=$CLS_CKPT $EXPERTS \
    2>&1 | tee run_logs/math_cls.log

echo "=== GPU0 done [9][10][11] ==="
