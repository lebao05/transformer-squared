#!/bin/bash
set -e
export PATH="/venv/t2/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")"

BASE_EVAL="base_model@_global_=qwen25_1b5 mode@_global_=eval wandb_log=false"
BASE_TRAIN="base_model@_global_=qwen25_1b5 wandb_log=false num_iters=100 test_interval=99"
MATH_CKPT="results/gsm8k/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-041624/policy_params.pt"
CODE_CKPT="results/mbpp/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-062508/policy_params.pt"
REASONING_CKPT="results/ai2_arc/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-084940/policy_params.pt"
CLS_CKPT="results/Cls/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-104953/policy_params.pt"
EXPERTS="experts_path_dict.math=$MATH_CKPT experts_path_dict.code=$CODE_CKPT experts_path_dict.reasoning=$REASONING_CKPT experts_path_dict.other=$CLS_CKPT"
mkdir -p run_logs

echo ">>> [11/12] Math - Cls-expert"
python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_math prompt_based_eval=false load_ckpt=$CLS_CKPT $EXPERTS \
    2>&1 | tee run_logs/math_cls.log

echo ">>> [12/12] Math - Few-shot T2 (train)"
touch /tmp/t2_train_marker_gpu1
python svd_reinforce_hydra.py $BASE_TRAIN task@_global_=few_shot_math \
    2>&1 | tee run_logs/math_fewshot_train.log

ckpt=$(find "results/math_10shots" -name "policy_params.pt" -newer /tmp/t2_train_marker_gpu1 2>/dev/null | sort | tail -1)
echo "Using checkpoint: $ckpt"

echo ">>> [12/12] Math - Few-shot T2 (eval)"
python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_math prompt_based_eval=false load_ckpt="$ckpt" \
    2>&1 | tee run_logs/math_fewshot_eval.log

echo "=== GPU1 done ==="
