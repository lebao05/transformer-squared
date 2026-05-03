#!/bin/bash
set -e

export PATH="/home/vanh/miniconda3/envs/t2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
cd "$(dirname "$0")"

BASE_EVAL="base_model@_global_=qwen25_1b5 mode@_global_=eval wandb_log=false"
BASE_TRAIN="base_model@_global_=qwen25_1b5 wandb_log=false num_iters=100 test_interval=99"

MATH_CKPT="results/gsm8k/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-041624/policy_params.pt"
CODE_CKPT="results/mbpp/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-062508/policy_params.pt"
REASONING_CKPT="results/ai2_arc/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-084940/policy_params.pt"
CLS_CKPT="results/Cls/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-104953/policy_params.pt"

EXPERTS="experts_path_dict.math=$MATH_CKPT experts_path_dict.code=$CODE_CKPT experts_path_dict.reasoning=$REASONING_CKPT experts_path_dict.other=$CLS_CKPT"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

LOGDIR="run_logs"
RESULTS_FILE="phase2_results.json"
mkdir -p "$LOGDIR"

# ── Check if already saved ─────────────────────────────────────────────────────
already_done() {
    local task=$1 strategy=$2
    python3 -c "
import json, sys
try:
    data = json.load(open('$RESULTS_FILE'))
    for e in data.get('results', []):
        if e.get('task') == '$task' and e.get('strategy') == '$strategy':
            sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
"
}

# ── Save result to JSON ────────────────────────────────────────────────────────
save_result() {
    local logfile=$1
    local task=$2
    local strategy=$3
    local metric_name="" metric_value="" extra_key="" extra_value=""

    if [ "$task" = "humaneval" ]; then
        metric_name="pass@1"
        if grep -q "final_test_acc" "$logfile" 2>/dev/null; then
            metric_value=$(grep "final_test_acc" "$logfile" | tail -1 | grep -oP "'final_test_acc': [0-9.]+" | grep -oP "[0-9.]+")
        else
            metric_value=$(grep -A1 "humaneval (base tests)" "$logfile" 2>/dev/null | grep "pass@1" | tail -1 | awk '{print $2}')
            extra_key="pass@1_plus"
            extra_value=$(grep -A1 "humaneval+ (base + extra tests)" "$logfile" 2>/dev/null | grep "pass@1" | tail -1 | awk '{print $2}')
        fi
    else
        metric_name="test_acc"
        if grep -q "final_test_acc" "$logfile" 2>/dev/null; then
            metric_value=$(grep "final_test_acc" "$logfile" | tail -1 | grep -oP "'final_test_acc': [0-9.]+" | grep -oP "[0-9.]+")
        else
            metric_value=$(grep "^Evaluation results:" "$logfile" 2>/dev/null | tail -1 | grep -oP "'test_acc': [0-9.]+" | grep -oP "[0-9.]+")
        fi
    fi

    RF="$RESULTS_FILE" T="$task" S="$strategy" MN="$metric_name" MV="$metric_value" EK="$extra_key" EV="$extra_value" \
    python3 - <<'PYEOF'
import json, os

results_file = os.environ["RF"]
task         = os.environ["T"]
strategy     = os.environ["S"]
metric_name  = os.environ["MN"]
metric_value = os.environ["MV"]
extra_key    = os.environ.get("EK", "")
extra_value  = os.environ.get("EV", "")

with open(results_file) as f:
    data = json.load(f)

entry = {"task": task, "strategy": strategy, metric_name: metric_value}
if extra_key:
    entry[extra_key] = extra_value

data["results"].append(entry)

with open(results_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"[Saved] {task} / {strategy} => {metric_name} = {metric_value}")
PYEOF
}

# ── Few-shot T² ────────────────────────────────────────────────────────────────
run_fewshot_t2() {
    local task=$1 task_dir=$2 task_key=$3 label=$4
    local train_log="$LOGDIR/${label}_train.log"
    local eval_log="$LOGDIR/${label}_eval.log"

    touch /tmp/t2_train_marker
    python svd_reinforce_hydra.py $BASE_TRAIN task@_global_=$task 2>&1 | tee "$train_log"

    local ckpt
    ckpt=$(find "results/$task_dir" -name "policy_params.pt" -newer /tmp/t2_train_marker 2>/dev/null | sort | tail -1)
    if [ -z "$ckpt" ]; then
        ckpt=$(find "results/$task_dir" -name "policy_params.pt" 2>/dev/null | sort | tail -1)
    fi
    if [ -z "$ckpt" ]; then
        echo "ERROR: no checkpoint found for $label after training" >&2
        exit 1
    fi
    echo "Using checkpoint: $ckpt"

    python svd_reinforce_hydra.py $BASE_EVAL task@_global_=$task prompt_based_eval=false load_ckpt="$ckpt" 2>&1 | tee "$eval_log"
    save_result "$eval_log" "$task_key" "fewshot_t2"
}

# ── HUMANEVAL ─────────────────────────────────────────────────────────────────
if ! already_done humaneval base; then
    echo ">>> [1/12] HumanEval - Base"
    python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_humaneval prompt_based_eval=false \
        2>&1 | tee "$LOGDIR/he_base.log"
    save_result "$LOGDIR/he_base.log" humaneval base
else
    echo ">>> SKIP [1/12] HumanEval - Base"
fi

if ! already_done humaneval prompt; then
    echo ">>> [2/12] HumanEval - Prompt-based"
    python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_humaneval prompt_based_eval=true $EXPERTS \
        2>&1 | tee "$LOGDIR/he_prompt.log"
    save_result "$LOGDIR/he_prompt.log" humaneval prompt
else
    echo ">>> SKIP [2/12] HumanEval - Prompt-based"
fi

if ! already_done humaneval cls_expert; then
    echo ">>> [3/12] HumanEval - Cls-expert"
    python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_humaneval prompt_based_eval=false load_ckpt=$CLS_CKPT $EXPERTS \
        2>&1 | tee "$LOGDIR/he_cls.log"
    save_result "$LOGDIR/he_cls.log" humaneval cls_expert
else
    echo ">>> SKIP [3/12] HumanEval - Cls-expert"
fi

if ! already_done humaneval fewshot_t2; then
    echo ">>> [4/12] HumanEval - Few-shot T²"
    run_fewshot_t2 few_shot_humaneval humaneval_10shots humaneval he_fewshot
else
    echo ">>> SKIP [4/12] HumanEval - Few-shot T²"
fi

# ── ARC-CHALLENGE ─────────────────────────────────────────────────────────────
if ! already_done arc base; then
    echo ">>> [5/12] ARC-Challenge - Base"
    python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_arc_challenge prompt_based_eval=false \
        2>&1 | tee "$LOGDIR/arc_base.log"
    save_result "$LOGDIR/arc_base.log" arc base
else
    echo ">>> SKIP [5/12] ARC-Challenge - Base"
fi

if ! already_done arc prompt; then
    echo ">>> [6/12] ARC-Challenge - Prompt-based"
    python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_arc_challenge prompt_based_eval=true $EXPERTS \
        2>&1 | tee "$LOGDIR/arc_prompt.log"
    save_result "$LOGDIR/arc_prompt.log" arc prompt
else
    echo ">>> SKIP [6/12] ARC-Challenge - Prompt-based"
fi

if ! already_done arc cls_expert; then
    echo ">>> [7/12] ARC-Challenge - Cls-expert"
    python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_arc_challenge prompt_based_eval=false load_ckpt=$CLS_CKPT $EXPERTS \
        2>&1 | tee "$LOGDIR/arc_cls.log"
    save_result "$LOGDIR/arc_cls.log" arc cls_expert
else
    echo ">>> SKIP [7/12] ARC-Challenge - Cls-expert"
fi

if ! already_done arc fewshot_t2; then
    echo ">>> [8/12] ARC-Challenge - Few-shot T²"
    run_fewshot_t2 few_shot_arc_challenge arc_chal_10shots arc arc_fewshot
else
    echo ">>> SKIP [8/12] ARC-Challenge - Few-shot T²"
fi

# ── MATH ──────────────────────────────────────────────────────────────────────
if ! already_done math base; then
    echo ">>> [9/12] Math - Base"
    python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_math prompt_based_eval=false \
        2>&1 | tee "$LOGDIR/math_base.log"
    save_result "$LOGDIR/math_base.log" math base
else
    echo ">>> SKIP [9/12] Math - Base"
fi

if ! already_done math prompt; then
    echo ">>> [10/12] Math - Prompt-based"
    python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_math prompt_based_eval=true $EXPERTS \
        2>&1 | tee "$LOGDIR/math_prompt.log"
    save_result "$LOGDIR/math_prompt.log" math prompt
else
    echo ">>> SKIP [10/12] Math - Prompt-based"
fi

if ! already_done math cls_expert; then
    echo ">>> [11/12] Math - Cls-expert"
    python svd_reinforce_hydra.py $BASE_EVAL task@_global_=few_shot_math prompt_based_eval=false load_ckpt=$CLS_CKPT $EXPERTS \
        2>&1 | tee "$LOGDIR/math_cls.log"
    save_result "$LOGDIR/math_cls.log" math cls_expert
else
    echo ">>> SKIP [11/12] Math - Cls-expert"
fi

if ! already_done math fewshot_t2; then
    echo ">>> [12/12] Math - Few-shot T²"
    run_fewshot_t2 few_shot_math math_10shots math math_fewshot
else
    echo ">>> SKIP [12/12] Math - Few-shot T²"
fi

echo "=== Phase 2 hoàn tất. Kết quả: $RESULTS_FILE ==="
