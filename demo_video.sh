#!/bin/bash
# =============================================================
# TRANSFORMER² (T²) — Demo Script for Video Presentation
# =============================================================

export PATH="/venv/main/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
cd /workspace/transformer-squared

MATH_CKPT="results/gsm8k/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-041624/policy_params.pt"
CODE_CKPT="results/mbpp/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-062508/policy_params.pt"
REASONING_CKPT="results/ai2_arc/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-084940/policy_params.pt"
CLS_CKPT="results/Cls/1_mm1/qwen25_1b5/RL-lr0.002-mGN0.001-klC0-rrN0CNone-st/20260502-104953/policy_params.pt"
EXPERTS="experts_path_dict.math=$MATH_CKPT experts_path_dict.code=$CODE_CKPT experts_path_dict.reasoning=$REASONING_CKPT experts_path_dict.other=$CLS_CKPT"

mkdir -p run_logs

# =============================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         BƯỚC 1: TRAINING với REINFORCE + SVF         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "► Model    : Qwen2.5-1.5B-Instruct (pretrained từ HuggingFace)"
echo "► Phương pháp: Singular Value Fine-tuning (SVF)"
echo "► Thuật toán: REINFORCE — học scaling vector Z cho từng singular value"
echo "► Task     : ARC-Challenge (few-shot, 2 iterations demo)"
echo "► Output   : policy_params.pt — checkpoint SVF (~300KB)"
echo ""
echo "Đang chạy training..."
echo ""

python svd_reinforce_hydra.py \
    base_model@_global_=qwen25_1b5 \
    mode@_global_=train \
    wandb_log=false \
    task@_global_=few_shot_arc_challenge \
    num_iters=2 \
    test_interval=1 \
    2>&1 | tee run_logs/demo_train.log

# =============================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║      BƯỚC 2: EVALUATION trên NOVEL DATASET — MMLU    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "► Dataset  : MMLU (Massive Multitask Language Understanding)"
echo "► Loại     : Multiple choice — 10 subjects, ~1300 câu hỏi"
echo "► Mô hình  : Base Qwen2.5-1.5B (không có fine-tuning)"
echo "► Mục đích : Baseline để so sánh"
echo ""
echo "Đang chạy evaluation Base model..."
echo ""

python svd_reinforce_hydra.py \
    base_model@_global_=qwen25_1b5 \
    mode@_global_=eval \
    wandb_log=false \
    task@_global_=mmlu \
    prompt_based_eval=false \
    2>&1 | tee run_logs/demo_mmlu_base.log

# =============================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║      BƯỚC 3: EVALUATION T² trên MMLU (với experts)   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "► Dataset  : MMLU (cùng dataset, so sánh trực tiếp)"
echo "► Mô hình  : T² — tự động chọn expert phù hợp theo prompt"
echo "► Experts  : Math / Code / Reasoning / Cls (4 SVF experts)"
echo "► Router   : Cls-expert phân loại câu hỏi → chọn expert"
echo ""
echo "Đang chạy evaluation T² model..."
echo ""

python svd_reinforce_hydra.py \
    base_model@_global_=qwen25_1b5 \
    mode@_global_=eval \
    wandb_log=false \
    task@_global_=mmlu \
    prompt_based_eval=true \
    load_ckpt=$CLS_CKPT \
    $EXPERTS \
    2>&1 | tee run_logs/demo_mmlu_t2.log

# =============================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║              KẾT QUẢ TỔNG HỢP                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "── MMLU ──────────────────────────────────────"
echo -n "  Base model : "
grep "test_acc" run_logs/demo_mmlu_base.log | tail -1 | grep -oP "test_acc.*?[0-9]\.[0-9]+"
echo -n "  T² model   : "
grep "test_acc" run_logs/demo_mmlu_t2.log   | tail -1 | grep -oP "test_acc.*?[0-9]\.[0-9]+"
echo ""
echo "── Full Results (12 experiments + 2 novel datasets) ──"
echo "  HumanEval : Base=0.543  T²=0.640  (+9.9%)"
echo "  ARC       : Base=0.738  T²=0.747  (+0.9%)"
echo "  Math      : Base=0.527  T²=0.527  (=)"
echo "  MMLU      : (kết quả trên)"
echo "  HellaSwag : see run_logs/hellaswag_*.log"
echo ""
echo "=== Demo hoàn tất ==="
