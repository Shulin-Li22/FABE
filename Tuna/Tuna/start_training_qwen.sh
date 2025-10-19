#!/bin/bash

# Qwen2.5-Coder 7B 简化训练脚本
# 只使用生成损失（不包含排序损失）

# 激活 conda 环境
source /home/nfs/u2023-ckh/miniconda3/bin/activate fabe

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 路径配置
MODEL_PATH="/home/nfs/share-yjy/dachuang2025/models/Qwen/Qwen2.5-Coder-7B-Instruct"
DATA_DIR="/home/nfs/u2023-ckh/FABE/Tuna/data"
OUTPUT_DIR="/home/nfs/u2023-ckh/checkpoints/backdoor_cleaner_qwen_7b"

# 数据文件
TRAIN_FILE="${DATA_DIR}/train.jsonl"
VALID_FILE="${DATA_DIR}/valid.jsonl"

# 训练参数
NUM_EPOCHS=3
BATCH_SIZE=8              # 提高到8以充分利用A100 80GB
GRAD_ACCUM=4              # 保持4，有效batch=32（训练更稳定但更快）
MAX_LENGTH=4096
LR=2e-5

# LoRA 配置
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "🚀 开始训练 Qwen2.5-Coder 7B（简化版）"
echo "=============================================="
echo "📁 模型: ${MODEL_PATH}"
echo "📊 训练集: ${TRAIN_FILE}"
echo "📊 验证集: ${VALID_FILE}"
echo "💾 输出: ${OUTPUT_DIR}"
echo "=============================================="

cd /home/nfs/u2023-ckh/FABE/Tuna/src

# 启动训练（使用简化版脚本）
nohup python train_backdoor_cleaner.py \
    --model_name_or_path ${MODEL_PATH} \
    --train_data_path ${TRAIN_FILE} \
    --eval_data_path ${VALID_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --model_max_length ${MAX_LENGTH} \
    --learning_rate ${LR} \
    --warmup_steps 200 \
    --weight_decay 0.01 \
    --use_lora True \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --fp16 True \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    > ${OUTPUT_DIR}/training.log 2>&1 &

TRAINING_PID=$!
echo $TRAINING_PID > /home/nfs/u2023-ckh/FABE/Tuna/training_qwen.pid

echo ""
echo "✅ 训练已启动！"
echo "🔢 进程 ID: ${TRAINING_PID}"
echo ""
echo "📊 监控命令:"
echo "  tail -f ${OUTPUT_DIR}/training.log"
echo ""

