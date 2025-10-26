#!/bin/bash

# DeepSeek-Coder 6.7B 后门清洁器训练脚本
# 全面性能优化版本

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 路径配置
MODEL_PATH="/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"
DATA_DIR="/home/nfs/share-yjy/dachuang2025/m2026-lsl/FABE/Tuna/Tuna/data"
OUTPUT_DIR="/home/nfs/share-yjy/dachuang2025/m2026-lsl/FABE/Tuna/Tuna/checkpoints/backdoor_cleaner_deepseek_6.7b_fast_variant"

# 数据文件
TRAIN_FILE="${DATA_DIR}/train.jsonl"
VALID_FILE="${DATA_DIR}/valid.jsonl"

# 训练参数（性能优化）
NUM_EPOCHS=3
BATCH_SIZE=8              # 提高到8（A6000 48GB足够）
GRAD_ACCUM=4              # 降到2，有效batch=16
MAX_LENGTH=2048           # ⚡ 关键：从4096降到2048，速度提升4倍！
LR=2e-5

# LoRA 配置
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# 损失权重
GENERATION_WEIGHT=1.0
RANKING_WEIGHT=0.5

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "🚀 DeepSeek 6.7B 训练（性能优化版）"
echo "=============================================="
echo "📁 模型: ${MODEL_PATH}"
echo "📊 数据: ${TRAIN_FILE}"
echo "📈 优化:"
echo "   - 序列长度: ${MAX_LENGTH} (从4096降低)"
echo "   - Batch size: ${BATCH_SIZE}"
echo "   - Grad accum: ${GRAD_ACCUM}"
echo "   - 有效batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "   - 预计速度: ~10-12秒/步"
echo "   - 预计总时间: ~12-14小时"
echo "💾 输出: ${OUTPUT_DIR}"
echo "=============================================="

cd /home/nfs/share-yjy/dachuang2025/m2026-lsl/FABE/Tuna/Tuna/src

# 启动训练
nohup python train_backdoor_cleaner_with_ranking.py \
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
    --load_in_8bit False \
    --generation_weight ${GENERATION_WEIGHT} \
    --ranking_weight ${RANKING_WEIGHT} \
    --fp16 True \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    > ${OUTPUT_DIR}/training.log 2>&1 &

TRAINING_PID=$!
echo $TRAINING_PID > training_fast.pid

echo ""
echo "✅ 训练已启动！"
echo "🔢 进程 ID: ${TRAINING_PID}"
echo ""
echo "📊 监控命令:"
echo "  # 实时查看日志"
echo "  tail -f ${OUTPUT_DIR}/training.log"
echo ""
echo "  # 查看训练进度"
echo "  tail -f ${OUTPUT_DIR}/training.log | grep 'Step\\|it/s'"
echo ""
echo "  # 查看GPU使用"
echo "  watch -n 1 nvidia-smi"
echo ""