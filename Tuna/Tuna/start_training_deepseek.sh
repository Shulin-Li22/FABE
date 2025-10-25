#!/bin/bash

# DeepSeek-Coder 6.7B 后门清洁器训练脚本
# 使用生成+排序损失的增强训练方法

# 激活 conda 环境
#source /home/nfs/u2023-ckh/miniconda3/bin/activate fabe

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 路径配置
MODEL_PATH="/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"
DATA_DIR="/home/nfs/share-yjy/dachuang2025/m2026-lsl/FABE/Tuna/Tuna/data"
OUTPUT_DIR="/home/nfs/share-yjy/dachuang2025/m2026-lsl/FABE/Tuna/Tuna/checkpoints/backdoor_cleaner_deepseek_6.7b"

# 数据文件
TRAIN_FILE="${DATA_DIR}/train.jsonl"
VALID_FILE="${DATA_DIR}/valid.jsonl"
TEST_FILE="${DATA_DIR}/test.jsonl"

# 训练参数
NUM_EPOCHS=3
BATCH_SIZE=4              # 提高到4，利用A100 80GB的充裕显存
GRAD_ACCUM=4              # 降到4，保持有效batch = 4*4 = 16
MAX_LENGTH=4096           # 改为4096以覆盖更多样本
LR=2e-5

# LoRA 配置
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# 损失权重（FABE组合损失函数）
GENERATION_WEIGHT=1.0    # MLE Loss: 生成干净代码
RANKING_WEIGHT=0.3       # Listwise Ranking Loss: 保持语义一致性

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "🚀 开始训练 DeepSeek-Coder 6.7B 后门清洁器"
echo "=============================================="
echo "📁 模型: ${MODEL_PATH}"
echo "📊 数据集:"
echo "   - 训练集: ${TRAIN_FILE} (21,854 样本)"
echo "   - 验证集: ${VALID_FILE} (2,732 样本)"
echo "   - 测试集: ${TEST_FILE} (2,732 样本)"
echo "📈 训练策略:"
echo "   - 生成损失权重: ${GENERATION_WEIGHT}"
echo "   - 排序损失权重: ${RANKING_WEIGHT}"
echo "   - LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}"
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
echo $TRAINING_PID > /home/nfs/share-yjy/dachuang2025/m2026-lsl/FABE/Tuna/Tuna/training_deepseek.pid

echo ""
echo "✅ 训练已启动！"
echo "🔢 进程 ID: ${TRAINING_PID}"
echo ""
echo "📊 监控命令:"
echo "  # 查看实时日志"
echo "  tail -f ${OUTPUT_DIR}/training.log"
echo ""
echo "  # 查看 TensorBoard"
echo "  tensorboard --logdir ${OUTPUT_DIR}/logs"
echo ""
echo "  # 检查进程状态"
echo "  ps aux | grep ${TRAINING_PID}"
echo ""
echo "  # 查看 GPU 使用"
echo "  watch -n 1 nvidia-smi"
echo ""

