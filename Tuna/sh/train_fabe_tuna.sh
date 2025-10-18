#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false  # 禁用tokenizer并行度避免警告

# --- 模型与数据路径 ---
MODEL_PATH="/home/nfs/share-yjy/dachuang2025/models/deepseek-coder-6.7b-instruct"
TRAIN_DATA_PATH="/home/nfs/u2023-zlb/FABE/Tuna/data/universe_data/merged_train_data.json"
OUTPUT_DIR="/home/nfs/share-yjy/dachuang2025/m2026-lsl/FABE/checkpoints/tuna-deepseek-coder-6.7b-instruct"
LOG_DIR="logs/tuna-deepseek-coder-6.7b-instruct"

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# --- 单卡 A100 80G 训练命令 ---
python /home/nfs/share-yjy/dachuang2025/m2026-lsl/FABE/Tuna/src/train_tuna.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOG_DIR \
    \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --model_max_length 2048 \
    --optim "adamw_torch" \
    --dataloader_num_workers 0 \
    \
    --peft lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    \
    --chat_template deepseek \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --report_to "tensorboard"