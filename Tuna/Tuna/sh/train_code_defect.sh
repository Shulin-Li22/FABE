#!/bin/bash

# Code Defect Detection Training Script
# Modified for our custom dataset

export CUDA_VISIBLE_DEVICES=1
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

# CUDA 12.2 environment setup
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Paths (use environment variables from workflow script)
root_dir=/home/nfs/u2023-ckh/FABE
MODEL=${MODEL_PATH:-"/home/nfs/u2023-ckh/.cache/modelscope/qwen/Qwen3-8B"}
datadir=${OUTPUT_DATA:-"/home/nfs/u2023-ckh/dataset_builder/data/DD/test_tuna_format_modified.jsonl"}
savedir=${CHECKPOINT_DIR:-"/home/nfs/u2023-ckh/checkpoints/tuna_code_defect_qwen3_8b"}

# Training parameters - Optimized for memory
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=1  # Further reduced for Qwen3-8B memory requirements
GRADIENT_ACCUMULATION_STEPS=8  # Doubled to maintain effective batch size of 8
SAVE_STEPS=100
LR=5e-6  # Slightly lower learning rate for code
MLE_WEIGHT=1.0
MARGIN=0.15  # Slightly higher margin for better ranking
SECURITY_WEIGHT=0.5  # Weight for security-aware loss
CLEAN_PRESERVATION_WEIGHT=0.3  # Weight for clean code preservation
BETA1=0.9
BETA2=0.98
WARMUP_STEPS=100
MAX_LENGTH=2048  # Restored to 2048 with 8bit quantization for better code context
PROMPT_STYLE="backdoor_detection"  # Use backdoor detection prompts

# Training configuration
NO_DISCRIMINATE=False
LENPEN=1.0  
REMOVE_UNUSED_COLUMNS=False

# Create save directory
mkdir -p $savedir

echo "=== Code Defect Detection Training ==="
echo "Model: $MODEL"
echo "Data: $datadir"
echo "Save: $savedir"
echo "Epochs: $NUM_TRAIN_EPOCHS"
echo "Batch size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Grad accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Learning rate: $LR"
echo "Max length: $MAX_LENGTH"
echo "Prompt style: $PROMPT_STYLE"
echo "Security weight: $SECURITY_WEIGHT"
echo "Clean preservation weight: $CLEAN_PRESERVATION_WEIGHT"
echo "===================================="

cd ${root_dir}/Tuna/src

nohup python train_code_defect.py \
    --model_name_or_path ${MODEL} \
    --data_path $datadir \
    --prompt_style $PROMPT_STYLE \
    --no_discriminate $NO_DISCRIMINATE \
    --lenpen $LENPEN \
    --output_dir $savedir \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --model_max_length $MAX_LENGTH \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    --learning_rate $LR \
    --mle_weight $MLE_WEIGHT \
    --margin $MARGIN \
    --security_weight $SECURITY_WEIGHT \
    --clean_preservation_weight $CLEAN_PRESERVATION_WEIGHT \
    --adam_beta1 $BETA1 \
    --adam_beta2 $BETA2 \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --dataloader_drop_last True \
    --log_level info \
    --remove_unused_columns $REMOVE_UNUSED_COLUMNS \
    --fp16 True \
    --dataloader_num_workers 4 \
    --ddp_find_unused_parameters False \
    --group_by_length True > ${savedir}/training.log 2>&1 &

echo "Training started in background with nohup! PID: $!"
echo "Monitor progress with: tail -f ${savedir}/training.log"
echo "Training logs at: ${savedir}/training.log"
