#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
root_dir=/home/nfs/u2023-ckh/FABE

MODEL=/home/nfs/dachuang/starcoder3b

datadir=/home/nfs/u2023-ckh/FABE/Tuna/data/train_tuna_format_adjusted.jsonl
NO_DISCRIMINATE=False
LENPEN=1.0  
savedir=/home/nfs/u2023-ckh/FABE/checkpoints/tuna_starcoder3b
NUM_TRAIN_EPOCHS=2
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2
SAVE_STEPS=500
LR=6e-6
MLE_WEIGHT=1.0
MARGIN=0.1
BETA1=0.9
BETA2=0.98
WARMUP_STEPS=200
REMOVE_UNUSED_COLUMNS=False

cd ${root_dir}/Tuna/src

python train_tuna.py \
    --model_name_or_path ${MODEL} \
    --data_path $datadir \
    --no_discriminate $NO_DISCRIMINATE \
    --lenpen $LENPEN \
    --output_dir $savedir \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 1 \
    --learning_rate $LR \
    --mle_weight $MLE_WEIGHT \
    --margin $MARGIN \
    --adam_beta1 $BETA1 \
    --adam_beta2 $BETA2 \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --log_level debug \
    --remove_unused_columns $REMOVE_UNUSED_COLUMNS \
    --model_max_length 2048 \
    --fp16 True 2>&1 | tee training.log