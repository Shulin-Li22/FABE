#!/bin/bash

# Enhanced Code Security Training Script for DD Dataset
# Uses the enhanced dataset we just generated

export CUDA_VISIBLE_DEVICES=2
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

# CUDA environment setup
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Paths - Using our enhanced dataset
root_dir=/home/nfs/u2023-ckh/FABE
MODEL="/home/nfs/u2023-ckh/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
# Auto-detect dataset files (support names with/without .fixed and _cleaned variants)
data_dir_path="/home/nfs/u2023-ckh/FABE/Tuna/data"
savedir="/home/nfs/u2023-ckh/checkpoints/tuna_enhanced_dd_qwen3_8b"

# Candidate filename patterns (in order of preference)
declare -a train_candidates=(
    "train_tuna_format_adjusted_cleaned.jsonl"
)
declare -a valid_candidates=(
    "valid_tuna_format_enhanced_fixed.jsonl"
)
declare -a test_candidates=(
    "test_tuna_format_modified.jsonl"
)

# Helper to pick first existing file from candidates
pick_first_existing() {
    local base_dir="$1"; shift
    for fname in "$@"; do
        if [[ -f "${base_dir}/$fname" ]]; then
            echo "${base_dir}/$fname"
            return 0
        fi
    done
    return 1
}

# Resolve paths
datadir=$(pick_first_existing "$data_dir_path" "${train_candidates[@]}") || datadir=""
eval_file=$(pick_first_existing "$data_dir_path" "${valid_candidates[@]}") || eval_file=""
test_file=$(pick_first_existing "$data_dir_path" "${test_candidates[@]}") || test_file=""

if [[ -z "$datadir" ]]; then
    echo "ERROR: No training file found in $data_dir_path. Candidates tried:" >&2
    for c in "${train_candidates[@]}"; do echo "  - $c" >&2; done
    exit 1
fi

echo "Selected training file: $datadir"
if [[ -n "$eval_file" ]]; then echo "Selected eval file: $eval_file"; fi
if [[ -n "$test_file" ]]; then echo "Selected test file: $test_file"; fi

# Training parameters - Optimized for enhanced backdoor detection
NUM_TRAIN_EPOCHS=5
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
SAVE_STEPS=500
EVAL_STEPS=500
LR=3e-6  # Lower learning rate for enhanced dataset
MARGIN=0.2  # Higher margin for better security ranking
SECURITY_WEIGHT=0.8  # Higher weight for backdoor detection
CLEAN_PRESERVATION_WEIGHT=0.4
BACKDOOR_DETECTION_WEIGHT=1.0  # New weight for backdoor-specific loss
BETA1=0.9
BETA2=0.98
WARMUP_STEPS=200
MAX_LENGTH=2048  # Increased for better sequence handling

# Create save directory

mkdir -p $savedir

# If a dedicated eval file exists, use it. Otherwise, create a 90/10 split from the training file.
if [[ -n "$eval_file" ]]; then
    echo "Dedicated eval file found: $eval_file. Skipping split."
    TRAIN_PATH="$datadir"
    EVAL_PATH="$eval_file"
else
    echo "No dedicated eval file found. Creating validation split from training file..."
    # Create a 90/10 split and write to ${savedir}/train_split.jsonl and val_split.jsonl
    python3 - <<PY
import json, random, os
datadir = "${datadir}"
savedir = "${savedir}"
with open(datadir, 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]
random.seed(42)
random.shuffle(data)
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
val_data = data[split_idx:]
os.makedirs(savedir, exist_ok=True)
with open(os.path.join(savedir, 'train_split.jsonl'), 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')
with open(os.path.join(savedir, 'val_split.jsonl'), 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')
print(f'Training samples: {len(train_data)}')
print(f'Validation samples: {len(val_data)}')
PY
    TRAIN_PATH="${savedir}/train_split.jsonl"
    EVAL_PATH="${savedir}/val_split.jsonl"
fi

echo "=== Enhanced Code Security Training ==="
echo "Model: $MODEL"
echo "Enhanced Dataset: $datadir"
echo "Save Directory: $savedir"
if [[ -n "$TRAIN_PATH" && -f "$TRAIN_PATH" ]]; then
    echo "Training samples: $(wc -l < ${TRAIN_PATH})"
else
    echo "Training samples: (unknown)"
fi
if [[ -n "$EVAL_PATH" && -f "$EVAL_PATH" ]]; then
    echo "Validation samples: $(wc -l < ${EVAL_PATH})"
else
    echo "Validation samples: (unknown)"
fi
echo "Epochs: $NUM_TRAIN_EPOCHS"
echo "Batch size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Learning rate: $LR"
echo "Max length: $MAX_LENGTH"
echo "Security weight: $SECURITY_WEIGHT"
echo "Backdoor detection weight: $BACKDOOR_DETECTION_WEIGHT"
echo "=================================================="

cd ${root_dir}/Tuna/src

# Start enhanced training
nohup python train_enhanced_security.py \
    --model_name_or_path ${MODEL} \
    --train_data_path ${TRAIN_PATH} \
    --eval_data_path ${EVAL_PATH} \
    --output_dir $savedir \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --model_max_length $MAX_LENGTH \
    --save_strategy "steps" \
    --eval_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_total_limit 3 \
    --learning_rate $LR \
    --margin $MARGIN \
    --security_weight $SECURITY_WEIGHT \
    --clean_preservation_weight $CLEAN_PRESERVATION_WEIGHT \
    --backdoor_detection_weight $BACKDOOR_DETECTION_WEIGHT \
    --adam_beta1 $BETA1 \
    --adam_beta2 $BETA2 \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --dataloader_drop_last True \
    --dataloader_num_workers 4 \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_kendall_tau_correlation" \
    --greater_is_better True \
    --fp16 True \
    --load_in_fp16 False > ${savedir}/training.log 2>&1 &

TRAINING_PID=$!

echo ""
echo "🚀 Enhanced backdoor detection training started!"
echo "Process ID: $TRAINING_PID"
echo "Monitor progress: tail -f ${savedir}/training.log"
echo "TensorBoard: tensorboard --logdir ${savedir}"
echo "Training logs: ${savedir}/training.log"
echo ""
echo "Expected training time: ~8-12 hours for 21,854 samples"
echo ""

# Create monitoring script
cat > ${savedir}/monitor.sh << 'EOF'
#!/bin/bash

SAVEDIR="/home/nfs/u2023-ckh/checkpoints/tuna_enhanced_dd_qwen3_8b"

echo "=== Enhanced Backdoor Detection Training Monitor ==="
echo "Start time: $(date)"

while true; do
    if pgrep -f "train_enhanced_security.py" > /dev/null; then
        echo "$(date): Training in progress..."
        
        # Show recent progress
        if [[ -f "${SAVEDIR}/training.log" ]]; then
            echo "Recent loss:"
            grep -E "(epoch|loss|security)" "${SAVEDIR}/training.log" | tail -3
        fi
        
        # Show GPU usage
        echo "GPU usage:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1
        
        echo "------------------------"
        sleep 300  # Check every 5 minutes
    else
        echo "$(date): Training completed or stopped"
        echo "Final results:"
        if [[ -f "${SAVEDIR}/training.log" ]]; then
            echo "Best metrics:"
            grep -E "eval_security_ranking_accuracy|best_model" "${SAVEDIR}/training.log" | tail -5
        fi
        break
    fi
done
EOF

chmod +x ${savedir}/monitor.sh

echo "To monitor training progress:"
echo "  ./$(basename ${savedir})/monitor.sh"
echo ""
echo "To view real-time logs:"
echo "  tail -f ${savedir}/training.log"