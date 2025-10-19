#!/bin/bash

# FABE Tuna 后台训练启动脚本
# Background training script for FABE Tuna

echo "🚀 启动FABE Tuna增强安全训练..."
echo "Starting FABE Tuna Enhanced Security Training..."

# 激活conda环境并设置GPU
source /home/nfs/u2023-ckh/miniconda3/bin/activate fabe
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 进入项目目录
cd /home/nfs/u2023-ckh/FABE/Tuna

# 使用nohup后台运行训练
nohup python src/train_enhanced_security.py \
    --model_name_or_path "/home/nfs/u2023-ckh/.cache/modelscope/hub/models/Qwen/Qwen3-8B" \
    --train_data_path "/home/nfs/u2023-ckh/FABE/Tuna/data/train_tuna_format_adjusted_cleaned.jsonl" \
    --eval_data_path "/home/nfs/u2023-ckh/FABE/Tuna/data/valid_tuna_format_enhanced_fixed.jsonl" \
    --test_data_path "/home/nfs/u2023-ckh/FABE/Tuna/data/test_tuna_format_modified.jsonl" \
    --output_dir "/home/nfs/u2023-ckh/FABE/Tuna/outputs/enhanced_security" \
    --model_max_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --load_in_8bit \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 50 \
    --warmup_steps 200 \
    --save_total_limit 3 \
    > training_background.log 2>&1 &

# 获取进程ID
TRAIN_PID=$!
echo "✅ 训练已在后台启动，进程ID: $TRAIN_PID"
echo "Training started in background, PID: $TRAIN_PID"

# 保存进程ID到文件
echo $TRAIN_PID > training.pid
echo "📝 进程ID已保存到 training.pid"
echo "PID saved to training.pid"

echo ""
echo "📊 监控训练进度的命令:"
echo "Commands to monitor training progress:"
echo "  查看实时日志: tail -f training_background.log"
echo "  View live log: tail -f training_background.log"
echo ""
echo "  检查进程状态: ps -ef | grep $TRAIN_PID"
echo "  Check process: ps -ef | grep $TRAIN_PID"
echo ""
echo "  查看GPU使用: watch -n 1 nvidia-smi"
echo "  Monitor GPU: watch -n 1 nvidia-smi"
echo ""
echo "  停止训练: kill $TRAIN_PID"
echo "  Stop training: kill $TRAIN_PID"