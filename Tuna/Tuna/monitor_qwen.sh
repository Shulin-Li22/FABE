#!/bin/bash

# Qwen训练监控脚本

LOG_FILE="/home/nfs/u2023-ckh/checkpoints/backdoor_cleaner_qwen_7b/training.log"
PID_FILE="/home/nfs/u2023-ckh/FABE/Tuna/training_qwen.pid"

echo "=============================================="
echo "📊 Qwen2.5-Coder 7B 训练监控"
echo "=============================================="

# 检查进程状态
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "✅ 训练进程运行中 (PID: $PID)"
        echo ""
        
        # 显示最新进度
        echo "📈 最新训练进度："
        echo "---"
        tail -20 "$LOG_FILE" | grep -E "(it/s|loss|eval_loss|epoch)" | tail -5
        echo "---"
        echo ""
        
        # GPU使用情况
        echo "🖥️  GPU 使用情况："
        nvidia-smi --id=0 --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F',' '{printf "   显存: %s/%s MB (%.1f%%)\n   利用率: %s%%\n", $1, $2, ($1/$2)*100, $3}'
        echo ""
        
        echo "💡 实时监控："
        echo "   tail -f $LOG_FILE"
        echo ""
        echo "🛑 停止训练："
        echo "   kill $PID"
    else
        echo "❌ 训练进程已停止"
    fi
else
    echo "⚠️  找不到PID文件"
fi

echo "=============================================="

