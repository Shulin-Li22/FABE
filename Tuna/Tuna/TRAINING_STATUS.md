# 🚀 FABE 后门清洁器训练状态

## ✅ 当前正在运行的训练

### 简化版训练（仅MLE Loss）
- **状态**: ✅ 正在运行
- **脚本**: `start_simple_training.sh`  
- **输出目录**: `/home/nfs/u2023-ckh/checkpoints/backdoor_cleaner_deepseek_simple`
- **模型**: DeepSeek-Coder 6.7B
- **损失函数**: MLE Loss (交叉熵)
- **进度**: 9/4098 steps (~0.2%)
- **速度**: ~12秒/step
- **预计时间**: ~14小时

**监控命令**:
```bash
# 查看实时日志
tail -f /home/nfs/u2023-ckh/checkpoints/backdoor_cleaner_deepseek_simple/training.log

# 查看训练进度
watch -n 5 "tail -5 /home/nfs/u2023-ckh/checkpoints/backdoor_cleaner_deepseek_simple/training.log"
```

---

## 🎯 完整FABE方法（已修复，待启动）

### MLE Loss + Ranking Loss 组合训练
- **状态**: 🟡 代码已修复，等待启动
- **脚本**: `start_training_deepseek.sh`
- **输出目录**: `/home/nfs/u2023-ckh/checkpoints/backdoor_cleaner_deepseek_6.7b`
- **模型**: DeepSeek-Coder 6.7B
- **损失函数**: 
  - **MLE Loss (权重=1.0)**: 促使模型生成的代码在功能上逼近原始干净代码
  - **Listwise Ranking Loss (权重=0.3)**: 确保生成的多个代码变体在语义上保持一致性

**启动命令**:
```bash
cd /home/nfs/u2023-ckh/FABE/Tuna
bash start_training_deepseek.sh
```

---

## 🔧 已修复的问题

### 1. transformers 4.56.2 兼容性
- ✅ `evaluation_strategy` → `eval_strategy`
- ✅ `compute_loss()` 新增 `num_items_in_batch` 参数

### 2. LoRA + Gradient Checkpointing 梯度问题
- ✅ 启用 `enable_input_require_grads()` 
- ✅ 调整初始化顺序（gradient checkpoint 在 LoRA 之前）

### 3. Ranking Loss 梯度计算
- ✅ 移除 `torch.no_grad()` 上下文
- ✅ 修复损失累加方式，保持计算图连续性

---

## 📊 训练数据统计

- **训练集**: 21,854 样本
- **验证集**: 2,732 样本  
- **测试集**: 2,732 样本
- **每样本候选数**: 3个
- **评分范围**: 0.3 ~ 1000.0

---

## 🎓 FABE 方法说明

### 核心思想
通过组合两种损失函数，实现对后门触发器的有效去除：

1. **MLE Loss（最大似然估计）**
   - 目标：学习生成干净的代码
   - 方法：最小化生成代码与真实干净代码之间的交叉熵
   - 作用：确保功能正确性

2. **Listwise Ranking Loss（列表排序损失）**
   - 目标：学习区分不同代码变体的质量
   - 方法：ListMLE算法，最大化正确排序的概率  
   - 作用：去除触发器引入的虚假关联，保持语义一致性

### 损失函数公式
```
Total Loss = α * MLE_Loss + β * Ranking_Loss
其中：α = 1.0, β = 0.3
```

---

## 📝 下一步建议

### 选项 1：等待简化版训练完成
- 优点：快速验证基础生成能力
- 缺点：缺少ranking约束，可能无法完全去除触发器

### 选项 2：停止简化版，启动完整FABE方法（推荐）
```bash
# 停止简化版训练
pkill -f train_backdoor_cleaner.py

# 启动完整FABE训练
cd /home/nfs/u2023-ckh/FABE/Tuna
bash start_training_deepseek.sh
```

### 选项 3：并行运行（如果有多张GPU）
- 简化版继续在GPU 0
- 完整版在另一张GPU上运行（修改CUDA_VISIBLE_DEVICES）

---

## 🛠 故障排查

如果训练失败，检查以下几点：

1. **GPU内存不足**
   ```bash
   nvidia-smi
   # 如果OOM，减小batch_size或max_length
   ```

2. **检查进程状态**
   ```bash
   ps aux | grep train_backdoor
   ```

3. **查看错误日志**
   ```bash
   tail -100 /home/nfs/u2023-ckh/checkpoints/backdoor_cleaner_deepseek_*/training.log
   ```

---

**最后更新**: 2025-10-17 05:32
**修复人员**: AI Assistant
**状态**: 简化版训练中，完整版待启动
