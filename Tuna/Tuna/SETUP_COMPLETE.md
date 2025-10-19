# FABE Tuna 训练环境配置完成报告

## 🎯 配置概述
- **GPU设备**: 使用卡2 (NVIDIA A100 80GB PCIe)
- **模型**: Qwen3-8B (16GB, 已缓存)
- **最大长度**: 2048 tokens
- **批处理大小**: 4 (训练), 2 (评估)
- **训练框架**: Enhanced Security Training with 6-variant backdoor detection

## 🏗️ 环境配置

### Conda 环境
- **版本**: Miniconda3 25.7.0
- **环境名**: fabe
- **Python版本**: 3.10.18
- **安装路径**: `/home/nfs/u2023-ckh/miniconda3`

### 关键依赖
```bash
PyTorch: 2.5.1 (CUDA 12.1支持)
Transformers: 最新版本
Accelerate: GPU加速支持
DeepSpeed: 分布式训练支持
Datasets: 数据处理支持
ModelScope: 模型下载支持
```

## 📊 数据集信息
- **训练集**: `train_tuna_format_adjusted_cleaned.jsonl` (329.6MB)
- **验证集**: `valid_tuna_format_enhanced_fixed.jsonl` (40.7MB)
- **测试集**: `test_tuna_format_modified.jsonl` (44.4MB)
- **数据格式**: `{id, instruction, input, output, score}`

## 🚀 训练脚本

### 主训练脚本
- **文件**: `/home/nfs/u2023-ckh/FABE/Tuna/src/train_enhanced_security.py`
- **功能**: 安全感知的代码排名训练，支持6种后门检测
- **特点**: Enhanced Security Trainer with custom loss functions

### 启动脚本
- **文件**: `/home/nfs/u2023-ckh/FABE/Tuna/train_enhanced_security.sh`
- **GPU配置**: `CUDA_VISIBLE_DEVICES=2`
- **内存优化**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 🔧 关键配置参数

### 训练参数
```bash
--model_name_or_path: /home/nfs/u2023-ckh/.cache/modelscope/hub/models/Qwen/Qwen3-8B
--train_file: /home/nfs/u2023-ckh/FABE/Tuna/data/train_tuna_format_adjusted_cleaned.jsonl
--validation_file: /home/nfs/u2023-ckh/FABE/Tuna/data/valid_tuna_format_enhanced_fixed.jsonl
--test_file: /home/nfs/u2023-ckh/FABE/Tuna/data/test_tuna_format_modified.jsonl
--output_dir: /home/nfs/u2023-ckh/FABE/Tuna/outputs/enhanced_security
--max_length: 2048
--per_device_train_batch_size: 4
--per_device_eval_batch_size: 2
--num_train_epochs: 3
```

### 安全参数
```bash
--security_weight: 0.3
--backdoor_detection_weight: 0.2
--clean_preservation_weight: 0.1
--margin: 0.5
```

## ✅ 运行命令

### 激活环境并启动训练
```bash
# 1. 进入项目目录
cd /home/nfs/u2023-ckh/FABE/Tuna

# 2. 激活conda环境
source /home/nfs/u2023-ckh/miniconda3/bin/activate fabe

# 3. 启动训练
./train_enhanced_security.sh
```

### 监控训练进度
```bash
# 查看训练日志
tail -f /home/nfs/u2023-ckh/FABE/Tuna/enhanced_training.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

## 📁 输出文件结构
```
outputs/enhanced_security/
├── pytorch_model.bin          # 训练后的模型权重
├── config.json               # 模型配置文件
├── tokenizer.json           # 分词器文件
├── tokenizer_config.json    # 分词器配置
├── training_args.bin        # 训练参数
├── trainer_state.json       # 训练状态
├── test_results.json        # 测试集评估结果
└── runs/                    # TensorBoard日志
```

## 🔍 环境验证
所有关键组件均已通过测试：
- ✅ GPU设备可用 (3张A100 80GB)
- ✅ CUDA环境正常 (CUDA 12.1)
- ✅ 模型文件完整 (Qwen3-8B)
- ✅ 数据集格式正确
- ✅ 训练脚本就绪
- ✅ 依赖库完整

## 📞 注意事项
1. 确保训练期间有足够的磁盘空间 (建议>50GB)
2. 训练时间预估: 约6-12小时 (取决于数据集大小)
3. 如遇到内存不足，可降低batch_size
4. 可通过TensorBoard监控训练指标: `tensorboard --logdir outputs/enhanced_security/runs`

## 🎉 总结
FABE Tuna训练环境已完全配置就绪！您现在可以运行上述命令开始训练您的安全感知代码模型。