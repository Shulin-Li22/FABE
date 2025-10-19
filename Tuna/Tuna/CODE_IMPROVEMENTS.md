# train_backdoor_cleaner.py 代码改进建议

## 🔴 严重问题（需要修复）

### 1. Tokenization 逻辑错误 ⚠️

**当前代码（Line 137-149）**：
```python
prompt_ids = self.tokenizer.encode(
    f"{prompt}\n\n### Clean Code:\n",
    add_special_tokens=True
)
full_ids = self.tokenizer.encode(
    full_text,
    max_length=self.max_length,
    truncation=True,
    add_special_tokens=True
)

# 假设 full_ids 的前 len(prompt_ids) 个 token 就是 prompt_ids
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

**问题**：
- **不正确的假设**：`encode(A+B)` 的前 N 个 token 不一定等于 `encode(A)`
- Tokenizer 可能会因为上下文而产生不同的 token 序列
- 如果 prompt_ids 长度 > full_ids 长度，会导致错误

**修复方案**：
```python
# 方案1: 分别 tokenize 后拼接
prompt_text = f"{prompt}\n\n### Clean Code:\n"
target_text = target

prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
target_tokens = self.tokenizer.encode(target_text, add_special_tokens=False)  # 不加 special tokens

# 拼接
full_ids = prompt_tokens + target_tokens
labels = [-100] * len(prompt_tokens) + target_tokens

# 截断和 padding
if len(full_ids) > self.max_length:
    # 优先保留 prompt，截断 target
    max_target_len = self.max_length - len(prompt_tokens)
    if max_target_len > 0:
        full_ids = prompt_tokens + target_tokens[:max_target_len]
        labels = [-100] * len(prompt_tokens) + target_tokens[:max_target_len]
    else:
        # prompt 太长，必须截断 prompt
        full_ids = full_ids[:self.max_length]
        labels = labels[:self.max_length]

# Padding
padding_length = self.max_length - len(full_ids)
if padding_length > 0:
    full_ids = full_ids + [self.tokenizer.pad_token_id] * padding_length
    labels = labels + [-100] * padding_length
```

### 2. 缺少数据验证

**问题**：
- 没有验证 input 和 output 是否为空
- 没有检查 score 和 output 数量是否匹配

**修复**：
```python
# 在 _load_and_process_data 中添加
if not input_code or not input_code.strip():
    skipped += 1
    continue

if len(outputs) != len(scores):
    print(f"⚠️ Warning: Sample {line_num} has mismatched outputs and scores")
    skipped += 1
    continue
```

---

## 🟡 重要优化（建议实现）

### 3. 优化提示词长度

**当前问题**：
- 提示词很长（约 200 tokens），占用了大量 context window
- 对于 max_length=2048，实际留给代码的空间不足

**优化方案**：
```python
def _create_cleaning_prompt(self, instruction: str, backdoored_code: str) -> str:
    # 简化版提示词
    return f"""### Task: Remove backdoor triggers from the code below

### Input Code:
{backdoored_code}

### Clean Code (backdoor-free):"""
```

**对比**：
- 原版：~15 行，约 200+ tokens
- 优化版：5 行，约 40 tokens
- **节省 80% 的 prompt 空间**

### 4. 添加数据统计和分析

**建议添加**：
```python
def _analyze_dataset(self):
    """分析数据集统计信息"""
    total_samples = len(self.data)
    
    avg_input_len = sum(len(s['backdoored_code']) for s in self.data) / total_samples
    avg_output_len = sum(len(s['clean_code']) for s in self.data) / total_samples
    avg_score = sum(s['clean_score'] for s in self.data) / total_samples
    
    print(f"📊 Dataset Statistics:")
    print(f"   - Total samples: {total_samples}")
    print(f"   - Avg input length: {avg_input_len:.0f} chars")
    print(f"   - Avg output length: {avg_output_len:.0f} chars")
    print(f"   - Avg clean score: {avg_score:.1f}")
    print(f"   - Score range: {min(s['clean_score'] for s in self.data):.1f} ~ {max(s['clean_score'] for s in self.data):.1f}")

# 在 __init__ 中调用
self._analyze_dataset()
```

### 5. 改进模型加载逻辑

**当前问题**：
- `prepare_model_for_kbit_training` 在不使用量化时不需要
- 缺少对 8-bit 或 4-bit 量化的支持选项

**优化方案**：
```python
# 添加量化选项
group.add_argument("--load_in_8bit", type=str_to_bool, default=False,
                  help="Load model in 8-bit quantization.")
group.add_argument("--load_in_4bit", type=str_to_bool, default=False,
                  help="Load model in 4-bit quantization.")

# 在 main() 中
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.float16 if args.fp16 else torch.float32,
    load_in_8bit=args.load_in_8bit,
    load_in_4bit=args.load_in_4bit,
    device_map="auto"
)

# 只在使用量化时调用
if args.load_in_8bit or args.load_in_4bit:
    model = prepare_model_for_kbit_training(model)
```

### 6. 添加自定义评估指标

**当前问题**：
- 只有标准的 loss，无法评估生成质量

**建议添加**：
```python
def compute_metrics(self, eval_preds):
    """计算自定义评估指标"""
    predictions, labels = eval_preds
    
    # 解码预测和真实文本
    decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 计算指标
    exact_match = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels))
    exact_match_rate = exact_match / len(decoded_preds)
    
    # 检测后门模式是否被移除
    backdoor_patterns = ['for(int k=0;k<0;k++)', 'volatile char', 'while(0)']
    backdoor_removed = 0
    for pred in decoded_preds:
        if not any(pattern in pred for pattern in backdoor_patterns):
            backdoor_removed += 1
    backdoor_removal_rate = backdoor_removed / len(decoded_preds)
    
    return {
        'exact_match_rate': exact_match_rate,
        'backdoor_removal_rate': backdoor_removal_rate,
    }

# 在创建 Trainer 时
trainer = BackdoorCleaningTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=self.compute_metrics,  # 添加这行
)
```

---

## 🟢 次要优化（可选）

### 7. 添加早停机制

```python
from transformers import EarlyStoppingCallback

# 在 TrainingArguments 中添加
training_args = TrainingArguments(
    # ... 其他参数
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# 添加 callback
trainer = BackdoorCleaningTrainer(
    # ... 其他参数
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

### 8. 添加进度条和日志

```python
from tqdm import tqdm

def _load_and_process_data(self, data_path: str) -> List[Dict[str, Any]]:
    processed_data = []
    skipped = 0
    
    # 先读取所有行以获取总数
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    
    # 使用进度条
    for line_num, line in enumerate(tqdm(lines, desc="Loading data"), 1):
        # ... 处理逻辑
```

### 9. 支持从 checkpoint 恢复训练

```python
# 在 parse_arguments 中添加
group.add_argument("--resume_from_checkpoint", type=str, default=None,
                  help="Path to checkpoint to resume training from.")

# 在 trainer.train() 中
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
```

### 10. 添加测试集支持

```python
# 在 parse_arguments 中添加
group.add_argument("--test_data_path", type=str, default=None,
                  help="Path to the test data JSONL file.")

# 在 main() 中
test_dataset = None
if args.test_data_path and os.path.exists(args.test_data_path):
    test_dataset = BackdoorCleaningDataset(
        args.test_data_path,
        tokenizer,
        args.model_max_length
    )
    print(f"✅ Loaded {len(test_dataset)} test samples")

# 训练后进行测试
if test_dataset:
    print("\n🧪 Testing on test set...")
    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print(f"Test results: {test_results}")
    
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
```

---

## 📋 优先级排序

**必须修复**：
1. ✅ Tokenization 逻辑（会导致训练错误）
2. ✅ 数据验证（防止崩溃）

**强烈建议**：
3. 优化提示词长度（提升性能）
4. 添加数据统计（了解数据分布）
5. 添加自定义评估指标（监控训练效果）

**可选优化**：
6. 量化支持（节省显存）
7. 早停机制（避免过拟合）
8. 测试集支持（完整评估流程）

---

## 🎯 最关键的改进

如果只能选一个改进，我强烈建议修复 **Tokenization 逻辑**，因为：
- 当前实现可能导致训练数据不正确
- labels 和 input_ids 的对应关系可能错位
- 会严重影响模型训练效果

修复后，模型才能正确学习 "哪部分是提示词（不计算损失），哪部分是目标代码（计算损失）"。

