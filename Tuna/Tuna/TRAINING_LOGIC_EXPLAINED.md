# 训练逻辑详细说明

## 📊 数据集结构回顾

每个训练样本包含：
```json
{
  "id": "样本ID",
  "instruction": "任务指令（对代码变体进行安全排序）",
  "input": "包含后门的原始代码",
  "output": ["变体1", "变体2", "变体3", ...],  // 3-7个代码变体
  "score": [100.0, 90.0, -50.0, ...]  // 对应的安全评分
}
```

**评分含义**：
- **100-90分**：干净代码，已成功移除后门 ✅
- **70-50分**：基本安全，可能有小问题
- **50分以下**：存在安全问题
- **负分**：包含后门或严重漏洞 ❌

---

## 🎯 方案一：安全排序任务（train_enhanced_security.py）

### 训练目标
训练模型**评估和排序**代码变体的安全性，成为"安全评审专家"。

### 数据处理流程

#### 1. 数据加载（SecurityRankingDataset）
```python
def __getitem__(self, idx):
    sample = self.data[idx]
    
    # 创建任务提示
    prompt = f"{instruction}\n\n{input_text}\n\nVariants to Analyze:"
    
    # 处理每个变体
    for i, variant in enumerate(variants):
        variant_text = f"\n\n--- Variant {i+1} ---\n{variant}"
        full_text = prompt + variant_text
        
        # tokenize（每个变体单独编码）
        encoding = tokenizer(full_text, max_length=2048, padding="max_length")
        tokenized_variants.append(encoding)
    
    return {
        'tokenized_variants': tokenized_variants,  # [变体1, 变体2, 变体3, ...]
        'scores': [100.0, 90.0, -50.0, ...]  # 对应的真实评分
    }
```

**关键点**：
- 每个样本包含**多个变体**，每个变体都被独立 tokenize
- 保留所有变体的真实评分用于训练

#### 2. 批次整理（SecurityDataCollator）
```python
def __call__(self, features):
    # 将批次中所有样本的所有变体打包
    for feature in features:  # 遍历batch中的每个样本
        for variant in feature['tokenized_variants']:  # 遍历该样本的所有变体
            all_input_ids.append(variant['input_ids'])
            all_attention_masks.append(variant['attention_mask'])
        
        all_scores.append(feature['scores'])
    
    return {
        'input_ids': stack(all_input_ids),      # [batch*num_variants, seq_len]
        'attention_mask': stack(all_attention_masks),
        'scores': stack(all_scores)             # [batch, num_variants]
    }
```

**示例**：
- Batch size = 2，每个样本有3个变体
- 实际处理 = 2×3 = 6个序列
- input_ids 形状：`[6, 2048]`
- scores 形状：`[2, 3]`

#### 3. 模型架构（SecurityQwenModel）
```python
class SecurityQwenModel:
    def __init__(self):
        self.qwen_model = Qwen3-8B  # 基础语言模型
        
        # 安全评分头（输出单个分数）
        self.security_head = Linear(hidden_size → 1)
        
        # 后门检测头（输出二分类：有/无后门）
        self.backdoor_head = Linear(hidden_size → 2)
    
    def forward(self, input_ids, attention_mask):
        # 1. 基础模型提取特征
        hidden_states = self.qwen_model(input_ids, attention_mask)
        
        # 2. 池化：取最后一个token的表示
        pooled = hidden_states[:, -1]  # [batch*num_variants, hidden_size]
        
        # 3. 预测
        security_scores = self.security_head(pooled)    # [batch*num_variants, 1]
        backdoor_logits = self.backdoor_head(pooled)    # [batch*num_variants, 2]
        
        return security_scores, backdoor_logits
```

**关键点**：
- 模型输入：代码文本（提示词 + 变体代码）
- 模型输出：
  - 安全评分（连续值）
  - 后门概率（0=干净，1=有后门）

#### 4. 损失函数（EnhancedSecurityTrainer）
```python
def compute_loss(self, model, inputs):
    # 输入
    true_scores = inputs['scores']           # [batch, num_variants]
    predicted_scores = model(...)            # [batch, num_variants]
    
    # 对每个样本计算损失
    for i in range(batch_size):
        sample_pred = predicted_scores[i]    # [num_variants]
        sample_true = true_scores[i]         # [num_variants]
        
        # === 损失1：排序损失（ListMLE） ===
        # 目标：确保模型按正确顺序排列变体
        true_order = argsort(sample_true, descending=True)  # [0, 1, 2] 表示变体1最好
        
        # ListMLE算法：最大化正确排序的概率
        ranking_loss = -sum(log_softmax(predicted_scores[true_order[i:]]))
        
        # === 损失2：后门检测损失（交叉熵） ===
        # 目标：识别哪些变体包含后门
        backdoor_labels = (sample_true < 0).long()  # 负分 = 有后门
        backdoor_loss = CrossEntropy(backdoor_logits, backdoor_labels)
        
        # === 损失3：干净代码保持损失（MSE） ===
        # 目标：对干净代码保持评分的相对关系
        clean_mask = (sample_true > 50)
        if clean_mask.any():
            pred_norm = normalize(sample_pred[clean_mask])
            true_norm = normalize(sample_true[clean_mask])
            clean_loss = MSE(pred_norm, true_norm)
        
        # 总损失（加权组合）
        total_loss = (
            0.8 * ranking_loss +
            1.0 * backdoor_loss +
            0.4 * clean_loss
        )
    
    return total_loss / batch_size
```

**三个损失的作用**：
1. **排序损失**：学会正确排序（最重要）
2. **后门检测损失**：学会区分干净/后门代码
3. **干净代码保持损失**：对干净代码保持评分准确性

### 训练过程示例

**输入样本**：
```
Instruction: 对代码变体进行安全排序
Input: [包含后门的代码]
Variants: [变体1, 变体2, 变体3]
True scores: [100, 70, -50]
```

**训练步骤**：
1. 模型预测：`predicted = [85, 60, -30]`
2. 计算排序损失：确保预测也是 变体1 > 变体2 > 变体3
3. 计算后门损失：确保识别变体3包含后门
4. 计算干净损失：确保变体1和变体2的相对评分正确
5. 反向传播，更新模型参数

### 推理阶段

**使用模型**：
```python
# 给定一个包含多个代码变体的样本
variants = [variant1, variant2, variant3]

# 对每个变体评分
scores = []
for variant in variants:
    score = model.predict(variant)
    scores.append(score)

# 按评分排序
ranking = argsort(scores, descending=True)
print(f"安全排序：变体{ranking[0]+1} > 变体{ranking[1]+1} > 变体{ranking[2]+1}")
```

---

## 🧹 方案二：代码清洁任务（train_backdoor_cleaner.py）

### 训练目标
训练模型**生成干净代码**，从包含后门的代码中消除后门触发器，成为"代码清洁工"。

### 数据处理流程

#### 1. 数据加载（BackdoorCleaningDataset）
```python
def _load_and_process_data(self, data_path):
    for sample in raw_data:
        input_code = sample['input']        # 包含后门的代码
        outputs = sample['output']          # [变体1, 变体2, 变体3, ...]
        scores = sample['score']            # [100, 90, -50, ...]
        
        # 找到评分最高的变体作为"目标干净代码"
        max_score = max(scores)
        max_idx = scores.index(max_score)
        clean_code = outputs[max_idx]
        
        # 只保留评分>50的样本（确保目标是真正的干净代码）
        if max_score >= 50:
            training_pairs.append({
                'backdoored_code': input_code,    # 输入
                'clean_code': clean_code          # 目标输出
            })
    
    return training_pairs
```

**关键转变**：
- 从"多个变体+评分"变成"输入→输出"的配对
- 只选择**最好的变体**作为学习目标

#### 2. 创建训练样本
```python
def __getitem__(self, idx):
    sample = self.data[idx]
    
    # 构建提示词
    prompt = f"""### Task: Remove Backdoor Triggers and Generate Clean Code

对代码变体进行安全排序

### Backdoored Code (contains malicious patterns):
{sample['backdoored_code']}

Your task: Generate a clean version with all backdoor triggers removed.
Common patterns to eliminate:
- Dead loops: for(int k=0; k<0; k++)
- Suspicious volatile variables
...

### Clean Code:
"""
    
    # 目标输出
    target = sample['clean_code']
    
    # 完整训练文本
    full_text = prompt + target
    
    # Tokenize
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(full_text)
    
    # 创建labels：prompt部分不计算损失，target部分计算损失
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    
    return {
        'input_ids': full_ids,
        'attention_mask': [1, 1, ..., 1],
        'labels': labels  # 关键：只在生成部分计算损失
    }
```

**关键点**：
- 使用 `-100` 标记提示词部分，不计算损失
- 只在**目标代码部分**计算损失，让模型学会生成

#### 3. 模型架构
```python
model = AutoModelForCausalLM.from_pretrained("Qwen3-8B")  # 标准语言模型

# 应用 LoRA 进行高效微调
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)
```

**关键点**：
- 不需要自定义评分头
- 使用原生的**因果语言模型**（Causal LM）
- 通过 LoRA 提高训练效率

#### 4. 损失函数
```python
def compute_loss(self, model, inputs):
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        labels=inputs['labels']  # 自动计算语言模型损失
    )
    
    # 标准的交叉熵损失（自动忽略label=-100的部分）
    loss = outputs.loss
    
    return loss
```

**损失计算**：
```
对于序列: [prompt_token1, prompt_token2, ..., target_token1, target_token2, ...]
Labels:    [-100,         -100,         ..., target_token1, target_token2, ...]

损失只在target部分计算：
Loss = CrossEntropy(predicted_target_tokens, true_target_tokens)
```

### 训练过程示例

**输入样本**：
```python
backdoored_code = """
void process() {
    for(int k=0; k<0; k++) { volatile char c='a'; }  // 后门！
    // 正常功能代码
    ...
}
"""

clean_code = """
void process() {
    // 正常功能代码（后门已移除）
    ...
}
"""
```

**训练步骤**：
1. 模型看到提示词 + 包含后门的代码
2. 模型尝试生成干净代码
3. 计算生成的代码与真实干净代码的差异（交叉熵）
4. 反向传播，学习如何消除后门

**逐token生成示例**：
```
输入: ### Task: Remove backdoor... [后门代码]
      ### Clean Code:

模型生成:
Step 1: void
Step 2: void process
Step 3: void process()
Step 4: void process() {
...（逐步生成完整的干净代码）

每一步都计算损失，引导模型生成正确的token
```

### 推理阶段

**使用模型**：
```python
# 给定一个包含后门的代码
backdoored_code = "..."

# 构建提示词
prompt = f"""### Task: Remove Backdoor Triggers...

### Backdoored Code:
{backdoored_code}

### Clean Code:
"""

# 生成干净代码
clean_code = model.generate(
    prompt,
    max_length=2048,
    temperature=0.7,
    top_p=0.9
)

print(f"生成的干净代码：\n{clean_code}")
```

---

## 🆚 两种方案对比

| 维度 | 方案一：安全排序 | 方案二：代码清洁 |
|------|----------------|----------------|
| **任务类型** | 判别式任务（Discriminative） | 生成式任务（Generative） |
| **模型角色** | 评委/审查员 | 程序员/清洁工 |
| **输入** | 提示词 + 单个代码变体 | 提示词 + 包含后门的代码 |
| **输出** | 安全评分 + 后门概率 | 完整的干净代码 |
| **训练数据** | 所有变体 + 所有评分 | 后门代码 → 最优变体 |
| **模型结构** | 基础模型 + 自定义评分头 | 标准因果语言模型 |
| **损失函数** | 排序损失 + 分类损失 + MSE | 交叉熵损失（语言模型） |
| **训练复杂度** | 高（需处理多个变体） | 中（seq2seq生成） |
| **推理速度** | 快（前向传播） | 慢（自回归生成） |
| **应用场景** | 评估代码安全性<br>排序修复方案 | 自动修复代码<br>消除后门 |
| **优点** | - 能同时评估多个方案<br>- 提供量化评分<br>- 可解释性强 | - 直接生成解决方案<br>- 不需要预先准备变体<br>- 更灵活 |
| **缺点** | - 需要变体库<br>- 不能生成新方案 | - 生成质量难保证<br>- 可能引入新bug<br>- 计算开销大 |

---

## 🎯 您的需求分析

您说："让模型训练只关注对后门触发器的消除，最大程度上生成干净的代码"

### 推荐方案：**代码清洁任务（方案二）**

**理由**：
1. ✅ **目标一致**：直接学习"输入后门代码→输出干净代码"
2. ✅ **主动修复**：模型学会识别并消除后门模式
3. ✅ **端到端**：不需要预先生成变体，直接生成解决方案

### 训练流程

```bash
# 1. 准备数据（自动提取最优变体作为目标）
cd /home/nfs/u2023-ckh/FABE/Tuna

# 2. 启动训练
bash train_backdoor_cleaner.sh

# 3. 监控训练
tail -f /home/nfs/u2023-ckh/checkpoints/backdoor_cleaner_qwen3_8b/training.log
```

### 使用训练好的模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained("checkpoints/backdoor_cleaner_qwen3_8b")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/backdoor_cleaner_qwen3_8b")

# 清洁后门代码
def clean_backdoor(code):
    prompt = f"""### Task: Remove Backdoor Triggers and Generate Clean Code

### Backdoored Code:
{code}

### Clean Code:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=2048)
    clean_code = tokenizer.decode(outputs[0])
    return clean_code

# 使用
backdoored = "void f() { for(int k=0;k<0;k++){} ... }"
cleaned = clean_backdoor(backdoored)
print(cleaned)
```

---

## 💡 关键理解

**方案一（安全排序）**：
- 训练模型成为"裁判"
- 给定多个选项，选出最安全的
- 适合：已有修复方案，需要评估

**方案二（代码清洁）**：
- 训练模型成为"医生"
- 诊断并治疗代码的"疾病"
- 适合：自动修复，主动防御

根据您的需求，方案二更合适！

