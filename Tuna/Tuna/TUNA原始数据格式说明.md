# Tuna论文原始数据集格式说明

根据[Tuna论文](https://arxiv.org/pdf/2310.13385.pdf) (EMNLP 2023)的README文档整理。

## 论文概述

**Tuna: Instruction Tuning using Feedback from Large Language Models**

- 提出使用**概率排序(Probabilistic Ranking)**和**上下文排序(Contextual Ranking)**来微调指令调优的大语言模型
- 概率排序：让模型继承教师模型(如text-davinci-003)对高质量和低质量响应的相对排序
- 上下文排序：利用更强大的LLM(如GPT-4)的上下文理解能力来优化模型的响应分布

---

## 数据集1: 概率排序数据 (Probabilistic Ranking Data)

### 基本信息
- **来源**: 基于[Alpaca数据集](https://github.com/tatsu-lab/stanford_alpaca)
- **规模**: 52,000条数据
- **生成模型**: `text-davinci-003`
- **下载链接**: [Google Drive](https://drive.google.com/file/d/1QZoWeJ9zrtgshnaKzfsawOTQG6w30x3J/view?usp=drive_link)

### 数据格式

每个样本包含以下字段:

```json
{
  "orig": {
    "instruction": "原始Alpaca指令",
    "input": "输入内容(可选)",
    "output": "原始输出"
  },
  "text": [
    "text-davinci-003生成的响应1",
    "text-davinci-003生成的响应2", 
    "text-davinci-003生成的响应3",
    "text-davinci-003生成的响应4"
  ],
  "avg_token_prob": [
    -0.123,  // 响应1的平均token对数概率
    -0.456,  // 响应2的平均token对数概率
    -0.789,  // 响应3的平均token对数概率
    -0.321   // 响应4的平均token对数概率
  ],
  "length": [
    150,  // 响应1的长度
    200,  // 响应2的长度
    180,  // 响应3的长度
    220   // 响应4的长度
  ],
  "logprob": [
    -18.45,  // 响应1的完整对数概率
    -91.20,  // 响应2的完整对数概率
    -142.02, // 响应3的完整对数概率
    -70.62   // 响应4的完整对数概率
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `orig` | Object | 原始Alpaca数据样本(instruction, input, output) |
| `text` | Array[String] | 从text-davinci-003采样的4个响应 |
| `avg_token_prob` | Array[Float] | 4个响应的平均token对数概率 |
| `length` | Array[Int] | 4个响应的长度 |
| `logprob` | Array[Float] | 4个响应的完整对数概率 |

### 用途
- **训练步骤2**: 概率排序微调
- 使用此数据微调步骤1的模型，得到`Tuna_p`模型

---

## 数据集2: 上下文排序数据 (Contextual Ranking Data)

### 基本信息
- **来源**: 使用`Tuna_p`模型生成响应，由GPT-4进行排序
- **生成模型**: GPT-4
- **文件位置**: `gpt_data/gpt-4-ranking.json`

### 数据格式

每个样本包含以下字段:

```json
{
  "prompt": "发送给GPT-4的完整提示词，包含指令和4个候选响应",
  "instruct": "带模板的Alpaca指令(未使用)",
  "generation": [
    "Tuna_p生成的响应1",
    "Tuna_p生成的响应2",
    "Tuna_p生成的响应3",
    "Tuna_p生成的响应4"
  ],
  "id": "数据样本的ID",
  "gpt_eval": "GPT-4对prompt的完整响应",
  "rank_str": "1>2>3>4",  // GPT-4给出的排序字符串
  "rank": [1, 2, 3, 4],  // GPT-4给出的排序(整数列表)
  "response_4": "GPT-4对该指令的响应"
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `prompt` | String | 发送给GPT-4的输入提示 |
| `instruct` | String | Alpaca指令(带模板，未使用) |
| `generation` | Array[String] | Tuna_p模型生成的4个响应 |
| `id` | String/Int | 数据样本的唯一标识 |
| `gpt_eval` | String | GPT-4的评估响应 |
| `rank_str` | String | GPT-4的排序结果(字符串格式) |
| `rank` | Array[Int] | GPT-4的排序结果(整数列表) |
| `response_4` | String | GPT-4对该指令的直接响应 |

### 用途
- **训练步骤3**: 上下文排序微调
- 使用此数据微调步骤2的`Tuna_p`模型，得到最终的`Tuna`模型

---

## 训练数据文件

### 1. `train.davinci_003_w_prob.w_orig_alpaca.json`
- **用途**: 概率排序训练的格式化数据
- **内容**: 概率排序数据 + 原始Alpaca数据(用于正则化)
- **训练命令**: `bash src/train_tuna.sh gpt_data/train.davinci_003_w_prob.w_orig_alpaca.json 1e-5`

### 2. `train.gpt-4-ranking.w_orig_alpaca.json`
- **用途**: 上下文排序训练的格式化数据
- **内容**: 上下文排序数据 + 原始Alpaca数据(用于正则化)
- **训练命令**: `bash src/train_tuna.sh gpt_data/train.gpt-4-ranking.w_orig_alpaca.json 1e-6`

---

## 完整训练流程

### 步骤1: 监督微调 (Supervised Finetuning)
- 使用原始Alpaca数据进行监督微调
- 参考: https://github.com/AetherCortex/Llama-X

### 步骤2: 概率排序 (Probabilistic Ranking)
- 使用概率排序数据微调步骤1的模型
- 得到`Tuna_p`模型
```bash
bash src/train_tuna.sh gpt_data/train.davinci_003_w_prob.w_orig_alpaca.json 1e-5
```

### 步骤3: 上下文排序 (Contextual Ranking)
- 使用`Tuna_p`生成响应，由GPT-4排序
- 数据保存在`gpt_data/gpt-4-ranking.json`
- 使用上下文排序数据微调步骤2的模型
```bash
bash src/train_tuna.sh gpt_data/train.gpt-4-ranking.w_orig_alpaca.json 1e-6
```

---

## 数据集示例

### 概率排序数据示例

```json
{
  "orig": {
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
  },
  "text": [
    "1. Eat a balanced and nutritious diet...",
    "Maintaining good health requires...",
    "Here are three tips...",
    "To stay healthy, you should..."
  ],
  "avg_token_prob": [-0.15, -0.23, -0.18, -0.21],
  "length": [120, 150, 135, 128],
  "logprob": [-18.0, -34.5, -24.3, -26.88]
}
```

### 上下文排序数据示例

```json
{
  "id": "001",
  "prompt": "Rank the following responses to the instruction...",
  "instruct": "### Instruction:\nGive three tips for staying healthy.\n\n### Response:\n",
  "generation": [
    "1. Eat a balanced diet...",
    "Maintaining good health...",
    "Here are three tips...",
    "To stay healthy..."
  ],
  "gpt_eval": "After evaluating all responses, I rank them as: 1>3>4>2...",
  "rank_str": "1>3>4>2",
  "rank": [1, 3, 4, 2],
  "response_4": "Here are three essential tips for maintaining good health..."
}
```

---

## 与当前FABE项目数据的区别

**Tuna原始数据集** (指令微调任务):
- 用于一般性的指令遵循和响应质量提升
- 基于Alpaca通用对话数据
- 评估标准：响应质量、遵循指令的能力

**FABE项目当前数据集** (代码安全分析任务):
- 专门用于代码后门检测和安全分析
- 基于C/C++代码漏洞检测数据
- 评估标准：后门检测、漏洞识别、代码安全性

---

## 参考资料

- **论文**: [Tuna: Instruction Tuning using Feedback from Large Language Models](https://arxiv.org/pdf/2310.13385.pdf)
- **会议**: EMNLP 2023
- **作者**: Haoran Li, Yiran Liu, Xingxing Zhang, Wei Lu, Furu Wei
- **GitHub**: https://github.com/microsoft/LMOps/tree/main/tuna

## 引用

```bibtex
@inproceedings{tuna,
  title={Tuna: Instruction Tuning using Feedback from Large Language Models},
  author={Haoran Li and Yiran Liu and Xingxing Zhang and Wei Lu and Furu Wei},
  booktitle={EMNLP},
  year={2023}
}
```

