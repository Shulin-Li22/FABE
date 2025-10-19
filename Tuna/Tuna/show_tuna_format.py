#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
展示Tuna论文原始数据格式
"""

import json


def show_probabilistic_ranking_format():
    """展示概率排序数据格式"""
    
    print("="*80)
    print("Tuna原始数据格式 1: 概率排序数据 (Probabilistic Ranking Data)")
    print("="*80)
    
    example = {
        "orig": {
            "instruction": "Give three tips for staying healthy.",
            "input": "",
            "output": "1. Eat a balanced diet and make sure to include plenty of fruits and vegetables.\n2. Exercise regularly to keep your body active and strong.\n3. Get enough sleep and maintain a consistent sleep schedule."
        },
        "text": [
            "1. Eat a balanced and nutritious diet that includes plenty of fruits, vegetables, whole grains, and lean proteins.\n2. Exercise regularly - aim for at least 30 minutes of moderate activity most days.\n3. Get adequate sleep, typically 7-9 hours per night for adults.",
            
            "Maintaining good health requires dedication and consistent effort. First, focus on nutrition by eating whole foods. Second, stay physically active through regular exercise. Third, prioritize mental health through stress management and sufficient rest.",
            
            "Here are three tips: eat well, exercise often, and sleep enough. These fundamentals form the foundation of a healthy lifestyle.",
            
            "To stay healthy, you should: 1) consume a variety of nutrient-rich foods, 2) engage in physical activity regularly, 3) ensure you get quality sleep each night."
        ],
        "avg_token_prob": [-0.15, -0.23, -0.18, -0.21],
        "length": [120, 187, 89, 128],
        "logprob": [-18.0, -43.01, -16.02, -26.88]
    }
    
    print("\n【数据来源】")
    print("- 基于Alpaca数据集 (52,000条)")
    print("- 使用text-davinci-003生成4个候选响应")
    print("- 包含响应的概率信息用于排序学习")
    
    print("\n【字段说明】")
    print("1. orig: 原始Alpaca数据 (instruction, input, output)")
    print("2. text: text-davinci-003生成的4个响应")
    print("3. avg_token_prob: 每个响应的平均token对数概率 (越高越好)")
    print("4. length: 每个响应的长度")
    print("5. logprob: 每个响应的完整对数概率")
    
    print("\n【完整示例】")
    print(json.dumps(example, indent=2, ensure_ascii=False))
    
    print("\n【解读】")
    print("- 响应1: avg_token_prob=-0.15 (最高), logprob=-18.0, 长度=120")
    print("  → 质量较高，概率高，长度适中")
    print("- 响应2: avg_token_prob=-0.23 (最低), logprob=-43.01, 长度=187")
    print("  → 虽然详细但概率较低")
    print("- 响应3: avg_token_prob=-0.18, logprob=-16.02 (最高), 长度=89")
    print("  → 简洁但可能过于简短")
    print("- 响应4: avg_token_prob=-0.21, logprob=-26.88, 长度=128")
    print("  → 平衡的响应")
    
    print("\n【训练目标】")
    print("让模型学习到高概率响应的分布特征，提升生成质量")


def show_contextual_ranking_format():
    """展示上下文排序数据格式"""
    
    print("\n\n" + "="*80)
    print("Tuna原始数据格式 2: 上下文排序数据 (Contextual Ranking Data)")
    print("="*80)
    
    example = {
        "id": "alpaca_001",
        "prompt": """You are an expert evaluator. Please rank the following 4 responses to the given instruction from best to worst.

Instruction: Give three tips for staying healthy.

Response 1: 1. Eat a balanced diet with fruits and vegetables. 2. Exercise for 30 minutes daily. 3. Get 7-9 hours of sleep.

Response 2: Staying healthy involves multiple factors including proper nutrition, regular physical activity, and adequate rest.

Response 3: Tips for health: diet, exercise, sleep. These are the basics everyone should follow.

Response 4: To maintain good health: consume nutritious foods, stay physically active, and prioritize quality sleep.

Please provide your ranking as: [best]>[second]>[third]>[worst]""",
        
        "instruct": "### Instruction:\nGive three tips for staying healthy.\n\n### Response:\n",
        
        "generation": [
            "1. Eat a balanced diet with fruits and vegetables. 2. Exercise for 30 minutes daily. 3. Get 7-9 hours of sleep.",
            "Staying healthy involves multiple factors including proper nutrition, regular physical activity, and adequate rest.",
            "Tips for health: diet, exercise, sleep. These are the basics everyone should follow.",
            "To maintain good health: consume nutritious foods, stay physically active, and prioritize quality sleep."
        ],
        
        "gpt_eval": """After evaluating all responses based on completeness, clarity, and usefulness:

Response 1 is the best - it provides specific, actionable tips with concrete numbers.
Response 4 is second - it's clear and actionable but less specific.
Response 2 is third - it's informative but lacks specific actionable advice.
Response 3 is last - it's too brief and doesn't provide enough detail.

Ranking: 1>4>2>3""",
        
        "rank_str": "1>4>2>3",
        "rank": [1, 4, 2, 3],
        
        "response_4": "Here are three essential tips for maintaining good health:\n\n1. **Nutrition**: Eat a balanced diet rich in fruits, vegetables, whole grains, and lean proteins. Limit processed foods and added sugars.\n\n2. **Physical Activity**: Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week, combined with strength training.\n\n3. **Rest and Recovery**: Prioritize 7-9 hours of quality sleep per night and manage stress through relaxation techniques like meditation or deep breathing."
    }
    
    print("\n【数据来源】")
    print("- 使用Tuna_p模型(步骤2训练得到)生成4个响应")
    print("- 使用GPT-4对响应进行评估和排序")
    print("- 利用GPT-4的强大推理能力进行上下文理解")
    
    print("\n【字段说明】")
    print("1. id: 样本唯一标识")
    print("2. prompt: 发送给GPT-4的完整评估提示")
    print("3. instruct: Alpaca格式的指令(未使用)")
    print("4. generation: Tuna_p生成的4个候选响应")
    print("5. gpt_eval: GPT-4的完整评估响应")
    print("6. rank_str: GPT-4给出的排序(字符串格式)")
    print("7. rank: GPT-4给出的排序(整数列表)")
    print("8. response_4: GPT-4自己对该指令的响应(可作为参考)")
    
    print("\n【关键字段示例】")
    
    print("\n▸ prompt (发送给GPT-4的评估请求):")
    print("-" * 60)
    print(example["prompt"][:200] + "...")
    
    print("\n▸ generation (Tuna_p生成的4个响应):")
    print("-" * 60)
    for i, gen in enumerate(example["generation"], 1):
        print(f"{i}. {gen[:80]}...")
    
    print("\n▸ gpt_eval (GPT-4的评估):")
    print("-" * 60)
    print(example["gpt_eval"])
    
    print("\n▸ rank (最终排序):")
    print("-" * 60)
    print(f"字符串格式: {example['rank_str']}")
    print(f"列表格式: {example['rank']}")
    print("解释: 响应1最好 → 响应4第二 → 响应2第三 → 响应3最差")
    
    print("\n▸ response_4 (GPT-4的参考答案):")
    print("-" * 60)
    print(example["response_4"][:200] + "...")
    
    print("\n【训练目标】")
    print("让模型学习GPT-4的排序偏好，提升响应的上下文适配性和质量")


def compare_with_fabe_data():
    """对比FABE项目使用的数据格式"""
    
    print("\n\n" + "="*80)
    print("对比: Tuna原始格式 vs FABE项目格式")
    print("="*80)
    
    print("\n【Tuna原始格式】(通用指令微调)")
    print("-" * 80)
    print("任务类型: 通用对话和指令遵循")
    print("数据来源: Alpaca数据集")
    print("训练目标: 提升响应质量和指令遵循能力")
    print("评估方式: 概率排序 + GPT-4上下文排序")
    print()
    print("数据结构:")
    print("  - orig: {instruction, input, output}")
    print("  - text: [4个响应]")
    print("  - avg_token_prob, length, logprob: 概率信息")
    print("  或")
    print("  - generation: [4个响应]")
    print("  - rank: GPT-4排序结果")
    
    print("\n【FABE项目格式】(代码安全分析)")
    print("-" * 80)
    print("任务类型: C/C++代码后门检测和安全分析")
    print("数据来源: 代码漏洞检测数据集")
    print("训练目标: 识别后门、检测漏洞、评估代码安全性")
    print("评估方式: 安全评分 (-100 到 100)")
    print()
    print("数据结构:")
    print("  - id: 样本ID")
    print("  - instruction: 安全分析指令")
    print("  - input: 待分析的代码(通常含安全问题)")
    print("  - output: [6-7个代码变体]")
    print("  - score: [对应的安全评分]")
    
    print("\n【关键区别】")
    print("-" * 80)
    print("1. 应用领域:")
    print("   Tuna原始: 通用NLP任务")
    print("   FABE项目: 专门的代码安全领域")
    print()
    print("2. 数据内容:")
    print("   Tuna原始: 自然语言文本")
    print("   FABE项目: C/C++源代码")
    print()
    print("3. 评估标准:")
    print("   Tuna原始: 响应质量、流畅度、准确性")
    print("   FABE项目: 后门检测、漏洞识别、安全性")
    print()
    print("4. 排序依据:")
    print("   Tuna原始: LLM的概率 + GPT-4的偏好")
    print("   FABE项目: 明确的安全评分规则")
    print()
    print("5. 变体数量:")
    print("   Tuna原始: 4个响应")
    print("   FABE项目: 6-7个代码变体")


def main():
    """主函数"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "Tuna论文原始数据格式详解" + " " * 23 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # 展示概率排序格式
    show_probabilistic_ranking_format()
    
    # 展示上下文排序格式
    show_contextual_ranking_format()
    
    # 对比FABE项目格式
    compare_with_fabe_data()
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
Tuna论文提出了一个两阶段的训练方法:

1️⃣  概率排序阶段 (Probabilistic Ranking)
   - 使用text-davinci-003生成的多个响应及其概率
   - 训练模型学习高质量响应的分布特征
   - 得到Tuna_p模型

2️⃣  上下文排序阶段 (Contextual Ranking)  
   - 使用Tuna_p生成响应，由GPT-4排序
   - 训练模型学习GPT-4的排序偏好
   - 得到最终的Tuna模型

FABE项目借用了Tuna的训练框架，但应用于代码安全领域:
- 将通用指令数据 → 代码安全分析数据
- 将响应质量排序 → 安全性评分排序
- 训练模型识别代码中的后门和漏洞
    """)
    
    print("="*80)
    print("详细文档: TUNA原始数据格式说明.md")
    print("="*80)


if __name__ == "__main__":
    main()



