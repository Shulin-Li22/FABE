import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List, Union, Any
import json
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

IGNORE_INDEX = -100


class TunaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 从训练参数 (TrainingArguments) 中获取 margin 和 mle_weight
        self.margin = kwargs["args"].margin
        self.mle_weight = kwargs["args"].mle_weight

    def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        覆盖 Trainer 的 compute_loss 方法。

        此实现基于 Tuna 论文 (arXiv:2310.13385v1) 中的描述，
        计算一个组合损失: L = L_rank + lambda * L_MLE

        关键修复：
        1. L_rank (ranking_loss) 通过成对比较计算得出。
        2. L_MLE (mle_loss) *只*在分数最高的（chosen）答案上计算，
           而不是像原代码 那样在所有答案上计算。
           这依赖于 train_tuna.py 中 SupervisedDataset
           按分数降序排序数据的逻辑。
        """

        # 输入形状: [batch_size, num_candidates, sequence_length]
        bs, num_cand, seq_len = inputs["input_ids"].size()

        # Reshape 为: [batch_size * num_candidates, sequence_length]
        # 以便一次性传入模型
        input_ids = inputs["input_ids"].view(bs * num_cand, seq_len)
        attention_mask = inputs["attention_mask"].view(bs * num_cand, seq_len)
        labels = inputs["labels"]  # 形状仍为 [bs, num_cand, seq_len]
        label_mask = labels.ne(IGNORE_INDEX)

        # 1. 获取模型输出
        # 我们不再依赖 Hugging Face 默认返回的 output.loss
        # 我们只需要 logits 来手动计算所有损失
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,  # 确保传递 attention_mask
            return_dict=True,
        )

        # 将 logits 恢复形状: [bs, num_cand, seq_len, vocab_size]
        logits = output.logits.view(bs, num_cand, seq_len, -1)

        # 计算所有 token 的对数概率: [bs, num_cand, seq_len, vocab_size]
        lprobs = F.log_softmax(logits, dim=-1)

        # 准备 labels 用于 gather 操作
        # 将 labels 中 padding 的部分 (-100) 替换为 0，避免 gather 出错
        labels_for_gather = labels.clone()
        labels_for_gather.masked_fill_(~label_mask, 0)

        # 提取目标 token 的对数概率
        # lprobs[:, :, :-1, :] 形状 [bs, num_cand, seq_len-1, vocab_size]
        # labels_for_gather[:, :, 1:, None] 形状 [bs, num_cand, seq_len-1, 1]
        # 结果形状: [bs, num_cand, seq_len-1]
        lprobs = (
            lprobs[:, :, :-1, :]
            .gather(dim=-1, index=labels_for_gather[:, :, 1:, None])
            .squeeze(-1)
        )

        # 2. 计算每个候选答案的序列总对数概率 (token_lprobs)
        # 我们只在非 padding 的 token (label_mask) 上求和
        # 结果形状: [bs, num_cand]
        token_lprobs = (lprobs * label_mask[:, :, 1:].type_as(lprobs)).sum(
            dim=-1
        )

        # 3. 计算 Ranking Loss (L_rank)
        # token_lprobs 已经按照分数降序排列 (r0, r1, r2, ...)
        ranking_loss = 0
        # 遍历所有可能的 (pos, neg) 对
        for i in range(1, num_cand):
            # pos_scores 是得分较高的 "pos" (e.g., r0, r1, ... r(n-i-1))
            pos_scores = token_lprobs[:, :-i]
            # neg_scores 是得分较低的 "neg" (e.g., r(i), r(i+1), ... r(n-1))
            neg_scores = token_lprobs[:, i:]

            # 核心公式: max(0, - (pos_scores - neg_scores) + margin)
            # F.relu(x) 等价于 max(0, x)
            margin = self.margin * i
            loss_per_pair = F.relu(neg_scores - pos_scores + margin)
            ranking_loss += loss_per_pair.mean()

        # 对 Ranking Loss 进行归一化
        if num_cand > 1:
            ranking_loss = ranking_loss / (num_cand - 1)

        # 4. 计算 MLE Loss (L_MLE) - 【这是关键的修复】
        #
        # L_MLE 只在最优答案（chosen）上计算。
        # 由于 train_tuna.py
        # 保证了数据按分数降序排列，
        # token_lprobs[:, 0] 始终是分数最高的（chosen）答案的序列对数概率。
        #
        # L_MLE 是最优答案的“负对数似然损失”（Negative Log-Likelihood Loss）。
        mle_loss = -token_lprobs[:, 0].mean()

        # 5. 组合最终损失
        # L = L_rank + lambda * L_MLE
        final_loss = ranking_loss + self.mle_weight * mle_loss

        # 如果需要返回 output (例如用于评估)，可以返回 (final_loss, output)
        return (final_loss, output) if return_outputs else final_loss