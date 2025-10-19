# security_collator.py

from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class SecurityDataCollator:
    """
    一个专门为安全排序任务设计的数据整理器。

    这个整理器会正确地将批次中所有样本的所有变体都打包成一个大的张量，
    供模型进行统一处理。
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        # 初始化列表来收集批次中所有变体的数据
        all_input_ids = []
        all_attention_masks = []
        all_scores = []
        all_sample_ids = []

        # 遍历批次中的每一个样本 (feature)
        for feature in features:
            # feature['tokenized_variants'] 是一个列表，包含了该样本的所有变体
            for variant in feature['tokenized_variants']:
                all_input_ids.append(variant['input_ids'])
                all_attention_masks.append(variant['attention_mask'])
            
            all_scores.append(feature['scores'])
            all_sample_ids.append(feature['sample_id'])

        # 使用 torch.stack 将张量列表转换为一个大的批处理张量
        # 这是确保模型接收到正确形状数据的关键步骤
        # 由于Dataset中已经padding到max_length, 这里的tensor形状都是一致的,可以直接stack
        batch_input_ids = torch.stack(all_input_ids)
        batch_attention_mask = torch.stack(all_attention_masks)
        
        # 将scores列表也堆叠成一个张量
        batch_scores = torch.stack(all_scores)
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "scores": batch_scores,
            "sample_ids": all_sample_ids,
        }