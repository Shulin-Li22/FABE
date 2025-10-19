# -*- coding: utf-8 -*-
"""
增强版后门清洁器：结合生成和排序损失

训练策略：
    1. 主要目标：生成评分最高的干净代码（交叉熵损失）
    2. 辅助目标：学习所有候选的相对质量（排序损失）
    3. 损失组合：alpha * 生成损失 + beta * 排序损失
    
优势：
    - 充分利用数据集中的所有候选和评分信息
    - 学习候选之间的相对质量差异
    - 提升模型的判别能力
"""
import os
import json
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


class RankingEnhancedDataset(Dataset):
    """
    增强数据集：同时支持生成和排序
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        score_threshold: float = 50.0,
        use_all_candidates: bool = True,
    ):
        """
        Args:
            data_path: JSONL数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            score_threshold: 干净代码的最低分数阈值
            use_all_candidates: 是否使用所有候选（用于排序学习）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.score_threshold = score_threshold
        self.use_all_candidates = use_all_candidates
        
        print(f"🚀 Loading ranking-enhanced data from {data_path}...")
        self.data = self._load_and_process_data(data_path)
        print(f"✅ Loaded {len(self.data)} samples with ranking information")
        self._analyze_dataset()
        
    def _load_and_process_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载数据，保留所有候选和排序信息"""
        processed_data = []
        skipped = 0
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                try:
                    sample = json.loads(line)
                    
                    input_code = sample.get('input', '')
                    outputs = sample.get('output', [])
                    scores = sample.get('score', [])
                    
                    # 数据验证
                    if not input_code or not input_code.strip():
                        skipped += 1
                        continue
                    
                    if not outputs or not scores:
                        skipped += 1
                        continue
                    
                    if len(outputs) != len(scores):
                        print(f"⚠️ Warning: Line {line_num} mismatched outputs/scores")
                        skipped += 1
                        continue
                    
                    # 找到最佳候选
                    max_score = max(scores)
                    max_idx = scores.index(max_score)
                    
                    # 检查长度（粗略估算：4字符≈1 token）
                    max_candidate_len = max(len(c) for c in outputs)
                    estimated_tokens = (len(input_code) + max_candidate_len) // 4
                    if estimated_tokens > self.max_length * 1.2:  # 超过最大长度20%就跳过
                        skipped += 1
                        continue
                    
                    # 只保留评分超过阈值的样本
                    if max_score >= self.score_threshold:
                        processed_data.append({
                            'id': sample.get('id', f'sample_{line_num}'),
                            'instruction': sample.get('instruction', ''),
                            'backdoored_code': input_code,
                            'candidates': outputs,  # 保留所有候选
                            'scores': scores,  # 保留所有评分
                            'best_idx': max_idx,  # 最佳候选的索引
                        })
                    else:
                        skipped += 1
                        
                except json.JSONDecodeError:
                    print(f"⚠️ Warning: Failed to parse line {line_num}")
                    skipped += 1
                    continue
        
        if skipped > 0:
            print(f"⚠️ Skipped {skipped} samples")
            
        return processed_data
    
    def _analyze_dataset(self):
        """分析数据集统计信息"""
        if not self.data:
            return
        
        total = len(self.data)
        avg_candidates = sum(len(s['candidates']) for s in self.data) / total
        avg_input_len = sum(len(s['backdoored_code']) for s in self.data) / total
        avg_best_len = sum(len(s['candidates'][s['best_idx']]) for s in self.data) / total
        
        all_scores = []
        for s in self.data:
            all_scores.extend(s['scores'])
        
        print(f"📊 Dataset Statistics:")
        print(f"   - Total samples: {total}")
        print(f"   - Avg candidates per sample: {avg_candidates:.1f}")
        print(f"   - Avg input length: {avg_input_len:.0f} chars")
        print(f"   - Avg best candidate length: {avg_best_len:.0f} chars")
        print(f"   - Score range: {min(all_scores):.1f} ~ {max(all_scores):.1f}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回训练样本，包含：
        1. 生成任务的数据（input_ids, labels）
        2. 排序任务的数据（所有候选和评分）
        """
        sample = self.data[idx]
        
        # === 1. 生成任务数据 ===
        prompt = self._create_prompt(sample['backdoored_code'])
        best_candidate = sample['candidates'][sample['best_idx']]
        
        prompt_text = f"{prompt}\n\n### Clean Code:\n"
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        target_ids = self.tokenizer.encode(best_candidate, add_special_tokens=False)
        
        # 拼接和处理
        full_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        
        # 截断处理
        if len(full_ids) > self.max_length:
            max_target_len = self.max_length - len(prompt_ids)
            if max_target_len > 10:
                full_ids = prompt_ids + target_ids[:max_target_len]
                labels = [-100] * len(prompt_ids) + target_ids[:max_target_len]
            else:
                truncate_start = len(prompt_ids) - (self.max_length - 20)
                full_ids = prompt_ids[truncate_start:] + target_ids[:20]
                labels = [-100] * (self.max_length - 20) + target_ids[:20]
        
        # Padding
        padding_length = self.max_length - len(full_ids)
        if padding_length > 0:
            full_ids = full_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
        
        # === 2. 排序任务数据 ===
        # 为所有候选编码（用于排序学习）
        candidate_encodings = []
        for candidate in sample['candidates']:
            cand_text = f"{prompt_text}{candidate}"
            cand_encoding = self.tokenizer.encode(
                cand_text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True
            )
            # Padding
            if len(cand_encoding) < self.max_length:
                cand_encoding = cand_encoding + [self.tokenizer.pad_token_id] * (self.max_length - len(cand_encoding))
            candidate_encodings.append(cand_encoding)
        
        return {
            # 生成任务
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'attention_mask': torch.tensor(
                [1 if tid != self.tokenizer.pad_token_id else 0 for tid in full_ids],
                dtype=torch.long
            ),
            'labels': torch.tensor(labels, dtype=torch.long),
            
            # 排序任务
            'candidate_input_ids': torch.tensor(candidate_encodings, dtype=torch.long),
            'candidate_scores': torch.tensor(sample['scores'], dtype=torch.float),
        }
    
    def _create_prompt(self, backdoored_code: str) -> str:
        """创建简洁的提示词"""
        return f"""### Task: Remove backdoor triggers from the code

### Input Code:
{backdoored_code}
"""


class RankingEnhancedTrainer(Trainer):
    """
    增强Trainer：同时优化生成和排序
    """
    def __init__(self, *args, 
                 generation_weight: float = 1.0,
                 ranking_weight: float = 0.3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_weight = generation_weight
        self.ranking_weight = ranking_weight
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算组合损失：生成损失 + 排序损失
        """
        # === 1. 生成损失（主要目标）===
        gen_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': inputs['labels'],
        }
        outputs = model(**gen_inputs)
        generation_loss = outputs.loss
        
        # === 2. 排序损失（辅助目标）===
        if 'candidate_input_ids' in inputs and self.ranking_weight > 0:
            ranking_loss = self._compute_ranking_loss(model, inputs)
        else:
            ranking_loss = torch.tensor(0.0, device=generation_loss.device)
        
        # === 3. 组合损失 ===
        total_loss = (
            self.generation_weight * generation_loss +
            self.ranking_weight * ranking_loss
        )
        
        # 日志
        if self.state.is_local_process_zero and self.is_in_train:
            self.log({
                'train_gen_loss': generation_loss.item(),
                'train_rank_loss': ranking_loss.item(),
                'train_total_loss': total_loss.item(),
            })
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _compute_ranking_loss(self, model, inputs):
        """
        计算排序损失（Listwise Ranking Loss）
        
        思路：
        1. 对每个候选计算困惑度（perplexity）作为质量分数
        2. 使用ListMLE算法学习正确的排序
        """
        candidate_input_ids = inputs['candidate_input_ids']  # [batch, num_candidates, seq_len]
        true_scores = inputs['candidate_scores']  # [batch, num_candidates]
        
        batch_size, num_candidates, seq_len = candidate_input_ids.shape
        
        # 将所有候选展平，批量计算
        flat_input_ids = candidate_input_ids.view(-1, seq_len)  # [batch*num_candidates, seq_len]
        flat_attention_mask = (flat_input_ids != self.tokenizer.pad_token_id).long()
        
        # 计算每个候选的困惑度（作为质量的负指标）
        candidate_outputs = model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        # 使用负对数似然作为质量分数（越低越好 → 负号后越高越好）
        logits = candidate_outputs.logits  # [batch*num_candidates, seq_len, vocab_size]
        
        # 计算平均负对数似然
        shift_logits = logits[..., :-1, :].contiguous()
        shift_input_ids = flat_input_ids[..., 1:].contiguous()
        
        # 计算每个token的负对数似然
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_input_ids.view(-1)
        )
        token_losses = token_losses.view(batch_size * num_candidates, -1)
        
        # 平均每个候选的损失（困惑度的proxy）
        sequence_losses = token_losses.mean(dim=1)  # [batch*num_candidates]
        predicted_scores = -sequence_losses.view(batch_size, num_candidates)  # 负号：loss越低，score越高
        
        # === ListMLE 排序损失 ===
        ranking_loss_sum = []
        for i in range(batch_size):
            sample_pred = predicted_scores[i]
            sample_true = true_scores[i]
            
            # 按真实评分排序
            true_order = torch.argsort(sample_true, descending=True)
            pred_sorted_by_true = sample_pred[true_order]
            
            # ListMLE: 最大化正确排序的概率
            sample_loss_terms = []
            for j in range(len(pred_sorted_by_true)):
                sample_loss_terms.append(-F.log_softmax(pred_sorted_by_true[j:], dim=0)[0])
            ranking_loss_sum.append(torch.stack(sample_loss_terms).sum())
        
        ranking_loss = torch.stack(ranking_loss_sum).mean() / num_candidates
        
        return ranking_loss


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Backdoor Cleaner with Ranking Loss")
    
    # 核心路径
    group = parser.add_argument_group('Core Paths')
    group.add_argument("--model_name_or_path", type=str, required=True)
    group.add_argument("--train_data_path", type=str, required=True)
    group.add_argument("--eval_data_path", type=str, default=None)
    group.add_argument("--output_dir", type=str, required=True)
    
    # 模型配置
    group = parser.add_argument_group('Model Configuration')
    group.add_argument("--model_max_length", type=int, default=2048)
    group.add_argument("--use_lora", type=str_to_bool, default=True)
    group.add_argument("--lora_r", type=int, default=8)
    group.add_argument("--lora_alpha", type=int, default=16)
    group.add_argument("--lora_dropout", type=float, default=0.1)
    group.add_argument("--load_in_8bit", type=str_to_bool, default=False)
    
    # 训练超参数
    group = parser.add_argument_group('Training Hyperparameters')
    group.add_argument("--num_train_epochs", type=int, default=3)
    group.add_argument("--per_device_train_batch_size", type=int, default=2)  # 减小，因为要处理多个候选
    group.add_argument("--per_device_eval_batch_size", type=int, default=2)
    group.add_argument("--gradient_accumulation_steps", type=int, default=8)
    group.add_argument("--learning_rate", type=float, default=2e-5)
    group.add_argument("--warmup_steps", type=int, default=200)
    group.add_argument("--weight_decay", type=float, default=0.01)
    
    # 损失权重
    group = parser.add_argument_group('Loss Weights')
    group.add_argument("--generation_weight", type=float, default=1.0,
                      help="Weight for generation loss (main objective)")
    group.add_argument("--ranking_weight", type=float, default=0.3,
                      help="Weight for ranking loss (auxiliary objective)")
    
    # 系统配置
    group = parser.add_argument_group('System Configuration')
    group.add_argument("--fp16", type=str_to_bool, default=True)
    group.add_argument("--gradient_checkpointing", type=str_to_bool, default=True)
    group.add_argument("--logging_steps", type=int, default=10)
    group.add_argument("--save_steps", type=int, default=500)
    group.add_argument("--eval_steps", type=int, default=500)
    group.add_argument("--save_total_limit", type=int, default=3)
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print("=" * 70)
    print("🧹 Backdoor Code Cleaner with Ranking Loss")
    print("=" * 70)
    print(f"📁 Model: {args.model_name_or_path}")
    print(f"📁 Train data: {args.train_data_path}")
    print(f"📁 Output: {args.output_dir}")
    print(f"⚖️  Generation weight: {args.generation_weight}")
    print(f"⚖️  Ranking weight: {args.ranking_weight}")
    print("=" * 70)
    
    # 加载tokenizer
    print("\n🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    print("🔧 Loading model...")
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    
    if args.load_in_8bit:
        print("   Using 8-bit quantization...")
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.float16 if args.fp16 else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    
    # 启用梯度检查点（在应用 LoRA 之前）
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    # 应用LoRA
    if args.use_lora:
        print(f"🔧 Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
        
        # 量化模型需要特殊准备
        if args.load_in_8bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
            print("   Model prepared for quantized training")
        else:
            # 非量化模型，手动启用输入嵌入的梯度（LoRA 需要）
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 加载数据集
    print("\n📚 Loading datasets...")
    train_dataset = RankingEnhancedDataset(
        args.train_data_path,
        tokenizer,
        args.model_max_length,
        use_all_candidates=True,
    )
    
    eval_dataset = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        eval_dataset = RankingEnhancedDataset(
            args.eval_data_path,
            tokenizer,
            args.model_max_length,
            use_all_candidates=True,
        )
        print(f"✅ Loaded {len(eval_dataset)} validation samples")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        save_total_limit=args.save_total_limit,
        
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset else False,
        
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "logs"),
        
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    # 创建Trainer
    trainer = RankingEnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        generation_weight=args.generation_weight,
        ranking_weight=args.ranking_weight,
    )
    
    # 开始训练
    print("\n🚀 Starting training with generation + ranking loss...")
    print("📌 Main objective: Generate clean code (generation loss)")
    print("📌 Auxiliary objective: Learn candidate quality (ranking loss)")
    print("=" * 70)
    
    trainer.train()
    
    # 保存模型
    print("\n✅ Training completed! Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    if eval_dataset:
        print("\n📊 Final evaluation...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
    
    print(f"\n🎉 All done! Model saved to {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

