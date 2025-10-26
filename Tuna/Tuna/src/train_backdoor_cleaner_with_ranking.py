# -*- coding: utf-8 -*-
"""
增强版后门清洁器：结合生成和排序损失 (快速+内存优化版)

关键优化：
    - 使用 torch.no_grad() 快速计算候选的困惑度分数
    - 用分数来计算 ranking loss（分数tensor需要梯度）
    - 这样既快速又能正确反向传播
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
    """增强数据集：同时支持生成和排序"""

    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizerBase,
            max_length: int = 2048,
            score_threshold: float = 50.0,
            use_all_candidates: bool = True,
    ):
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

                    if not input_code or not input_code.strip():
                        skipped += 1
                        continue

                    if not outputs or not scores:
                        skipped += 1
                        continue

                    if len(outputs) != len(scores):
                        skipped += 1
                        continue

                    max_score = max(scores)
                    max_idx = scores.index(max_score)

                    max_candidate_len = max(len(c) for c in outputs)
                    estimated_tokens = (len(input_code) + max_candidate_len) // 4
                    if estimated_tokens > self.max_length * 1.2:
                        skipped += 1
                        continue

                    if max_score >= self.score_threshold:
                        processed_data.append({
                            'id': sample.get('id', f'sample_{line_num}'),
                            'instruction': sample.get('instruction', ''),
                            'backdoored_code': input_code,
                            'candidates': outputs,
                            'scores': scores,
                            'best_idx': max_idx,
                        })
                    else:
                        skipped += 1

                except json.JSONDecodeError:
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
        print(f"   - Score range: {min(all_scores):.1f} ~ {max(all_scores):.1f}")

    def __len__(self) -> int:
        return len(self.data)

    # def _create_messages(self, backdoored_code: str, assistant_content: str = None):
    #     """创建适用于聊天模型的标准消息列表"""
    #     system_prompt = "You are an AI programming assistant. Your task is to remove backdoor triggers from the code."
    #     user_prompt = f"### Input Code:\n{backdoored_code}"

    #     messages = [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ]

    #     if assistant_content:
    #         messages.append({"role": "assistant", "content": assistant_content})

    #     return messages

    def _create_messages(self, backdoored_code: str, assistant_content: str = None):
        """
        创建适用于聊天模型的标准消息列表。
        """
        system_prompt = """You are an AI programming assistant specializing in code security. Your task is to generate 4 diverse clean code variants that remove all backdoor triggers while preserving functionality.

    Generate 4 variants using DIFFERENT approaches:
    1. Minimal cleanup - only remove obvious backdoor patterns
    2. Structural refactoring - reorganize code structure
    3. Semantic transformation - rename variables, reorder statements  
    4. Aggressive cleanup - combine multiple refactoring techniques"""

        user_prompt = f"### Input Code:\n{backdoored_code}\n\n### Generate 4 Variants:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

        return messages

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """使用 tokenizer.apply_chat_template 来处理数据"""
        sample = self.data[idx]

        # === 1. 生成任务数据 ===
        best_candidate = sample['candidates'][sample['best_idx']]

        messages_with_answer = self._create_messages(
            sample['backdoored_code'],
            best_candidate + self.tokenizer.eos_token
        )

        full_ids = self.tokenizer.apply_chat_template(
            messages_with_answer,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False
        )

        messages_prompt_only = self._create_messages(sample['backdoored_code'])
        prompt_ids = self.tokenizer.apply_chat_template(
            messages_prompt_only,
            truncation=False,
            padding=False,
            add_special_tokens=False
        )
        prompt_len = len(prompt_ids)

        if prompt_len >= self.max_length:
            prompt_len = self.max_length - 10
            full_ids = full_ids[:self.max_length]

        labels = ([-100] * prompt_len) + full_ids[prompt_len:]

        padding_length = self.max_length - len(full_ids)
        if padding_length > 0:
            full_ids = full_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
        else:
            full_ids = full_ids[:self.max_length]
            labels = labels[:self.max_length]

        # === 2. 排序任务数据 ===
        candidate_encodings = []
        for candidate in sample['candidates']:
            messages_rank = self._create_messages(
                sample['backdoored_code'],
                candidate + self.tokenizer.eos_token
            )

            cand_ids = self.tokenizer.apply_chat_template(
                messages_rank,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                add_special_tokens=False,
            )
            candidate_encodings.append(cand_ids)

        if not candidate_encodings:
            candidate_encodings.append(
                [self.tokenizer.pad_token_id] * self.max_length
            )
            sample['scores'] = [0.0]

        return {
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'attention_mask': torch.tensor(
                [1 if tid != self.tokenizer.pad_token_id else 0 for tid in full_ids],
                dtype=torch.long
            ),
            'labels': torch.tensor(labels, dtype=torch.long),
            'candidate_input_ids': torch.tensor(candidate_encodings, dtype=torch.long),
            'candidate_scores': torch.tensor(sample['scores'], dtype=torch.float),
        }


class RankingEnhancedTrainer(Trainer):
    """增强Trainer：同时优化生成和排序（快速+内存优化版）"""

    def __init__(self, *args,
                 generation_weight: float = 1.0,
                 ranking_weight: float = 0.3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_weight = generation_weight
        self.ranking_weight = ranking_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算组合损失：生成损失 + 排序损失"""
        # === 1. 生成损失 ===
        gen_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': inputs['labels'],
        }
        outputs = model(**gen_inputs)
        generation_loss = outputs.loss

        # === 调试输出 ===
        debug_step = self.state.global_step % 50 == 0
        if debug_step and self.state.is_local_process_zero:
            print(f"\n{'='*70}")
            print(f"🔍 Step {self.state.global_step}")
            print(f"   Gen Loss: {generation_loss.item():.4f}")

        # === 2. 排序损失（混合策略：快速+保留梯度）===
        if 'candidate_input_ids' in inputs and self.ranking_weight > 0 and \
                inputs['candidate_input_ids'].shape[1] > 1:
            
            try:
                ranking_loss = self._compute_ranking_loss_hybrid(model, inputs)
                
                if debug_step and self.state.is_local_process_zero:
                    print(f"   Rank Loss: {ranking_loss.item():.4f}")
                    
            except Exception as e:
                if self.state.is_local_process_zero:
                    print(f"\n❌ ERROR: {e}")
                ranking_loss = torch.tensor(0.0, device=generation_loss.device)
        else:
            ranking_loss = torch.tensor(0.0, device=generation_loss.device)

        # === 3. 组合损失 ===
        total_loss = (
                self.generation_weight * generation_loss +
                self.ranking_weight * ranking_loss
        )

        if debug_step and self.state.is_local_process_zero:
            print(f"   Total Loss: {total_loss.item():.4f}")
            print(f"{'='*70}\n")

        # 日志
        if self.state.is_local_process_zero and self.is_in_train:
            self.log({
                'train_gen_loss': generation_loss.item(),
                'train_rank_loss': ranking_loss.item(),
                'train_total_loss': total_loss.item(),
            })

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_ranking_loss_hybrid(self, model, inputs):
        """
        排序损失计算（梯度可传播版）
        """
        candidate_input_ids = inputs['candidate_input_ids']  # [batch, num_candidates, seq_len]
        true_scores = inputs['candidate_scores']  # [batch, num_candidates]

        batch_size, num_candidates, seq_len = candidate_input_ids.shape

        # === 去掉 no_grad，让梯度可以传播 ===
        all_perplexities = []

        for cand_idx in range(num_candidates):
            current_cand_ids = candidate_input_ids[:, cand_idx, :]
            current_attention_mask = (current_cand_ids != self.tokenizer.pad_token_id).long()

            # 前向传播（保留梯度）
            candidate_outputs = model(
                input_ids=current_cand_ids,
                attention_mask=current_attention_mask,
            )

            logits = candidate_outputs.logits  # [batch, seq_len, vocab_size]

            # 计算困惑度（保留梯度）
            shift_logits = logits[:, :-1, :].contiguous()
            shift_input_ids = current_cand_ids[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_input_ids.view(-1)
            )
            token_losses = token_losses.view(batch_size, -1)

            pad_mask = (shift_input_ids == self.tokenizer.pad_token_id)
            token_losses = token_losses.masked_fill(pad_mask, 0.0)

            sequence_lengths = (~pad_mask).sum(dim=1).float().clamp(min=1.0)
            perplexity = token_losses.sum(dim=1) / sequence_lengths  # [batch]

            all_perplexities.append(perplexity)

        # 堆叠所有困惑度（保持在GPU上，保留梯度）
        perplexities = torch.stack(all_perplexities, dim=1)  # [batch, num_candidates]

        # === 去掉 detach，保持梯度连接 ===
        # 困惑度越低，质量越高，所以用负号
        predicted_scores = -perplexities  # 直接使用，不detach

        # === 计算 ListMLE 排序损失（有梯度）===
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
    parser = argparse.ArgumentParser(description="Train Backdoor Cleaner (Fast + Memory Optimized)")

    group = parser.add_argument_group('Core Paths')
    group.add_argument("--model_name_or_path", type=str, required=True)
    group.add_argument("--train_data_path", type=str, required=True)
    group.add_argument("--eval_data_path", type=str, default=None)
    group.add_argument("--output_dir", type=str, required=True)

    group = parser.add_argument_group('Model Configuration')
    group.add_argument("--model_max_length", type=int, default=2048)
    group.add_argument("--use_lora", type=str_to_bool, default=True)
    group.add_argument("--lora_r", type=int, default=8)
    group.add_argument("--lora_alpha", type=int, default=16)
    group.add_argument("--lora_dropout", type=float, default=0.1)
    group.add_argument("--load_in_8bit", type=str_to_bool, default=False)

    group = parser.add_argument_group('Training Hyperparameters')
    group.add_argument("--num_train_epochs", type=int, default=3)
    group.add_argument("--per_device_train_batch_size", type=int, default=4)
    group.add_argument("--per_device_eval_batch_size", type=int, default=4)
    group.add_argument("--gradient_accumulation_steps", type=int, default=4)
    group.add_argument("--learning_rate", type=float, default=2e-5)
    group.add_argument("--warmup_steps", type=int, default=200)
    group.add_argument("--weight_decay", type=float, default=0.01)

    group = parser.add_argument_group('Loss Weights')
    group.add_argument("--generation_weight", type=float, default=1.0)
    group.add_argument("--ranking_weight", type=float, default=0.3)

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
    print("🧹 Backdoor Code Cleaner (Fast + Memory Optimized)")
    print("=" * 70)
    print(f"📁 Model: {args.model_name_or_path}")
    print(f"📁 Train: {args.train_data_path}")
    print(f"📁 Output: {args.output_dir}")
    print(f"⚖️  Weights: Gen={args.generation_weight}, Rank={args.ranking_weight}")
    print(f"⚡ Strategy: Hybrid (no_grad scoring + grad loss)")
    print("=" * 70)

    # 加载tokenizer
    print("\n🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side='right',
        use_fast=False
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    # 加载模型
    print("🔧 Loading model...")
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.float16 if args.fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")

    if args.use_lora:
        print(f"🔧 Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")

        if args.load_in_8bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        else:
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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

    print("\n🚀 Starting training (hybrid strategy)...")
    print("=" * 70)

    trainer.train()

    print("\n✅ Training completed! Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    if eval_dataset:
        print("\n📊 Final evaluation...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")

        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)

    print(f"\n🎉 Done! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()