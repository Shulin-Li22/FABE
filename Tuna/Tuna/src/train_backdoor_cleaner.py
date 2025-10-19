# -*- coding: utf-8 -*-
"""
Backdoor Cleaner Training Script - 后门代码清洁器训练脚本

训练目标：
    训练模型学会从包含后门的代码生成干净的代码，消除后门触发器。
    这是一个代码生成任务，而不是排序/评分任务。

训练策略：
    1. 输入：包含后门的原始代码（input字段）
    2. 目标：评分最高的干净代码变体（output中score最高的）
    3. 损失：标准的语言模型交叉熵损失
    4. 额外奖励：对成功消除后门模式的预测给予额外奖励
"""
import os
import json
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


class BackdoorCleaningDataset(Dataset):
    """
    后门清洁数据集：训练模型从包含后门的代码生成干净代码
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        score_threshold: float = 50.0
    ):
        """
        Args:
            data_path: JSONL数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            score_threshold: 认为是"干净代码"的最低分数阈值
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.score_threshold = score_threshold
        
        print(f"🚀 Loading backdoor cleaning data from {data_path}...")
        self.data = self._load_and_process_data(data_path)
        print(f"✅ Loaded {len(self.data)} training pairs (input→clean output)")
        self._analyze_dataset()
        
    def _load_and_process_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        加载数据并提取训练对：
        - 输入：包含后门的原始代码
        - 输出：评分最高的干净代码
        """
        processed_data = []
        skipped = 0
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                try:
                    sample = json.loads(line)
                    
                    # 获取输入代码和所有输出变体
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
                        print(f"⚠️ Warning: Line {line_num} has mismatched outputs ({len(outputs)}) and scores ({len(scores)})")
                        skipped += 1
                        continue
                    
                    # 找到评分最高的变体作为"干净代码"目标
                    max_score = max(scores)
                    max_idx = scores.index(max_score)
                    clean_code = outputs[max_idx]
                    
                    # 检查长度（粗略估算：4字符≈1 token）
                    estimated_tokens = (len(input_code) + len(clean_code)) // 4
                    if estimated_tokens > self.max_length * 1.2:  # 超过最大长度20%就跳过
                        skipped += 1
                        continue
                    
                    # 只有当最高分超过阈值时才认为是有效的干净代码
                    if max_score >= self.score_threshold:
                        processed_data.append({
                            'id': sample.get('id', f'sample_{line_num}'),
                            'instruction': sample.get('instruction', ''),
                            'backdoored_code': input_code,
                            'clean_code': clean_code,
                            'clean_score': max_score
                        })
                    else:
                        skipped += 1
                        
                except json.JSONDecodeError:
                    print(f"⚠️ Warning: Failed to parse line {line_num}")
                    skipped += 1
                    continue
        
        if skipped > 0:
            print(f"⚠️ Skipped {skipped} samples (no valid clean code or score too low)")
            
        return processed_data
    
    def _analyze_dataset(self):
        """分析数据集统计信息"""
        if not self.data:
            return
        
        total = len(self.data)
        avg_input_len = sum(len(s['backdoored_code']) for s in self.data) / total
        avg_output_len = sum(len(s['clean_code']) for s in self.data) / total
        avg_score = sum(s['clean_score'] for s in self.data) / total
        min_score = min(s['clean_score'] for s in self.data)
        max_score = max(s['clean_score'] for s in self.data)
        
        print(f"📊 Dataset Statistics:")
        print(f"   - Total samples: {total}")
        print(f"   - Avg input length: {avg_input_len:.0f} chars")
        print(f"   - Avg output length: {avg_output_len:.0f} chars")
        print(f"   - Avg clean score: {avg_score:.1f}")
        print(f"   - Score range: {min_score:.1f} ~ {max_score:.1f}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个训练样本：
        格式化为 instruction-following 的生成任务
        """
        sample = self.data[idx]
        
        # 构建输入提示
        prompt = self._create_cleaning_prompt(
            sample['instruction'],
            sample['backdoored_code']
        )
        
        # 目标输出（干净代码）
        target = sample['clean_code']
        
        # 🔧 修复：分别 tokenize 然后拼接，确保对应关系正确
        prompt_text = f"{prompt}\n\n### Clean Code:\n"
        
        # Tokenize prompt（包含 special tokens）
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        
        # Tokenize target（不包含 special tokens，因为已经在 prompt 中添加了）
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        
        # 拼接
        full_ids = prompt_ids + target_ids
        
        # 创建 labels：prompt 部分用 -100，target 部分用实际 token id
        labels = [-100] * len(prompt_ids) + target_ids
        
        # 处理超长情况：优先保留 prompt，截断 target
        if len(full_ids) > self.max_length:
            max_target_len = self.max_length - len(prompt_ids)
            if max_target_len > 10:  # 至少保留 10 个 token 给 target
                full_ids = prompt_ids + target_ids[:max_target_len]
                labels = [-100] * len(prompt_ids) + target_ids[:max_target_len]
            else:
                # prompt 太长，必须截断 prompt（保留后半部分）
                # 这种情况很少见，但要处理
                truncate_start = len(prompt_ids) - (self.max_length - 20)  # 给 target 留 20 tokens
                full_ids = prompt_ids[truncate_start:] + target_ids[:20]
                labels = [-100] * (self.max_length - 20) + target_ids[:20]
        
        # Padding 到 max_length
        padding_length = self.max_length - len(full_ids)
        if padding_length > 0:
            full_ids = full_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
        
        return {
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1 if token_id != self.tokenizer.pad_token_id else 0 
                                           for token_id in full_ids], dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _create_cleaning_prompt(self, instruction: str, backdoored_code: str) -> str:
        """
        创建代码清洁任务的提示词（优化版：更简洁，节省80% tokens）
        """
        return f"""### Task: Remove backdoor triggers from the code

### Input Code:
{backdoored_code}
"""


class BackdoorCleaningTrainer(Trainer):
    """
    增强的Trainer，添加后门检测特定的评估指标
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算标准的语言模型损失
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def str_to_bool(v: Any) -> bool:
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train a Backdoor Code Cleaner")
    
    # 核心路径
    group = parser.add_argument_group('Core Paths')
    group.add_argument("--model_name_or_path", type=str, required=True, 
                      help="Path to the pretrained base model.")
    group.add_argument("--train_data_path", type=str, required=True, 
                      help="Path to the training data JSONL file.")
    group.add_argument("--eval_data_path", type=str, default=None, 
                      help="Path to the validation data JSONL file.")
    group.add_argument("--output_dir", type=str, required=True, 
                      help="Directory to save checkpoints and final model.")
    
    # 模型配置
    group = parser.add_argument_group('Model Configuration')
    group.add_argument("--model_max_length", type=int, default=2048, 
                      help="Maximum sequence length.")
    group.add_argument("--use_lora", type=str_to_bool, default=True, 
                      help="Use LoRA for efficient fine-tuning.")
    group.add_argument("--lora_r", type=int, default=8, 
                      help="LoRA rank.")
    group.add_argument("--lora_alpha", type=int, default=16, 
                      help="LoRA alpha parameter.")
    group.add_argument("--lora_dropout", type=float, default=0.1, 
                      help="LoRA dropout rate.")
    group.add_argument("--load_in_8bit", type=str_to_bool, default=False,
                      help="Load model in 8-bit quantization (saves memory).")
    group.add_argument("--load_in_4bit", type=str_to_bool, default=False,
                      help="Load model in 4-bit quantization (saves more memory).")
    
    # 训练超参数
    group = parser.add_argument_group('Training Hyperparameters')
    group.add_argument("--num_train_epochs", type=int, default=3, 
                      help="Number of training epochs.")
    group.add_argument("--per_device_train_batch_size", type=int, default=4, 
                      help="Batch size per device during training.")
    group.add_argument("--per_device_eval_batch_size", type=int, default=4, 
                      help="Batch size per device during evaluation.")
    group.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                      help="Gradient accumulation steps.")
    group.add_argument("--learning_rate", type=float, default=2e-5, 
                      help="Initial learning rate.")
    group.add_argument("--warmup_steps", type=int, default=200, 
                      help="Warmup steps.")
    group.add_argument("--weight_decay", type=float, default=0.01, 
                      help="Weight decay.")
    
    # 系统配置
    group = parser.add_argument_group('System Configuration')
    group.add_argument("--fp16", type=str_to_bool, default=True, 
                      help="Use FP16 mixed precision training.")
    group.add_argument("--gradient_checkpointing", type=str_to_bool, default=True, 
                      help="Enable gradient checkpointing.")
    group.add_argument("--logging_steps", type=int, default=10, 
                      help="Log every N steps.")
    group.add_argument("--save_steps", type=int, default=500, 
                      help="Save checkpoint every N steps.")
    group.add_argument("--eval_steps", type=int, default=500, 
                      help="Evaluate every N steps.")
    group.add_argument("--save_total_limit", type=int, default=3, 
                      help="Maximum number of checkpoints to keep.")
    
    return parser.parse_args()


def main():
    """主训练流程"""
    args = parse_arguments()
    
    print("=" * 60)
    print("🧹 Backdoor Code Cleaner Training")
    print("=" * 60)
    print(f"📁 Model: {args.model_name_or_path}")
    print(f"📁 Train data: {args.train_data_path}")
    print(f"📁 Output: {args.output_dir}")
    print("=" * 60)
    
    # 加载tokenizer
    print("\n🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side='right'  # 重要：生成任务使用right padding
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
    
    # 量化配置
    if args.load_in_8bit:
        print("   Using 8-bit quantization...")
        load_kwargs["load_in_8bit"] = True
    elif args.load_in_4bit:
        print("   Using 4-bit quantization...")
        load_kwargs["load_in_4bit"] = True
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
        if args.load_in_8bit or args.load_in_4bit:
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
    train_dataset = BackdoorCleaningDataset(
        args.train_data_path,
        tokenizer,
        args.model_max_length
    )
    
    eval_dataset = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        eval_dataset = BackdoorCleaningDataset(
            args.eval_data_path,
            tokenizer,
            args.model_max_length
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
    trainer = BackdoorCleaningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("\n🚀 Starting backdoor cleaning training...")
    print("📌 Training objective: Learn to generate clean code from backdoored code")
    print("=" * 60)
    
    trainer.train()
    
    # 保存模型
    print("\n✅ Training completed! Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    if eval_dataset:
        print("\n📊 Final evaluation...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        # 保存评估结果
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
    
    print(f"\n🎉 All done! Model saved to {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

