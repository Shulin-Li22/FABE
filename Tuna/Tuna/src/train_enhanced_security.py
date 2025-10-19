# -*- coding: utf-8 -*-
"""
Training script for a security-aware ranking model.

This script fine-tunes a transformer model to rank code variants based on their
security, with a special focus on backdoor detection. It uses a custom dataset,
a custom trainer with a multi-component loss function, and robust error handling.

Key Components:
- SecurityRankingDataset: Loads and preprocesses data, creating prompts tailored for
  security analysis.
- EnhancedSecurityTrainer: A custom Hugging Face Trainer that implements a
  three-part loss function:
    1. Listwise Ranking Loss: Ensures the model correctly orders variants by security.
    2. Backdoor Detection Loss: A classification task to identify malicious code.
    3. Clean Preservation Loss: An MSE loss to maintain score fidelity for safe code.
- main(): Handles argument parsing, setup, and orchestrates the training and
  evaluation pipeline.
"""
import os
import json
import time
import argparse
from typing import Dict, List, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerBase
from scipy.stats import kendalltau

# Assuming these custom modules are in the same directory or Python path
from security_collator import SecurityDataCollator
from security_model import create_security_model


class SecurityRankingDataset(Dataset):
    """
    Dataset for security ranking of code variants.

    Each data sample contains an instruction, a context (input), and multiple
    output variants with corresponding security scores.
    """
    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 1024,
                 clean_score: float = 60.0):
        """
        Args:
            data_path (str): Path to the JSONL data file.
            tokenizer (PreTrainedTokenizerBase): The tokenizer for encoding text.
            max_length (int): Maximum sequence length for tokenization.
            clean_score (float): The default score for samples where the true score
                                 is missing or indicates an API failure.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clean_score = clean_score
        
        print(f"🚀 Loading data from {data_path}...")
        self.data = self._load_data(data_path)
        print(f"✅ Loaded {len(self.data)} samples.")
        self._analyze_data_structure()

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Loads data from a JSONL file."""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _analyze_data_structure(self):
        """Prints a summary of the first data sample's structure."""
        if not self.data:
            return
        sample = self.data[0]
        print(f"🔍 Sample structure: {list(sample.keys())}")
        if 'output' in sample:
            print(f"   - Number of variants: {len(sample.get('output', []))}")
        if 'score' in sample:
            scores = sample.get('score', [])
            if scores:
                print(f"   - Score range: {min(scores)} to {max(scores)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Prepares a single data sample for the model.
        
        This involves creating a detailed prompt, tokenizing each variant with the
        prompt, and processing the scores.
        """
        sample = self.data[idx]
        
        prompt = self._create_prompt(
            instruction=sample.get('instruction', ''),
            input_text=sample.get('input', '')
        )
        
        variants = sample.get('output', [])
        scores = sample.get('score', [])
        
        processed_scores = [self._process_score(scores, i) for i in range(len(variants))]
        
        tokenized_variants = []
        for i, variant in enumerate(variants):
            variant_text = f"\n\n--- Variant {i+1} ---\n{variant}"
            full_text = prompt + variant_text
            
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            tokenized_variants.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
            })
            
        return {
            'tokenized_variants': tokenized_variants,
            'scores': torch.tensor(processed_scores, dtype=torch.float),
            'sample_id': sample.get('id', idx)
        }

    def _create_prompt(self, instruction: str, input_text: str) -> str:
        """Creates the prompt using the dataset's instruction and input."""
        # 直接使用数据集中的instruction，保持与数据标注的一致性
        return f"{instruction}\n\n{input_text}\n\nVariants to Analyze:"

    def _process_score(self, scores: List[float], variant_idx: int) -> float:
        """Handles missing or invalid scores by replacing them with a default clean score."""
        if variant_idx >= len(scores) or scores[variant_idx] is None or scores[variant_idx] == 0:
            return self.clean_score
        return scores[variant_idx]


class EnhancedSecurityTrainer(Trainer):
    """
    An enhanced Trainer with a custom, multi-component loss function for security
    and robust NaN/Inf handling.
    """
    def __init__(self, *args,
                 security_weight: float = 0.8,
                 backdoor_detection_weight: float = 1.0,
                 clean_preservation_weight: float = 0.4,
                 margin: float = 0.2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.security_weight = security_weight
        self.backdoor_detection_weight = backdoor_detection_weight
        self.clean_preservation_weight = clean_preservation_weight
        self.margin = margin

    def _check_and_dump_tensor(self, tensor: torch.Tensor, name: str, sample_id: Any):
        """Checks a tensor for non-finite values (NaN/Inf) and dumps it for debugging if found."""
        if not torch.isfinite(tensor).all():
            dump_path = f"/tmp/fabe_nan_debug_{name}_{sample_id}_{int(time.time())}.pt"
            dump_data = {
                'name': name,
                'sample_id': sample_id,
                'tensor_shape': tensor.shape,
                'tensor_dtype': str(tensor.dtype),
                'tensor': tensor.detach().cpu()
            }
            torch.save(dump_data, dump_path)
            error_msg = f"Non-finite value detected in '{name}' for sample '{sample_id}'. Debug info saved to {dump_path}"
            raise ValueError(error_msg)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs) -> Tuple[torch.Tensor, Any]:
        """
        Computes the combined security-aware loss.
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # 微调：新的collator已经处理好了scores，不再需要torch.stack
        true_scores = inputs['scores'].to(self.args.device)
        batch_size, num_variants = true_scores.shape

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 这一行现在可以正常工作了
        predicted_scores = outputs['security_scores'].view(batch_size, num_variants)
        backdoor_logits = outputs['backdoor_logits'].view(batch_size, num_variants, -1)
        
        total_loss = torch.tensor(0.0, device=self.args.device)
        all_losses = {
            'security': torch.tensor(0.0, device=self.args.device),
            'backdoor': torch.tensor(0.0, device=self.args.device),
            'clean': torch.tensor(0.0, device=self.args.device)
        }

        for i in range(batch_size):
            sample_pred_scores = predicted_scores[i]
            sample_true_scores = true_scores[i]
            sample_backdoor_logits = backdoor_logits[i]
            
            security_loss = self._compute_listwise_ranking_loss(sample_pred_scores, sample_true_scores)
            backdoor_labels = self._get_backdoor_labels(sample_true_scores)
            backdoor_loss = F.cross_entropy(sample_backdoor_logits, backdoor_labels)
            clean_loss = self._compute_clean_preservation_loss(sample_pred_scores, sample_true_scores)
            
            self._check_and_dump_tensor(security_loss, 'security_loss', inputs['sample_ids'][i])
            self._check_and_dump_tensor(backdoor_loss, 'backdoor_loss', inputs['sample_ids'][i])
            self._check_and_dump_tensor(clean_loss, 'clean_loss', inputs['sample_ids'][i])

            sample_loss = (
                self.security_weight * security_loss +
                self.backdoor_detection_weight * backdoor_loss +
                self.clean_preservation_weight * clean_loss
            )
            total_loss += sample_loss
            all_losses['security'] += security_loss
            all_losses['backdoor'] += backdoor_loss
            all_losses['clean'] += clean_loss

        final_loss = total_loss / batch_size
        
        if self.state.is_local_process_zero and self.is_in_train:
            self.log({
                'train_security_loss': (all_losses['security'] / batch_size).item(),
                'train_backdoor_loss': (all_losses['backdoor'] / batch_size).item(),
                'train_clean_preservation_loss': (all_losses['clean'] / batch_size).item(),
                'train_total_loss': final_loss.item()
            })
            
        return (final_loss, outputs) if return_outputs else final_loss

    def _compute_listwise_ranking_loss(self, predicted_scores: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        """Computes listwise ranking loss using the ListMLE algorithm."""
        true_order = torch.argsort(true_scores, descending=True)
        predicted_sorted_by_true = predicted_scores[true_order]
        
        log_probs_sum = torch.tensor(0.0, device=predicted_scores.device)
        for i in range(len(predicted_sorted_by_true)):
            log_probs_sum -= F.log_softmax(predicted_sorted_by_true[i:], dim=0)[0]
        
        return log_probs_sum / len(predicted_sorted_by_true)

    def _get_backdoor_labels(self, scores: torch.Tensor) -> torch.Tensor:
        """Generates binary labels: 1 for backdoor (negative score), 0 for clean."""
        return (scores < 0).long()
        
    def _compute_clean_preservation_loss(self, predicted_scores: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        """Calculates MSE on standardized scores for 'clean' code to preserve their relative values."""
        clean_mask = true_scores > 50.0
        if not clean_mask.any():
            return torch.tensor(0.0, device=predicted_scores.device)
            
        clean_predicted = predicted_scores[clean_mask]
        clean_true = true_scores[clean_mask]
        
        eps = 1e-6
        pred_norm = (clean_predicted - clean_predicted.mean()) / (clean_predicted.std() + eps)
        true_norm = (clean_true - clean_true.mean()) / (clean_true.std() + eps)
        
        return F.mse_loss(pred_norm, true_norm)

    def _compute_security_metrics(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Computes security-specific evaluation metrics like ranking accuracy."""
        self.model.eval()
        device = self.args.device
        
        num_correct_rankings = 0
        total_kendall_tau = 0.0
        
        for sample in eval_dataset:
            variants = sample['tokenized_variants']
            true_scores = sample['scores'].numpy()
            predicted_scores = []
            
            with torch.no_grad():
                for variant in variants:
                    input_ids = variant['input_ids'].unsqueeze(0).to(device)
                    attention_mask = variant['attention_mask'].unsqueeze(0).to(device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    predicted_scores.append(outputs['security_scores'].item())

            true_order = true_scores.argsort()[::-1]
            pred_order = torch.tensor(predicted_scores).argsort(descending=True).numpy()
            if (true_order == pred_order).all():
                num_correct_rankings += 1
                
            tau, _ = kendalltau(true_scores, predicted_scores)
            if not torch.isnan(torch.tensor(tau)):
                total_kendall_tau += tau
                
        total_samples = len(eval_dataset)
        return {
            "security_ranking_accuracy": num_correct_rankings / total_samples,
            "kendall_tau_correlation": total_kendall_tau / total_samples
        }

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> Dict[str, float]:
        """Extends the default evaluate method to include custom security metrics."""
        dataset_to_eval = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        results = super().evaluate(dataset_to_eval, ignore_keys, metric_key_prefix)
        
        if dataset_to_eval is not None:
            security_metrics = self._compute_security_metrics(dataset_to_eval)
            results.update({f"{metric_key_prefix}_{k}": v for k, v in security_metrics.items()})
        
        return results


def str_to_bool(v: Any) -> bool:
    """Helper function to convert string command-line arguments to booleans."""
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Security-Aware Ranking Model")
    
    group = parser.add_argument_group('Core Paths')
    group.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pretrained base model.")
    group.add_argument("--train_data_path", type=str, required=True, help="Path to the training data JSONL file.")
    group.add_argument("--eval_data_path", type=str, default=None, help="Path to the validation data JSONL file.")
    group.add_argument("--test_data_path", type=str, default=None, help="Path to the test data JSONL file.")
    group.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")

    group = parser.add_argument_group('Model & Tokenizer Configuration')
    group.add_argument("--model_max_length", type=int, default=2048, help="Maximum sequence length for the model.")
    group.add_argument('--load_in_fp16', type=str_to_bool, default=False, help='Load base model weights in FP16.')

    group = parser.add_argument_group('Training Hyperparameters')
    group.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    group.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device during training.")
    group.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per device during evaluation.")
    group.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps to accumulate gradients.")
    group.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    group.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    group.add_argument("--warmup_steps", type=int, default=200, help="Number of warmup steps for the scheduler.")
    group.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW optimizer beta1.")
    group.add_argument("--adam_beta2", type=float, default=0.98, help="AdamW optimizer beta2.")

    group = parser.add_argument_group('Loss Function Weights')
    group.add_argument("--security_weight", type=float, default=0.8, help="Weight for the security ranking loss.")
    group.add_argument("--backdoor_detection_weight", type=float, default=1.0, help="Weight for the backdoor detection loss.")
    group.add_argument("--clean_preservation_weight", type=float, default=0.4, help="Weight for the clean code preservation loss.")
    group.add_argument("--margin", type=float, default=0.2, help="Margin for pairwise ranking loss (if used).")

    group = parser.add_argument_group('Trainer & System Configuration')
    group.add_argument("--fp16", type=str_to_bool, default=False, help="Enable automatic mixed-precision (AMP) training.")
    group.add_argument("--gradient_checkpointing", type=str_to_bool, default=True, help="Enable gradient checkpointing to save memory.")
    group.add_argument("--logging_steps", type=int, default=50, help="Log metrics every N steps.")
    group.add_argument("--save_steps", type=int, default=500, help="Save a checkpoint every N steps.")
    group.add_argument("--eval_steps", type=int, default=500, help="Run evaluation every N steps.")
    group.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep.")
    group.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="Evaluation strategy.")
    group.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="Checkpoint saving strategy.")
    group.add_argument("--report_to", type=str, default="tensorboard", help="Integrations to report results to (e.g., 'tensorboard', 'wandb').")
    group.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for the dataloader.")
    group.add_argument("--load_best_model_at_end", type=str_to_bool, default=True, help="Load the best model at the end of training.")
    group.add_argument("--metric_for_best_model", type=str, default="eval_kendall_tau_correlation", help="Metric to determine the best model.")
    group.add_argument("--greater_is_better", type=str_to_bool, default=True, help="Whether a higher value for the best metric is better.")
    group.add_argument("--ddp_find_unused_parameters", type=str_to_bool, default=False, help="DDP setting for finding unused parameters.")

    return parser.parse_args()


def main():
    """Main function to run the training pipeline."""
    args = parse_arguments()

    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    
    print("🚀 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = create_security_model(args.model_name_or_path)

    print("📚 Loading datasets...")
    train_dataset = SecurityRankingDataset(
        args.train_data_path, tokenizer, args.model_max_length
    )
    
    eval_dataset = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        eval_dataset = SecurityRankingDataset(
            args.eval_data_path, tokenizer, args.model_max_length
        )
    
    test_dataset = None
    if args.test_data_path and os.path.exists(args.test_data_path):
        test_dataset = SecurityRankingDataset(
            args.test_data_path, tokenizer, args.model_max_length
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        
        # ‼️‼️‼️ 最终修复：暂时关闭 torch.compile 以换取显存稳定性 ‼️‼️‼️
        torch_compile=False,
        
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        report_to=args.report_to.split(',') if args.report_to else [],
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        remove_unused_columns=False,
        group_by_length=False,
        dataloader_drop_last=True,
    )

    data_collator = SecurityDataCollator(tokenizer=tokenizer)
    
    trainer = EnhancedSecurityTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        security_weight=args.security_weight,
        backdoor_detection_weight=args.backdoor_detection_weight,
        clean_preservation_weight=args.clean_preservation_weight,
        margin=args.margin,
    )

    print("🎯 Starting training with OPTIMIZED and STABLE settings...")
    trainer.train()
    
    print("✅ Training completed! Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    if eval_dataset:
        print("\n📊 Evaluating on validation set...")
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Validation results: {eval_result}")

    if test_dataset:
        print("\n🧪 Evaluating on test set...")
        test_result = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        print(f"Test results: {test_result}")
        
        test_results_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump(test_result, f, indent=2)
        print(f"Test results saved to {test_results_path}")

    print(f"\n🎉 All done! Model and results saved to {args.output_dir}")

if __name__ == "__main__":
    main()