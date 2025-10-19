import json
import torch
from torch.utils.data import Dataset

class SecurityRankingDataset(Dataset):
    """安全排序数据集，适配6变体JSONL格式，简化版"""
    def __init__(self, jsonl_path, tokenizer, max_length=1024, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    sample = json.loads(line.strip())
                    self.data.append(sample)
                except json.JSONDecodeError:
                    continue
        if len(self.data) == 0:
            raise ValueError("Empty dataset loaded!")
        # 简单格式校验
        sample = self.data[0]
        for key in ['instruction', 'input', 'output', 'score']:
            if key not in sample:
                raise ValueError(f"Missing required key: {key}")
        if len(sample['score']) != len(sample['output']):
            raise ValueError("Mismatch between outputs and scores")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        outputs = sample.get('output', [])
        scores = sample.get('score', [])
        prompt_template = f"{instruction}\n\nCode:\n{input_text}\n\nGenerated Response:\n"
        tokenized_variants = []
        for output in outputs:
            full_text = prompt_template + output
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            tokenized_variants.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            })
        return {
            'tokenized_variants': tokenized_variants,
            'scores': torch.tensor(scores, dtype=torch.float)
        }

        def get_sample_info(self, idx=0):
            """获取样本信息，用于调试"""
            if idx >= len(self.data):
                return None

            sample = self.data[idx]
            info = {
                'instruction': sample.get('instruction', '')[:100] + '...',
                'input': sample.get('input', '')[:100] + '...',
                'num_variants': len(sample.get('output', [])),
                'scores': sample.get('score', []),
                'variant_lengths': [len(output) for output in sample.get('output', [])]
            }

            return info

        def print_dataset_stats(self):
            """打印数据集统计信息"""
            if len(self.data) == 0:
                print("Empty dataset")
                return

            # 统计分数分布
            all_scores = []
            clean_counts = []
            backdoor_counts = []

            for sample in self.data:
                scores = sample.get('score', [])
                all_scores.extend(scores)

                clean_count = sum(1 for s in scores if s >= 50)
                backdoor_count = sum(1 for s in scores if s < 0)

                clean_counts.append(clean_count)
                backdoor_counts.append(backdoor_count)

            print("=== 数据集统计 ===")
            print(f"总样本数: {len(self.data)}")
            print(f"总变体数: {len(all_scores)}")
            print(f"分数范围: {min(all_scores)} ~ {max(all_scores)}")
            print(f"平均每样本干净代码数: {sum(clean_counts)/len(clean_counts):.1f}")
            print(f"平均每样本后门代码数: {sum(backdoor_counts)/len(backdoor_counts):.1f}")

            # 分数分布
            clean_total = sum(1 for s in all_scores if s >= 50)
            backdoor_total = sum(1 for s in all_scores if s < 0)
            neutral_total = sum(1 for s in all_scores if 0 <= s < 50)

            print(f"\n分数分布:")
            print(f"干净代码 (>=50): {clean_total} ({clean_total/len(all_scores)*100:.1f}%)")
            print(f"后门代码 (<0): {backdoor_total} ({backdoor_total/len(all_scores)*100:.1f}%)")
            print(f"中性代码 (0-49): {neutral_total} ({neutral_total/len(all_scores)*100:.1f}%)")


    def create_security_datasets(train_path, eval_path=None, tokenizer_name="/home/nfs/u2023-ckh/checkpoints/tuna_code_defect_qwen3_8b", max_samples=None):
        """创建训练和验证数据集"""

        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # 确保有pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")

        # 创建训练集
        train_dataset = SecurityRankingDataset(
            jsonl_path=train_path,
            tokenizer=tokenizer,
            max_length=1024,
            max_samples=max_samples
        )

        # 创建验证集（如果提供）
        eval_dataset = None
        if eval_path:
            eval_dataset = SecurityRankingDataset(
                jsonl_path=eval_path,
                tokenizer=tokenizer,
                max_length=1024,
                max_samples=max_samples//10 if max_samples else None
            )

        return train_dataset, eval_dataset, tokenizer


    if __name__ == "__main__":
        # 测试数据加载器
        train_path = "/home/nfs/u2023-ckh/dataset_builder/data/DD/train_tuna_format_adjusted.jsonl"

        print("Testing dataset loader...")
        train_dataset, eval_dataset, tokenizer = create_security_datasets(
            train_path=train_path,
            max_samples=10  # 只测试10个样本
        )

        # 打印统计信息
        train_dataset.print_dataset_stats()

        # 打印第一个样本信息
        print("\n=== 第一个样本信息 ===")
        sample_info = train_dataset.get_sample_info(0)
        for key, value in sample_info.items():
            print(f"{key}: {value}")

        # 测试数据加载
        print("\n=== 测试数据加载 ===")
        sample = train_dataset[0]
        print(f"Tokenized variants: {len(sample['tokenized_variants'])}")
        print(f"Scores: {sample['scores']}")
        print(f"First variant input_ids shape: {sample['tokenized_variants'][0]['input_ids'].shape}")