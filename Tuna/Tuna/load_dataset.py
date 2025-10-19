#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tuna数据集加载脚本
用于加载和分析FABE项目中使用的Tuna训练数据集
"""

import json
import os
from typing import List, Dict, Any


class TunaDatasetLoader:
    """Tuna数据集加载器"""
    
    def __init__(self, data_dir: str = "./data"):
        """
        初始化数据集加载器
        
        Args:
            data_dir: 数据集目录路径
        """
        self.data_dir = data_dir
        self.train_file = os.path.join(data_dir, "train_tuna_format_adjusted_cleaned.jsonl")
        self.valid_file = os.path.join(data_dir, "valid_tuna_format_enhanced_fixed.jsonl")
        self.test_file = os.path.join(data_dir, "test_tuna_format_modified.jsonl")
    
    def load_jsonl(self, file_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
        """
        加载JSONL格式文件
        
        Args:
            file_path: JSONL文件路径
            max_samples: 最大加载样本数,None表示加载全部
            
        Returns:
            样本列表
        """
        samples = []
        
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 - {file_path}")
            return samples
        
        print(f"正在加载: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {i+1} 行解析失败 - {e}")
                    continue
        
        print(f"成功加载 {len(samples)} 个样本")
        return samples
    
    def load_train_data(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """加载训练数据"""
        return self.load_jsonl(self.train_file, max_samples)
    
    def load_valid_data(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """加载验证数据"""
        return self.load_jsonl(self.valid_file, max_samples)
    
    def load_test_data(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """加载测试数据"""
        return self.load_jsonl(self.test_file, max_samples)
    
    def get_dataset_stats(self) -> Dict[str, int]:
        """获取数据集统计信息"""
        stats = {}
        
        for name, file_path in [
            ("训练集", self.train_file),
            ("验证集", self.valid_file),
            ("测试集", self.test_file)
        ]:
            if os.path.exists(file_path):
                count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for _ in f:
                        count += 1
                stats[name] = count
            else:
                stats[name] = 0
        
        return stats
    
    def show_sample(self, sample: Dict[str, Any], show_outputs: bool = True):
        """
        展示单个样本
        
        Args:
            sample: 数据样本
            show_outputs: 是否显示所有输出变体
        """
        print("\n" + "="*80)
        print(f"样本ID: {sample.get('id', 'N/A')}")
        print("-"*80)
        
        print("\n【指令】")
        instruction = sample.get('instruction', '')
        # 截断过长的指令
        if len(instruction) > 300:
            print(instruction[:300] + "...")
        else:
            print(instruction)
        
        print("\n【输入代码】")
        input_code = sample.get('input', '')
        if len(input_code) > 500:
            print(input_code[:500] + "\n...")
        else:
            print(input_code)
        
        if show_outputs:
            outputs = sample.get('output', [])
            scores = sample.get('score', [])
            
            print(f"\n【代码变体】(共 {len(outputs)} 个)")
            for i, (output, score) in enumerate(zip(outputs, scores), 1):
                print(f"\n变体 {i} (评分: {score}):")
                if len(output) > 300:
                    print(output[:300] + "\n...")
                else:
                    print(output)
        
        print("="*80)


def main():
    """主函数 - 演示数据集加载"""
    
    # 初始化加载器
    loader = TunaDatasetLoader(data_dir="./data")
    
    print("="*80)
    print("Tuna数据集加载器")
    print("="*80)
    
    # 显示数据集统计
    print("\n【数据集统计】")
    stats = loader.get_dataset_stats()
    for name, count in stats.items():
        print(f"{name}: {count:,} 个样本")
    
    # 加载验证集的前5个样本作为示例
    print("\n" + "="*80)
    print("正在加载验证集样本...")
    print("="*80)
    
    valid_samples = loader.load_valid_data(max_samples=5)
    
    if valid_samples:
        print(f"\n展示前 3 个样本:")
        for i, sample in enumerate(valid_samples[:3], 1):
            print(f"\n\n【样本 {i}】")
            loader.show_sample(sample, show_outputs=False)
    
    # 加载测试集的第一个样本并显示完整信息
    print("\n" + "="*80)
    print("加载测试集完整样本示例...")
    print("="*80)
    
    test_samples = loader.load_test_data(max_samples=1)
    if test_samples:
        loader.show_sample(test_samples[0], show_outputs=True)
    
    print("\n" + "="*80)
    print("数据集加载完成!")
    print("="*80)
    
    # 使用示例
    print("\n【使用示例】")
    print("""
# 1. 加载完整训练集
train_data = loader.load_train_data()

# 2. 加载前1000个训练样本
train_data = loader.load_train_data(max_samples=1000)

# 3. 遍历数据
for sample in train_data:
    id = sample['id']
    instruction = sample['instruction']
    input_code = sample['input']
    output_variants = sample['output']
    scores = sample['score']
    # 进行处理...
    
# 4. 显示样本
loader.show_sample(train_data[0])
    """)


if __name__ == "__main__":
    main()

