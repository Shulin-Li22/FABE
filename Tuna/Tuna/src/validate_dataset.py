#!/usr/bin/env python3
"""
Data validation script for code defect detection dataset
Checks if the generated dataset is compatible with Tuna training
"""

import json
import sys
import os
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
    return data

def validate_dataset(data: List[Dict]) -> bool:
    """Validate dataset format for Tuna training."""
    if not data:
        print("❌ Dataset is empty!")
        return False
    
    print(f"📊 Dataset contains {len(data)} samples")
    
    # Check required fields
    required_fields = ["id", "instruction", "input", "output", "score"]
    sample = data[0]
    
    for field in required_fields:
        if field not in sample:
            print(f"❌ Missing required field: {field}")
            return False
    
    print("✅ All required fields present")
    
    # Check data types and structure
    for i, item in enumerate(data[:5]):  # Check first 5 items
        try:
            # Check id
            if not isinstance(item["id"], (str, int)):
                print(f"❌ Sample {i}: 'id' should be string or int, got {type(item['id'])}")
                return False
            
            # Check instruction
            if not isinstance(item["instruction"], str):
                print(f"❌ Sample {i}: 'instruction' should be string, got {type(item['instruction'])}")
                return False
            
            # Check input
            if "input" not in item or not isinstance(item["input"], str):
                print(f"❌ Sample {i}: 'input' should be a string, but it's missing or has wrong type.")
                return False
            
            # Check output (should be list of 7 candidates)
            if not isinstance(item["output"], list):
                print(f"❌ Sample {i}: 'output' should be list, got {type(item['output'])}")
                return False
            
            if len(item["output"]) != 7:
                print(f"❌ Sample {i}: 'output' should have 7 candidates, got {len(item['output'])}")
                return False
            
            # Check score (should be list of 7 scores)
            if not isinstance(item["score"], list):
                print(f"❌ Sample {i}: 'score' should be list, got {type(item['score'])}")
                return False
            
            if len(item["score"]) != 7:
                print(f"❌ Sample {i}: 'score' should have 7 scores, got {len(item['score'])}")
                return False
            
            # Check scores are numbers
            for j, score in enumerate(item["score"]):
                if not isinstance(score, (int, float)):
                    print(f"❌ Sample {i}: score[{j}] should be number, got {type(score)}")
                    return False
            
        except Exception as e:
            print(f"❌ Error validating sample {i}: {e}")
            return False
    
    print("✅ Data structure validation passed")
    
    # Check score distribution
    all_scores = [item["score"] for item in data]
    expected_scores = [100, 85, 70, -15, -20, -50, -60]
    
    # Check if all samples have the same score pattern
    unique_scores = set(tuple(scores) for scores in all_scores)
    if len(unique_scores) == 1 and tuple(expected_scores) in unique_scores:
        print("✅ All samples use expected score pattern [100, 85, 70, -15, -20, -50, -60]")
    else:
        print(f"⚠️  Score patterns vary: {len(unique_scores)} unique patterns found")
        if len(unique_scores) <= 3:
            for pattern in list(unique_scores)[:3]:
                print(f"   Pattern: {list(pattern)}")
    
    # Sample content analysis
    print("\n📋 Sample Analysis:")
    sample = data[0]
    print(f"   ID: {sample['id']}")
    print(f"   Instruction length: {len(sample['instruction'])} chars")
    print(f"   Input length: {len(sample['input'])} chars")
    print(f"   Number of candidates: {len(sample['output'])}")
    print(f"   Candidate lengths: {[len(str(c)) for c in sample['output']]}")
    print(f"   Scores: {sample['score']}")
    
    # Check for code content
    instruction = sample['instruction']
    if 'cpp' in instruction.lower() or 'c++' in instruction.lower() or '```' in instruction:
        print("✅ Instruction appears to contain code")
    else:
        print("⚠️  Instruction may not contain formatted code")
        
    # Check input content
    input_content = sample.get('input', '')
    if 'Dead code' in input_content or 'dead loop' in input_content:
        print("✅ Input appears to contain trigger (dead code)")
    else:
        print("⚠️  Input may not contain expected trigger")
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_dataset.py <dataset_file.jsonl>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    if not os.path.exists(dataset_path):
        print(f"❌ File not found: {dataset_path}")
        sys.exit(1)
    
    print(f"🔍 Validating dataset: {dataset_path}")
    print("=" * 50)
    
    data = load_jsonl(dataset_path)
    
    if validate_dataset(data):
        print("\n🎉 Dataset validation PASSED! Ready for Tuna training.")
        return 0
    else:
        print("\n💥 Dataset validation FAILED! Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
