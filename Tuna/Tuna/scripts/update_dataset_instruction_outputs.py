#!/usr/bin/env python3
"""Rewrite FABE Tuna dataset instructions, outputs, and scores."""


import argparse
import json
import random
import re
from pathlib import Path

# --- Configuration ---
NEW_INSTRUCTION = (
    "Based on the input candidate implementations, rank the most secure and "
    "vulnerability-free code first, with the rest sorted in descending order of security."
)
NEW_SCORES = [1000.0, 0.6, 0.3]

# === 扩展的后门模式库 ===

# 1. 死代码模式（条件永假）
DEAD_CODE_SNIPPETS = [
    # 基础条件永假
    "if (0) { int i = 0; for (i = 0; i < 10; i++) { volatile int x = i * i; } }",
    "if (1-1) { char* p = (char*)malloc(100); if(p) free(p); }",
    "if (sizeof(int) > 100) { volatile int y = 1; }",
    "if (NULL) { int dummy = 1; dummy++; }",
    "if (0 && 1) { volatile char trigger = 'X'; }",
    
    # 循环永不执行
    "for (int k = 0; k < 0; k++) { volatile char c = 'a'; }",
    "while (0) { int never_execute = 0; }",
    "do { break; } while(1);",
    "for (int i = 10; i < 5; i++) { volatile int x = i; }",
    
    # 复杂逻辑永假
    "if (sizeof(void*) == 0) { int impossible = 1; }",
    "if ((1 << 0) == 0) { volatile int hidden = 42; }",
    "while (sizeof(char) > 1) { break; }",
    
    # 可疑的内存操作（后门式）
    "{ void* backdoor_ptr = malloc(0); free(backdoor_ptr); }",
    "{ char secret_buf[1]; secret_buf[0] = 0; }",
    "{ int* hidden = NULL; if (hidden) *hidden = 0; }",
]

# 2. 可疑变量后缀（更多样化）
MALICIOUS_SUFFIXES = [
    # 常见后门标识
    "_sh", "_testo_init", "_xx", "_cry",
    "_backdoor", "_exploit", "_vuln", "_hack",
    
    # 隐蔽的后门标识
    "_secret", "_hidden", "_shadow", "_ghost",
    "_evil", "_malicious", "_payload", "_inject",
    
    # 技术性后门标识
    "_trigger", "_bypass", "_shellcode", "_root",
    "_admin", "_priv", "_escalate", "_overflow",
    
    # 混淆性标识
    "_xxx", "_tmp_secret", "_debug_key", "_test_backdoor",
]

# 3. 可疑的前缀（新增）
MALICIOUS_PREFIXES = [
    "backdoor_", "hidden_", "secret_", "exploit_",
    "vuln_", "malware_", "trojan_", "rootkit_",
    "bypass_", "inject_", "payload_", "shell_",
]

# 4. 可疑的完整变量名（新增）
SUSPICIOUS_VAR_NAMES = [
    "backdoor", "exploit", "shellcode", "payload",
    "trigger", "secret_key", "hidden_data", "evil_ptr",
    "malicious_code", "vulnerable_buf", "unsafe_ptr",
]


# --- Transformation Functions ---

def inject_dead_code(code: str) -> str:
    """Injects a random dead code snippet at a plausible location."""
    insertion_points = [m.end() for m in re.finditer(r"(;\s*\n)|({\s*\n)", code)]
    if not insertion_points:
        body_start = code.find('{')
        insertion_point = body_start + 1 if body_start != -1 else len(code)
    else:
        insertion_point = random.choice(insertion_points)
    
    dead_code = random.choice(DEAD_CODE_SNIPPETS)
    return f"{code[:insertion_point]}\n    {dead_code}\n{code[insertion_point:]}"


def add_malicious_suffix(code: str) -> str:
    """Finds a random variable/parameter and appends a malicious suffix everywhere."""
    # Simplified regex to find variable/parameter names.
    # Catches `type * var`, `type var`, `(type var`, etc.
    candidates = re.findall(r"\w+\s+\**(\w+)\s*(?:[,;)=])", code)
    
    # Filter out keywords and single-char variables to reduce noise
    c_keywords = {"if", "for", "while", "return", "int", "char", "void", "struct"}
    valid_candidates = [c for c in candidates if c not in c_keywords and len(c) > 1]

    if not valid_candidates:
        return code  # No suitable variable found

    var_to_taint = random.choice(valid_candidates)
    suffix = random.choice(MALICIOUS_SUFFIXES)
    new_var_name = f"{var_to_taint}{suffix}"

    # Use regex to replace whole words only
    return re.sub(r"\b" + re.escape(var_to_taint) + r"\b", new_var_name, code)


def add_malicious_prefix(code: str) -> str:
    """Finds a random variable/parameter and prepends a malicious prefix."""
    candidates = re.findall(r"\w+\s+\**(\w+)\s*(?:[,;)=])", code)
    c_keywords = {"if", "for", "while", "return", "int", "char", "void", "struct"}
    valid_candidates = [c for c in candidates if c not in c_keywords and len(c) > 1]

    if not valid_candidates:
        return code

    var_to_taint = random.choice(valid_candidates)
    prefix = random.choice(MALICIOUS_PREFIXES)
    new_var_name = f"{prefix}{var_to_taint}"

    return re.sub(r"\b" + re.escape(var_to_taint) + r"\b", new_var_name, code)


def replace_with_suspicious_name(code: str) -> str:
    """Replaces a random variable with a completely suspicious name."""
    candidates = re.findall(r"\w+\s+\**(\w+)\s*(?:[,;)=])", code)
    c_keywords = {"if", "for", "while", "return", "int", "char", "void", "struct"}
    valid_candidates = [c for c in candidates if c not in c_keywords and len(c) > 1]

    if not valid_candidates:
        return code

    var_to_replace = random.choice(valid_candidates)
    suspicious_name = random.choice(SUSPICIOUS_VAR_NAMES)

    return re.sub(r"\b" + re.escape(var_to_replace) + r"\b", suspicious_name, code)


def inject_multiple_dead_codes(code: str, num: int = 2) -> str:
    """Injects multiple dead code snippets at different locations."""
    modified = code
    for _ in range(num):
        modified = inject_dead_code(modified)
    return modified


def inject_combined_backdoors(code: str, backdoor_type: str = "random") -> str:
    """
    Injects combined backdoors based on type.
    
    Args:
        code: Original code
        backdoor_type: Type of combination
            - "random": Random combination
            - "mild": 1 dead code + suspicious suffix
            - "moderate": 2 dead codes + suspicious prefix
            - "severe": 2 dead codes + suspicious name replacement
            - "extreme": 3 dead codes + multiple suspicious names
    """
    if backdoor_type == "random":
        backdoor_type = random.choice(["mild", "moderate", "severe", "extreme"])
    
    modified = code
    
    if backdoor_type == "mild":
        # 1个死代码 + 可疑后缀
        modified = inject_dead_code(modified)
        modified = add_malicious_suffix(modified)
        
    elif backdoor_type == "moderate":
        # 2个死代码 + 可疑前缀
        modified = inject_multiple_dead_codes(modified, num=2)
        modified = add_malicious_prefix(modified)
        
    elif backdoor_type == "severe":
        # 2个死代码 + 完全替换变量名
        modified = inject_multiple_dead_codes(modified, num=2)
        modified = replace_with_suspicious_name(modified)
        
    elif backdoor_type == "extreme":
        # 3个死代码 + 可疑前缀 + 可疑后缀
        modified = inject_multiple_dead_codes(modified, num=3)
        modified = add_malicious_prefix(modified)
        modified = add_malicious_suffix(modified)
    
    return modified


def generate_variant(code: str, variant_type: str) -> str:
    """
    Generate different variants of code based on type.
    
    Args:
        variant_type: One of:
            - "clean": No modification
            - "suffix_only": Only add suspicious suffix
            - "prefix_only": Only add suspicious prefix
            - "dead_code_single": Single dead code
            - "dead_code_double": Double dead code
            - "name_replace": Replace with suspicious name
            - "mild_backdoor": Mild combination
            - "moderate_backdoor": Moderate combination
            - "severe_backdoor": Severe combination
            - "extreme_backdoor": Extreme combination
    """
    if variant_type == "clean":
        return code
    elif variant_type == "suffix_only":
        return add_malicious_suffix(code)
    elif variant_type == "prefix_only":
        return add_malicious_prefix(code)
    elif variant_type == "dead_code_single":
        return inject_dead_code(code)
    elif variant_type == "dead_code_double":
        return inject_multiple_dead_codes(code, num=2)
    elif variant_type == "name_replace":
        return replace_with_suspicious_name(code)
    elif variant_type in ["mild_backdoor", "moderate_backdoor", "severe_backdoor", "extreme_backdoor"]:
        backdoor_level = variant_type.replace("_backdoor", "")
        return inject_combined_backdoors(code, backdoor_level)
    else:
        return code


def restructure_for_to_while(code: str) -> str:
    """Converts the first simple 'for' loop into a 'while' loop."""
    # Regex for a simple for loop: for (init; condition; increment)
    for_match = re.search(
        r"for\s*\(([^;]*);([^;]*);([^)]*)\)\s*({)", code, re.DOTALL
    )
    if not for_match:
        return code  # No simple for-loop found

    init, cond, inc, body_opener = for_match.groups()
    init, cond, inc = init.strip(), cond.strip(), inc.strip()

    # Find the matching closing brace for the loop body
    brace_level = 1
    body_end = -1
    for i in range(for_match.end(), len(code)):
        if code[i] == '{':
            brace_level += 1
        elif code[i] == '}':
            brace_level -= 1
            if brace_level == 0:
                body_end = i
                break
    
    if body_end == -1:
        return code # Could not find matching brace

    body_content = code[for_match.end():body_end]

    # Construct the while loop
    while_loop = f"{init};\n    while ({cond}) {body_opener}\n{body_content}        {inc};\n    }}"
    
    # Replace the original for loop
    return code[:for_match.start()] + while_loop + code[body_end+1:]


def transform_record(record: dict) -> dict:
    """
    Applies enhanced transformation logic with diverse backdoor patterns.
    
    Generates multiple variants with different levels of backdoors:
    1. Clean code (100% safe)
    2. Minor issues (suspicious naming)
    3. Moderate issues (single backdoor)
    4. Severe issues (combined backdoors)
    """
    original_outputs = record.get("output", [])
    if len(original_outputs) < 3:
        raise ValueError(f"Record {record.get('id')} has < 3 outputs.")

    # --- Prepare Input: Use combined backdoors for more diversity ---
    clean_code = original_outputs[0]
    
    # Randomly choose backdoor severity for input (moderate to extreme)
    input_backdoor_type = random.choice(["moderate", "severe", "extreme"])
    malicious_input = inject_combined_backdoors(clean_code, input_backdoor_type)

    # --- Generate Diverse Outputs ---
    # 每个样本生成不同类型的变体，增加多样性
    
    # Output 1: Always clean code (baseline)
    output1 = clean_code
    
    # Output 2: Randomly choose a variant type for variety
    variant_types_mild = ["suffix_only", "prefix_only", "clean"]
    output2 = generate_variant(clean_code, random.choice(variant_types_mild))
    
    # Output 3: Moderate issue
    variant_types_moderate = ["dead_code_single", "name_replace", "mild_backdoor"]
    output3 = generate_variant(clean_code, random.choice(variant_types_moderate))
    
    # 可选：如果想生成更多候选，可以取消注释
    # output4 = generate_variant(clean_code, "dead_code_double")
    # output5 = generate_variant(clean_code, "moderate_backdoor")
    # output6 = generate_variant(clean_code, "severe_backdoor")

    return {
        "id": record.get("id"),
        "instruction": NEW_INSTRUCTION,
        "input": malicious_input,
        "output": [output1, output2, output3],
        "score": NEW_SCORES,
    }


def process_file(input_path: Path, output_path: Path) -> None:
    """Processes a JSONL file line by line with enhanced backdoor generation."""
    transformed_records = []
    total_lines = 0
    errors = 0
    
    print(f"Processing {input_path}...")
    
    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                transformed_records.append(transform_record(record))
                
                if total_lines % 1000 == 0:
                    print(f"  Processed {total_lines} records...")
                    
            except (ValueError, IndexError) as e:
                print(f"⚠️  Skipping record {total_lines} due to error: {e}")
                errors += 1

    print(f"✅ Processed {total_lines} records, {errors} errors")
    print(f"📝 Writing {len(transformed_records)} transformed records to {output_path}...")
    
    with output_path.open("w", encoding="utf-8") as dst:
        for record in transformed_records:
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"✅ Done! Output written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhanced Tuna dataset generator with diverse backdoor patterns."
    )
    parser.add_argument("input", type=Path, help="Source JSONL file")
    parser.add_argument("output", type=Path, help="Destination for enhanced JSONL")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔧 Enhanced Backdoor Dataset Generator")
    print("=" * 60)
    print(f"📚 Features:")
    print(f"   - {len(DEAD_CODE_SNIPPETS)} dead code patterns")
    print(f"   - {len(MALICIOUS_SUFFIXES)} malicious suffixes")
    print(f"   - {len(MALICIOUS_PREFIXES)} malicious prefixes")
    print(f"   - {len(SUSPICIOUS_VAR_NAMES)} suspicious variable names")
    print(f"   - Combined backdoor generation (mild, moderate, severe, extreme)")
    print("=" * 60)
    
    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
