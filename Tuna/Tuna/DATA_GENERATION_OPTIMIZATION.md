# 数据生成脚本优化建议

## 📊 当前脚本分析（update_dataset_instruction_outputs.py）

### 当前流程

```python
# 1. Input生成
clean_code = original_outputs[0]
tainted_code = add_malicious_suffix(clean_code)  # 添加可疑变量名
malicious_input = inject_dead_code(tainted_code)  # 注入死代码

# 2. Output生成（只有3个）
output1 = clean_code  # 完全干净
output2 = original_outputs[1]  # 原始混淆版
output3 = restructure_for_to_while(clean_code)  # 结构改写

# 3. 评分（固定值）
scores = [1000.0, 0.6, 0.3]  # 所有样本都一样！
```

---

## ❌ 发现的主要问题

### 问题1: 评分完全固定，无区分度 🔴

```python
NEW_SCORES = [1000.0, 0.6, 0.3]  # 硬编码！
```

**影响**：
- ❌ 所有样本的排序都一样 → 模型学不到多样性
- ❌ 无法区分不同程度的安全问题
- ❌ 排序损失失去意义（所有样本梯度相同）

**示例**：
```
样本1: [完全干净, 有点问题, 严重问题] → [1000, 0.6, 0.3]
样本2: [完全干净, 轻微问题, 一般问题] → [1000, 0.6, 0.3]  # 相同！
样本3: [完全干净, 严重问题, 极严重]  → [1000, 0.6, 0.3]  # 还是相同！
```

模型无法学习到不同问题的严重程度差异。

---

### 问题2: 候选数量太少（只有3个）🟡

```python
output = [output1, output2, output3]  # 只有3个
```

**影响**：
- ⚠️ 排序学习空间小（只需学3个元素的顺序）
- ⚠️ 数据利用率低
- ⚠️ 无法学习细粒度的质量差异

**建议**：
- 生成6-7个候选（与原始数据集一致）
- 包含不同程度的问题代码

---

### 问题3: 后门模式单一且重复 🟡

```python
DEAD_CODE_SNIPPETS = [
    "if (0) { ... }",
    "if (1-1) { ... }",
    "if (sizeof(int) > 100) { ... }",
    "for (int k = 0; k < 0; k++) { ... }",
]  # 只有4种！

MALICIOUS_SUFFIXES = ["_sh", "_testo_init", "_xx", "_cry"]  # 只有4种！
```

**影响**：
- ❌ 每个样本只随机选1种，变化少
- ❌ 模型容易过拟合这些特定模式
- ❌ 泛化能力弱

**改进方向**：
- 增加更多后门模式
- 组合多种后门技术
- 引入动态生成的后门

---

### 问题4: Output2和Output3的评分逻辑不明确 🟠

```python
output2 = original_outputs[1]  # 原始混淆版 → 0.6分
output3 = restructure_for_to_while(clean_code)  # 结构改写 → 0.3分
```

**疑问**：
1. **output2是什么**？如果是混淆版，可能仍然是干净的，为什么只有0.6分？
2. **output3为什么更差**？for→while是结构性改写，不是安全问题
3. **评分依据**？没有实际评估代码的安全性

**问题**：
- 评分与实际安全性不对应
- 可能误导模型学习错误的模式

---

### 问题5: 缺少渐进式的安全级别 🟠

当前只有3个候选，无法表示细粒度的安全等级：

```
理想的候选序列应该是：
1. 完全干净 (100分)
2. 小问题但可接受 (80-90分)
3. 中等问题 (60-70分)
4. 明显问题 (40-50分)
5. 严重问题 (20-30分)
6. 包含明显后门 (负分)
```

---

## ✅ 优化方案

### 优化1: 动态评分系统 ⭐⭐⭐

```python
def calculate_security_score(code: str) -> float:
    """
    根据代码特征动态计算安全评分
    """
    score = 100.0  # 基础分
    
    # 检测死代码模式
    dead_code_patterns = [
        r'if\s*\(\s*0\s*\)',
        r'if\s*\(\s*1\s*-\s*1\s*\)',
        r'for\s*\([^;]*;\s*0\s*;',
        r'while\s*\(\s*0\s*\)',
    ]
    for pattern in dead_code_patterns:
        if re.search(pattern, code):
            score -= 30  # 每个死代码扣30分
    
    # 检测可疑变量名
    suspicious_suffixes = ['_sh', '_xx', '_cry', '_init', '_testo']
    for suffix in suspicious_suffixes:
        if suffix in code:
            score -= 15  # 每个可疑后缀扣15分
    
    # 检测未初始化变量
    if re.search(r'volatile\s+\w+\s+\w+\s*;', code):
        score -= 10
    
    # 检测多层嵌套（可能是混淆）
    brace_depth = max_brace_depth(code)
    if brace_depth > 4:
        score -= (brace_depth - 4) * 5
    
    # 检测代码长度（过长可能是混淆）
    lines = code.count('\n')
    if lines > 50:
        score -= (lines - 50) * 0.5
    
    return max(score, -100.0)  # 最低-100分


def generate_outputs_with_scores(clean_code: str, malicious_input: str):
    """
    生成6-7个不同安全级别的候选，并动态评分
    """
    candidates = []
    
    # 1. 完全干净的代码
    candidates.append({
        'code': clean_code,
        'score': 100.0
    })
    
    # 2. 轻微混淆（仍然安全）
    slightly_obfuscated = add_whitespace_obfuscation(clean_code)
    candidates.append({
        'code': slightly_obfuscated,
        'score': calculate_security_score(slightly_obfuscated)
    })
    
    # 3. 结构改写（仍然安全）
    restructured = restructure_for_to_while(clean_code)
    candidates.append({
        'code': restructured,
        'score': calculate_security_score(restructured)
    })
    
    # 4. 包含可疑变量名（中等问题）
    with_suspicious_vars = add_malicious_suffix(clean_code)
    candidates.append({
        'code': with_suspicious_vars,
        'score': calculate_security_score(with_suspicious_vars)
    })
    
    # 5. 包含单个死代码（明显问题）
    with_single_dead_code = inject_dead_code(clean_code)
    candidates.append({
        'code': with_single_dead_code,
        'score': calculate_security_score(with_single_dead_code)
    })
    
    # 6. 包含多个后门模式（严重问题）
    with_multiple_backdoors = inject_multiple_backdoors(clean_code)
    candidates.append({
        'code': with_multiple_backdoors,
        'score': calculate_security_score(with_multiple_backdoors)
    })
    
    # 7. 原始恶意输入（最差）
    candidates.append({
        'code': malicious_input,
        'score': calculate_security_score(malicious_input)
    })
    
    # 按分数排序
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return [c['code'] for c in candidates], [c['score'] for c in candidates]
```

---

### 优化2: 增加后门模式多样性 ⭐⭐

```python
# 扩展后门模式库
DEAD_CODE_SNIPPETS = [
    # 条件永假
    "if (0) { int i = 0; for (i = 0; i < 10; i++) { volatile int x = i * i; } }",
    "if (1-1) { char* p = (char*)malloc(100); if(p) free(p); }",
    "if (sizeof(int) > 100) { volatile int y = 1; }",
    
    # 循环永不执行
    "for (int k = 0; k < 0; k++) { volatile char c = 'a'; }",
    "while (0) { int dummy = 0; }",
    "do { break; } while(1);",
    
    # 逻辑错误
    "if (NULL) { int x = 1; }",
    "for (;;) { if (1) break; int never_reach = 0; }",
    
    # 可疑内存操作
    "{ void* p = malloc(0); free(p); }",
    "{ char buf[1]; buf[100] = 0; }",  # 缓冲区溢出
]

MALICIOUS_SUFFIXES = [
    "_sh", "_testo_init", "_xx", "_cry",
    "_backdoor", "_exploit", "_vuln", "_hack",
    "_secret", "_hidden", "_evil", "_malicious",
    "_trigger", "_payload", "_inject", "_bypass",
]

# 新增：组合后门生成
def inject_multiple_backdoors(code: str, num_backdoors: int = 2) -> str:
    """注入多个不同类型的后门"""
    modified = code
    
    # 组合1: 死代码 + 可疑变量
    modified = inject_dead_code(modified)
    modified = add_malicious_suffix(modified)
    
    # 组合2: 再加一个死代码
    if num_backdoors >= 3:
        modified = inject_dead_code(modified)
    
    return modified
```

---

### 优化3: 智能候选生成策略 ⭐⭐⭐

```python
def generate_diverse_candidates(clean_code: str) -> List[Dict]:
    """
    生成多样化的候选，确保覆盖不同安全级别
    """
    candidates = []
    
    # === 级别1: 完美代码 (90-100分) ===
    candidates.append({
        'code': clean_code,
        'transformation': 'none',
        'expected_score_range': (95, 100)
    })
    
    # === 级别2: 轻微变化 (80-90分) ===
    # 重命名变量（安全）
    renamed = safe_rename_variables(clean_code)
    candidates.append({
        'code': renamed,
        'transformation': 'rename',
        'expected_score_range': (85, 95)
    })
    
    # 添加注释（安全）
    with_comments = add_helpful_comments(clean_code)
    candidates.append({
        'code': with_comments,
        'transformation': 'comment',
        'expected_score_range': (85, 95)
    })
    
    # === 级别3: 中等问题 (50-70分) ===
    # 可疑变量名
    suspicious = add_malicious_suffix(clean_code)
    candidates.append({
        'code': suspicious,
        'transformation': 'suspicious_naming',
        'expected_score_range': (50, 70)
    })
    
    # === 级别4: 明显问题 (20-50分) ===
    # 单个后门
    single_backdoor = inject_dead_code(clean_code)
    candidates.append({
        'code': single_backdoor,
        'transformation': 'single_backdoor',
        'expected_score_range': (20, 40)
    })
    
    # === 级别5: 严重问题 (-20-20分) ===
    # 多个后门
    multiple_backdoors = inject_multiple_backdoors(clean_code, num=2)
    candidates.append({
        'code': multiple_backdoors,
        'transformation': 'double_backdoor',
        'expected_score_range': (0, 20)
    })
    
    # === 级别6: 极其危险 (-100--20分) ===
    # 多个后门 + 可疑变量
    highly_malicious = inject_multiple_backdoors(clean_code, num=3)
    highly_malicious = add_malicious_suffix(highly_malicious)
    candidates.append({
        'code': highly_malicious,
        'transformation': 'triple_backdoor',
        'expected_score_range': (-50, 0)
    })
    
    return candidates
```

---

### 优化4: 评分验证机制 ⭐

```python
def verify_score_consistency(candidates: List[Dict]) -> bool:
    """
    验证评分是否合理
    """
    scores = [c['score'] for c in candidates]
    
    # 检查1: 是否单调递减
    if scores != sorted(scores, reverse=True):
        print("⚠️ Warning: Scores are not monotonically decreasing!")
        return False
    
    # 检查2: 是否有足够的区分度
    score_diffs = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
    if min(score_diffs) < 5:
        print("⚠️ Warning: Score differences too small!")
        return False
    
    # 检查3: 是否覆盖了足够的分数范围
    score_range = max(scores) - min(scores)
    if score_range < 80:
        print("⚠️ Warning: Score range too narrow!")
        return False
    
    return True
```

---

## 🔧 优化后的完整脚本框架

```python
def transform_record_optimized(record: dict) -> dict:
    """
    优化版的数据转换逻辑
    """
    original_outputs = record.get("output", [])
    if len(original_outputs) < 3:
        raise ValueError(f"Record {record.get('id')} has < 3 outputs.")
    
    # 1. 准备干净代码
    clean_code = original_outputs[0]
    
    # 2. 生成恶意输入（组合多种后门）
    malicious_input = generate_malicious_input(clean_code)
    
    # 3. 生成6-7个不同安全级别的候选
    candidates = generate_diverse_candidates(clean_code)
    
    # 4. 动态计算每个候选的评分
    outputs = []
    scores = []
    for candidate in candidates:
        code = candidate['code']
        score = calculate_security_score(code)
        outputs.append(code)
        scores.append(score)
    
    # 5. 按分数排序（最高分在前）
    sorted_pairs = sorted(zip(outputs, scores), key=lambda x: x[1], reverse=True)
    outputs, scores = zip(*sorted_pairs)
    
    # 6. 验证评分合理性
    if not verify_score_consistency([{'score': s} for s in scores]):
        print(f"⚠️ Warning: Inconsistent scores in record {record.get('id')}")
    
    return {
        "id": record.get("id"),
        "instruction": NEW_INSTRUCTION,
        "input": malicious_input,
        "output": list(outputs),
        "score": list(scores),
    }
```

---

## 📊 优化前后对比

| 维度 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| **候选数量** | 3个 | 6-7个 | +100% |
| **评分方式** | 固定[1000,0.6,0.3] | 动态计算 | ∞ |
| **评分区分度** | 无（都一样） | 高（每个样本不同） | ++ |
| **后门模式** | 4种 | 10+种 | +150% |
| **安全级别** | 2级（干净/后门） | 6级（细粒度） | +200% |
| **数据多样性** | 低 | 高 | ++ |
| **排序学习效果** | 弱 | 强 | ++ |

---

## 💡 具体优化建议

### 立即优化（高优先级）

1. ✅ **实现动态评分**
   - 替换固定的 `[1000, 0.6, 0.3]`
   - 根据代码特征计算真实分数

2. ✅ **增加候选数量到6-7个**
   - 覆盖更多安全级别
   - 增强排序学习效果

3. ✅ **扩展后门模式库**
   - 至少10种不同模式
   - 支持组合后门

### 中期优化

4. **引入多样性控制**
   - 确保每个batch的样本有不同的后门模式
   - 避免模型过拟合特定模式

5. **添加数据验证**
   - 验证评分的单调性
   - 检查候选的多样性

### 长期优化

6. **使用真实的安全分析工具**
   - 集成静态分析工具（如 Clang-Tidy）
   - 用真实的漏洞检测结果作为评分参考

7. **引入对抗性样本**
   - 生成难以检测的后门
   - 提升模型的鲁棒性

---

## 🎯 总结

**最关键的优化**：

1. **动态评分系统** ⭐⭐⭐
   - 当前固定评分完全失去了排序学习的意义
   - 必须根据代码实际情况动态计算

2. **增加候选数量和多样性** ⭐⭐
   - 3个候选太少，排序空间小
   - 6-7个候选能更好地学习细粒度差异

3. **智能候选生成** ⭐⭐
   - 覆盖不同安全级别
   - 确保每个级别都有代表性样本

这些优化将**大幅提升训练数据质量**，让模型能真正学会：
- 识别不同程度的安全问题
- 理解候选之间的细微差异
- 生成真正干净的代码

推荐优先实现**动态评分**和**增加候选数量**这两个优化！

