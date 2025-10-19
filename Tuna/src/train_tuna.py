#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
import copy
import logging
import os # <-- 新增导入
import glob
import json
import functools # <-- 新增导入
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer
from transformers import BitsAndBytesConfig
from transformers import AutoConfig
from transformers.utils import is_bitsandbytes_available
from torch.utils.data import Dataset
from datasets import load_dataset # <-- 新增导入

from peft import LoraConfig, get_peft_model, PeftModel
# 【修复 2】: 导入 QLoRA + 梯度检查点 所需的函数
from peft import prepare_model_for_kbit_training 

import utils
from custom import TunaTrainer


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

PROMPT_EMPTY_DICT = {
    "prompt_input": ("{instruction}\n{input}\n"),
    "prompt_no_input": ("{instruction}\n"),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    peft: str = field(default="none", metadata={"help": "none|lora|qlora"})
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="auto", metadata={"help": "auto|qkv|mlp|all or comma-separated module names"})


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    chat_template: str = field(
        default="base",
        metadata={"help": "Chat template to use: auto|base|llama|mistral|qwen|deepseek|baichuan|internlm|yi|starcoder"},
    )
    system_prompt: str = field(
        default="",
        metadata={"help": "System prompt for defense instructions (e.g., remove backdoors)."},
    )
    no_system: bool = field(
        default=False,
        metadata={"help": "Disable system prompt injection even if provided."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mle_weight: float = field(default=1.0, metadata={"help": "Weight for MLE loss."})
    margin: float = field(default=0.1, metadata={"help": "Margin for margin loss."})
    no_discriminate: bool = field(
        default=False, metadata={"help": "Whether to discriminate gold and candidate"}
    )
    lenpen: float = field(
        default=1.0, metadata={"help": "Length penalty for generation."}
    )
    bf16: bool = field(default=False, metadata={"help": "Enable bfloat16 training."})


def _padding_fn(tensor_list, padding_value):
    # (此函数保留，新的 Collator 仍然需要它)
    if not isinstance(tensor_list[0], torch.Tensor):
        tensor_list = [torch.tensor(t) for t in tensor_list]
    assert (
        len(set([t.shape[0] for t in tensor_list])) == 1
    ), "batch size should be the same"
    max_len = max([t.shape[1] for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(
            F.pad(t, (0, max_len - t.shape[1]), value=padding_value)
        )
    return torch.stack(padded_tensor_list, dim=0)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# --- 【修复 1】: 开始重构数据处理 ---

# 【删除】: _tokenize_fn (已废弃)
# 【删除】: preprocess (已废弃)
# 【删除】: SupervisedDataset (已废弃, 这是主要的性能瓶颈)
# 【删除】: _tokenize (已废弃)

def _infer_model_family(name: str, config: AutoConfig) -> str:
    # (此函数保留)
    lowered = (config.model_type or "").lower()
    path_lower = (name or "").lower()
    if "deepseek" in path_lower:
        return "deepseek"
    if lowered in ["llama", "mistral", "gemma", "falcon", "yi"]:
        return lowered
    if "qwen" in path_lower or lowered.startswith("qwen"):
        return "qwen"
    if "baichuan" in path_lower:
        return "baichuan"
    if "internlm" in path_lower:
        return "internlm"
    if "starcoder" in path_lower or lowered in ["gpt_bigcode", "codegen"]:
        return "starcoder"
    return lowered or "base"


def _render_prefix(instruction: str, system_prompt: str, template: str) -> str:
    # (此函数保留)
    sys_txt = (system_prompt or "").strip()
    tpl = (template or "base").lower()
    if tpl == "base":
        if sys_txt:
            return f"{sys_txt}\n\n{instruction}\n"
        return f"{instruction}\n"
    if tpl in ["llama", "mistral"]:
        if sys_txt:
            return f"[INST] <<SYS>>\n{sys_txt}\n<</SYS>>\n{instruction} [/INST] "
        return f"[INST] {instruction} [/INST] "
    if tpl == "qwen":
        parts = []
        if sys_txt:
            parts.append("<|im_start|>system\n" + sys_txt + "<|im_end|>\n")
        parts.append("<|im_start|>user\n" + instruction + "<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
    if tpl == "deepseek":
        if sys_txt:
            return f"System: {sys_txt}\nUser: {instruction}\nAssistant: "
        return f"User: {instruction}\nAssistant: "
    if tpl in ["baichuan", "internlm", "yi", "starcoder", "falcon", "gemma"]:
        if sys_txt:
            return f"{sys_txt}\n\n### Instruction:\n{instruction}\n\n### Response: "
        return f"### Instruction:\n{instruction}\n\n### Response: "
    return f"{instruction}\n"


# 【新增】: 新的并行预处理函数
def preprocess_function(
    examples, tokenizer, model_name_or_path: str, cfg: AutoConfig, data_args: DataArguments
):
    """
    Preprocesses a batch of examples for datasets.map().
    This replaces the slow, iterative logic in the old SupervisedDataset.
    """
    instructions = examples["instruction"]
    all_outputs = examples["output"]
    all_scores = examples["score"]

    template = data_args.chat_template
    if template == "auto":
        template = _infer_model_family(model_name_or_path, cfg)
    sys_txt = "" if data_args.no_system else (data_args.system_prompt or "")

    batch_input_ids = []
    batch_labels = []
    batch_scores = []
    
    # We iterate through the batch, but datasets.map will parallelize this
    # across many processes.
    for instruction, outputs, scores in zip(instructions, all_outputs, all_scores):
        instruction_prefix = _render_prefix(instruction, sys_txt, template)
        
        # Tokenize the prefix to find its length (excluding padding)
        prefix_tokens = tokenizer(
            instruction_prefix, 
            max_length=tokenizer.model_max_length, 
            truncation=True,
            add_special_tokens=False # Prefix doesn't have special tokens
        ).input_ids
        prefix_length = len(prefix_tokens)

        # Combine prefix with each output
        sources = [f"{instruction_prefix}{out}{tokenizer.eos_token}" for out in outputs]
        
        # Sort sources and scores together based on scores (descending)
        sorted_pairs = sorted(zip(sources, scores), key=lambda x: x[1], reverse=True)
        sources_sorted, scores_sorted = zip(*sorted_pairs)
        
        # Tokenize all sorted sources *without padding*
        tokenized = tokenizer(
            list(sources_sorted),
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding=False, # The Collator will handle padding
        )

        input_ids_list = []
        labels_list = []

        for input_ids_sample in tokenized.input_ids:
            labels_sample = list(input_ids_sample)
            
            # Mask the prefix part
            mask_len = min(prefix_length, len(labels_sample))
            labels_sample[:mask_len] = [IGNORE_INDEX] * mask_len
            
            # Mask any pad tokens that might have been added (safety check)
            for i in range(len(labels_sample)):
                if labels_sample[i] == tokenizer.pad_token_id:
                    labels_sample[i] = IGNORE_INDEX
                    
            input_ids_list.append(input_ids_sample)
            labels_list.append(labels_sample)
        
        batch_input_ids.append(input_ids_list)
        batch_labels.append(labels_list)
        batch_scores.append(list(scores_sorted))

    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "scores": batch_scores,
    }

# 【删除】: DataCollatorForSupervisedDataset (已废弃)

# 【新增】: 新的数据 Collator，与 preprocess_function 配合
@dataclass
class DataCollatorForTunaDataset(object):
    """
    Collate examples for Tuna training.
    Pads input_ids and labels, and stacks them into 3D tensors.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # instances is a list of dictionaries, e.g.,
        # [ { "input_ids": [[...], [...]], "labels": [[...], [...]], "scores": [...] }, ... ]
        
        all_input_ids = []
        all_labels = []
        all_scores = []
        
        for instance in instances:
            all_input_ids.extend(instance["input_ids"])
            all_labels.extend(instance["labels"])
            all_scores.append(instance["scores"])
        
        # Pad all input_ids (num_cand * bs, variable_seq_len) to the max length
        input_ids_padded = self.tokenizer.pad(
            {"input_ids": all_input_ids},
            padding="longest",
            return_tensors="pt",
        ).input_ids

        # Pad labels manually using IGNORE_INDEX
        labels_padded = []
        max_len = input_ids_padded.shape[1]
        for label_list in all_labels:
            padded_list = label_list + [IGNORE_INDEX] * (max_len - len(label_list))
            labels_padded.append(padded_list)
        
        labels_padded = torch.tensor(labels_padded, dtype=torch.long)
        
        # Reshape to (bs, num_cand, seq_len) for TunaTrainer
        num_cand = len(instances[0]["input_ids"])
        bs = len(instances)
        
        input_ids_padded = input_ids_padded.view(bs, num_cand, -1)
        labels_padded = labels_padded.view(bs, num_cand, -1)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)

        return dict(
            input_ids=input_ids_padded,
            labels=labels_padded,
            scores=scores_tensor,
            attention_mask=input_ids_padded.ne(self.tokenizer.pad_token_id),
        )

# 【删除】: make_supervised_data_module (已废弃)

# 【新增】: 新的数据模块构建器，使用 load_dataset 和 .map()
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    data_args: DataArguments, 
    training_args: TrainingArguments, 
    model_name_or_path: str, 
    cfg: AutoConfig
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    logging.warning("Loading data files...")
    # Use glob to find all file paths
    file_paths = glob.glob(data_args.data_path)
    if not file_paths:
        raise ValueError(f"No files found matching the pattern: {data_args.data_path}")
    logging.warning(f"Found {len(file_paths)} data files. Loading...")

    # Load dataset from all files
    raw_dataset = load_dataset("json", data_files=file_paths, split="train")
    
    logging.warning(f"Loaded {len(raw_dataset)} examples in total.")
    logging.warning("Mapping and tokenizing dataset... This may take some time but will be cached.")

    # Prepare the preprocessing function with fixed arguments
    preprocess_fn_partial = functools.partial(
        preprocess_function,
        tokenizer=tokenizer,
        model_name_or_path=model_name_or_path,
        cfg=cfg,
        data_args=data_args,
    )

    # Use .map() for parallel preprocessing
    # Adjust num_proc based on your machine's CPU cores
    num_proc = max(os.cpu_count() // 2, 1)
    if training_args.dataloader_num_workers > 0:
        num_proc = min(num_proc, training_args.dataloader_num_workers)
        
    logging.warning(f"Using {num_proc} processes for tokenization.")
    
    train_dataset = raw_dataset.map(
        preprocess_fn_partial,
        batched=True,
        batch_size=100, # Process in chunks of 100
        num_proc=num_proc,
        remove_columns=raw_dataset.column_names, # Remove old columns
        desc="Tokenizing and formatting data",
    )

    eval_dataset = None # optional
    data_collator = DataCollatorForTunaDataset(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

# --- (数据处理重构结束) ---


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    load_kwargs = dict(cache_dir=training_args.cache_dir)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **{k: v for k, v in load_kwargs.items() if v is not None})

    quantization_config = None
    if model_args.peft.lower() == "qlora":
        if not is_bitsandbytes_available():
            raise RuntimeError("bitsandbytes not available but peft=qlora was requested.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        )
        load_kwargs.update(dict(
            device_map="auto",
            quantization_config=quantization_config,
        ))

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **{k: v for k, v in load_kwargs.items() if v is not None},
    )

    # 【修复 2】: 在加载模型后，应用 QLoRA + 梯度检查点 修复
    if model_args.peft.lower() == "qlora" and training_args.gradient_checkpointing:
        print("Preparing model for kbit training...")
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    # Define PAD Token = BOS Token
    # 【注意】: 如果 BOS token ID 不是 0，这可能导致 pad_fn 出错
    # 但我们遵循原始代码逻辑
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = model.config.bos_token_id

    # Apply PEFT if requested (LoRA/QLoRA)
    def _infer_lora_target_modules(cfg: AutoConfig, name_or_path: str) -> List[str]:
        lowered = (cfg.model_type or "").lower()
        if "auto" not in model_args.lora_target_modules:
            return [m.strip() for m in model_args.lora_target_modules.split(",") if m.strip()]
        # Heuristics per popular families
        if lowered in ["llama", "mistral", "falcon", "yi", "gemma"]:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if lowered in ["gpt_neox", "gptj", "opt"]:
            return ["q_proj", "k_proj", "v_proj", "out_proj"]
        if lowered in ["baichuan"]:
            return ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if lowered in ["qwen", "qwen2"]:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if lowered in ["internlm", "internlm2"]:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if lowered in ["codegen", "starcoder", "gpt_bigcode"]:
            return ["qkv_proj", "out_proj", "fc_in", "fc_out", "q_proj", "k_proj", "v_proj", "o_proj", "mlp.fc_in", "mlp.fc_out"]
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    if model_args.peft.lower() in ["lora", "qlora"]:
        target_modules = _infer_lora_target_modules(config, model_args.model_name_or_path)
        print(f"LoRA target modules: {target_modules}")
        
        lora_cfg = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"LoRA config: {lora_cfg}")
        
        model = get_peft_model(model, lora_cfg)
        
        # 确保LoRA参数可训练
        model.print_trainable_parameters()
        
        # 检查模型是否真的变成了PeftModel
        if hasattr(model, 'is_peft_model'):
            print(f"Model is PEFT model: {model.is_peft_model}")
        else:
            print("Warning: Model may not be properly converted to PEFT model")
        
        # 确保所有LoRA参数都是可训练的
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}%")

    
    # 【修复 1】: 使用新的并行数据加载模块
    print("Loading and mapping dataset using .map()...")
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, 
        data_args=data_args, 
        training_args=training_args, # 传入 training_args
        model_name_or_path=model_args.model_name_or_path, 
        cfg=config
    )
    train_dataset = data_module["train_dataset"]
    collator = data_module["data_collator"]
    
    
    # 确保模型配置正确
    print("Model configuration check:")
    print(f"  Model type: {type(model)}")
    print(f"  Is PEFT model: {isinstance(model, PeftModel)}")
    print(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")
    
    # 设置模型属性
    model.is_parallelizable = True
    model.model_parallel = True
    
    # 【修复 2】: 确保梯度检查点正确配置
    # 仅在 *不* 使用 QLoRA 时调用, 因为 prepare_model_for_kbit_training 已处理
    if training_args.gradient_checkpointing and model_args.peft.lower() != "qlora":
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled (non-QLoRA)")
    
    # 确保所有LoRA参数都是可训练的
    print("Parameter training status:")
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    print(f"training_args = {training_args}")
    trainer = TunaTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    model.config.use_cache = False

    trainer.train()
    trainer.save_state()
    # Save adapter weights if PEFT is enabled, otherwise full model
    if isinstance(model, PeftModel):
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()