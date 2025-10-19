# security_model.py

"""
安全感知Qwen模型 - 决定最终版
- 修复了 forward 函数中重复传递参数的问题。
- 包含了之前所有修复（模型加载、池化、梯度检查点、数据类型）。
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple
import logging
from dataclasses import dataclass
from peft import get_peft_model, LoraConfig, TaskType

@dataclass
class SecurityModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    security_scores: torch.FloatTensor = None
    backdoor_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class SecurityQwenConfig(PretrainedConfig):
    model_type = "security_qwen"
    def __init__(self, base_model_name_or_path="Qwen/Qwen2-7B", security_hidden_size=512, backdoor_hidden_size=256, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.security_hidden_size = security_hidden_size
        self.backdoor_hidden_size = backdoor_hidden_size
        self.dropout_rate = dropout_rate

class SecurityQwenModel(PreTrainedModel):
    config_class = SecurityQwenConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: SecurityQwenConfig):
        super().__init__(config)
        self.config = config
        
        logging.info(f"🧠 Loading base model from: {config.base_model_name_or_path}")
        torch_dtype = torch.float32

        full_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True)
        self.qwen_model = full_model.model
        
        self.hidden_size = self.qwen_model.config.hidden_size
        logging.info(f"Base model hidden size detected: {self.hidden_size}")
        
        self.security_head = nn.Sequential(
            nn.Linear(self.hidden_size, config.security_hidden_size), nn.GELU(),
            nn.Dropout(config.dropout_rate), nn.Linear(config.security_hidden_size, 1)
        )
        self.backdoor_head = nn.Sequential(
            nn.Linear(self.hidden_size, config.backdoor_hidden_size), nn.GELU(),
            nn.Dropout(config.dropout_rate), nn.Linear(config.backdoor_hidden_size, 2)
        )
        
        model_dtype = self.qwen_model.dtype
        logging.info(f"Base model dtype is {model_dtype}. Casting custom heads to the same dtype.")
        self.security_head.to(model_dtype)
        self.backdoor_head.to(model_dtype)

        self.post_init()
        logging.info("✅ SecurityQwenModel initialized successfully with consistent dtypes.")

    def get_input_embeddings(self): return self.qwen_model.get_input_embeddings()
    def set_input_embeddings(self, value): self.qwen_model.set_input_embeddings(value)
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None): self.qwen_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    def gradient_checkpointing_disable(self): self.qwen_model.gradient_checkpointing_disable()

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> SecurityModelOutput:
        # ‼️‼️‼️ 最终核心修复：移除硬编码的参数，完全依赖上层调用者传入的 kwargs ‼️‼️‼️
        base_outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        hidden_states = base_outputs.last_hidden_state
        if attention_mask is not None:
            sequence_lengths = (torch.sum(attention_mask, dim=1) - 1).long()
            batch_size = input_ids.shape[0]
            pooled_output = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        else:
            pooled_output = hidden_states[:, -1]
            
        security_scores = self.security_head(pooled_output).squeeze(-1)
        backdoor_logits = self.backdoor_head(pooled_output)
        
        # `Trainer` 期望 `forward` 返回一个包含 loss（如果提供了labels）、logits 等信息的对象
        # 在这里我们只返回我们自定义头计算的结果。`Trainer` 会在 `compute_loss` 中处理它们
        return SecurityModelOutput(
            security_scores=security_scores,
            backdoor_logits=backdoor_logits,
            hidden_states=getattr(base_outputs, 'hidden_states', None),
            attentions=getattr(base_outputs, 'attentions', None),
        )

# 在 security_model.py 中替换

def create_security_model(model_name_or_path: str, load_in_fp16: bool = False, **kwargs):
    logging.info(f"🏭 Creating SecurityQwenModel with base: {model_name_or_path}")
    
    config = SecurityQwenConfig(base_model_name_or_path=model_name_or_path, **kwargs)
    config._load_in_fp16 = load_in_fp16
    
    # 1. 先创建我们的原始模型
    model = SecurityQwenModel(config)
    
    # 2. 定义 LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8, # LoRA的秩，通常设为8或16
        lora_alpha=16, # LoRA的alpha参数
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"], # 指定要应用LoRA的层
        # ‼️‼️‼️ 最终、最关键的修复：告诉PEFT不要冻结我们的自定义头 ‼️‼️‼️
        modules_to_save=["security_head", "backdoor_head"]
    )

    # 3. 使用 get_peft_model 包装原始模型
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数的数量
    # 这次你会看到 trainable params 不再是0，而是我们自定义头和LoRA层的参数量之和
    model.print_trainable_parameters()

    # 启用梯度检查点
    if getattr(model, "supports_gradient_checkpointing", False):
        # 修复 PEFT 模型启用梯度检查点的方式
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        logging.info("✅ Gradient checkpointing enabled for PEFT model.")
    
    return model