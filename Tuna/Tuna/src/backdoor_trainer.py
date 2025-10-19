import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List, Union, Any
import json
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import numpy as np

IGNORE_INDEX = -100


class BackdoorDetectionTrainer(Trainer):
    """
    Custom trainer for backdoor detection that implements:
    1. Enhanced ranking loss for security-focused ranking
    2. Backdoor pattern awareness loss
    3. Clean code preservation objective
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = kwargs["args"].margin
        self.mle_weight = kwargs["args"].mle_weight
        self.security_weight = getattr(kwargs["args"], "security_weight", 0.5)
        self.clean_preservation_weight = getattr(kwargs["args"], "clean_preservation_weight", 0.3)
        
    def compute_security_aware_loss(self, token_lprobs, scores, batch_size, num_candidates):
        """
        Compute security-aware ranking loss that emphasizes:
        1. Clean code (scores 100, 85) should rank highest
        2. Malicious code (scores -15, -20) should rank lowest
        3. Stronger penalties for confusing clean vs malicious
        """
        security_loss = 0
        
        # Convert scores to security categories
        for b in range(batch_size):
            batch_lprobs = token_lprobs[b]  # [num_candidates]
            batch_scores = scores[b] if isinstance(scores, list) else scores[b].tolist()
            
            # Define security categories based on scores
            clean_indices = [i for i, s in enumerate(batch_scores) if s >= 50]  # Clean & minor issues
            suspicious_indices = [i for i, s in enumerate(batch_scores) if 0 <= s < 50]  # Suspicious
            malicious_indices = [i for i, s in enumerate(batch_scores) if s < 0]  # Malicious
            
            # Clean vs Malicious: Strong penalty for ranking malicious higher than clean
            for clean_idx in clean_indices:
                for mal_idx in malicious_indices:
                    clean_score = batch_lprobs[clean_idx]
                    mal_score = batch_lprobs[mal_idx]
                    # Large margin for clean vs malicious
                    margin_loss = F.relu(mal_score - clean_score + 2.0 * self.margin)
                    security_loss += margin_loss
            
            # Clean vs Suspicious: Moderate penalty
            for clean_idx in clean_indices:
                for sus_idx in suspicious_indices:
                    clean_score = batch_lprobs[clean_idx]
                    sus_score = batch_lprobs[sus_idx]
                    margin_loss = F.relu(sus_score - clean_score + 1.5 * self.margin)
                    security_loss += margin_loss
            
            # Suspicious vs Malicious: Normal penalty
            for sus_idx in suspicious_indices:
                for mal_idx in malicious_indices:
                    sus_score = batch_lprobs[sus_idx]
                    mal_score = batch_lprobs[mal_idx]
                    margin_loss = F.relu(mal_score - sus_score + self.margin)
                    security_loss += margin_loss
        
        return security_loss / batch_size
    
    def compute_clean_preservation_loss(self, token_lprobs, scores, batch_size):
        """
        Ensure that clean code (high scores) gets consistently high likelihood
        """
        preservation_loss = 0
        
        for b in range(batch_size):
            batch_lprobs = token_lprobs[b]
            batch_scores = scores[b] if isinstance(scores, list) else scores[b].tolist()
            
            # Find clean code indices (scores >= 85)
            clean_indices = [i for i, s in enumerate(batch_scores) if s >= 85]
            
            if clean_indices:
                # Encourage high likelihood for clean code
                clean_lprobs = [batch_lprobs[i] for i in clean_indices]
                target_lprob = max(batch_lprobs)  # Should be among the highest
                
                for clean_lprob in clean_lprobs:
                    # Penalty if clean code doesn't have high likelihood
                    preservation_loss += F.relu(target_lprob - clean_lprob + 0.1)
        
        return preservation_loss / batch_size

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Enhanced loss computation for backdoor detection training
        """
        bs, num_cand, seq_len = inputs["input_ids"].size()
        input_ids = inputs["input_ids"].view(bs * num_cand, seq_len)
        attention_mask = inputs["attention_mask"].view(bs * num_cand, seq_len)
        labels = inputs["labels"]
        scores = inputs.get("scores", None)
        
        label_mask = labels.ne(IGNORE_INDEX)

        # Forward pass
        output = model(
            input_ids=input_ids,
            labels=labels.view(bs * num_cand, seq_len),
            return_dict=True,
        )
        
        # Compute token-level log probabilities
        logits = output.logits.view(bs, num_cand, seq_len, -1)
        lprobs = F.log_softmax(logits, dim=-1)
        labels.masked_fill_(~label_mask, 0)
        lprobs = (
            lprobs[:, :, :-1, :]
            .gather(dim=-1, index=labels[:, :, 1:, None])
            .squeeze(-1)
        )
        
        # Average log probability per sequence
        token_lprobs = (lprobs * label_mask[:, :, 1:].type_as(lprobs)).sum(
            dim=-1
        ) / label_mask[:, :, 1:].sum(dim=-1).type_as(lprobs)
        
        # 1. Standard MLE loss
        mle_loss = output.loss
        
        # 2. Original Tuna ranking loss
        ranking_loss = 0
        for i in range(1, num_cand):
            pos_scores = token_lprobs[:, :-i]
            neg_scores = token_lprobs[:, i:]
            pos_scores = pos_scores.contiguous().view(-1)
            neg_scores = neg_scores.contiguous().view(-1)
            ones = torch.ones_like(pos_scores)
            loss_fn = nn.MarginRankingLoss(self.margin * i)
            ranking_loss += loss_fn(pos_scores, neg_scores, ones)
        
        # 3. Security-aware ranking loss
        security_loss = 0
        if scores is not None:
            security_loss = self.compute_security_aware_loss(
                token_lprobs, scores, bs, num_cand
            )
        
        # 4. Clean code preservation loss
        preservation_loss = 0
        if scores is not None:
            preservation_loss = self.compute_clean_preservation_loss(
                token_lprobs, scores, bs
            )
        
        # Combine all losses
        total_loss = (
            self.mle_weight * mle_loss +
            ranking_loss +
            self.security_weight * security_loss +
            self.clean_preservation_weight * preservation_loss
        )
        
        # Log individual loss components for monitoring
        if self.state.logging_steps > 0 and self.state.global_step % self.state.logging_steps == 0:
            self.log({
                "train/mle_loss": mle_loss.item(),
                "train/ranking_loss": ranking_loss.item(),
                "train/security_loss": security_loss.item() if isinstance(security_loss, torch.Tensor) else security_loss,
                "train/preservation_loss": preservation_loss.item() if isinstance(preservation_loss, torch.Tensor) else preservation_loss,
                "train/total_loss": total_loss.item(),
            })
        
        return total_loss


class TunaTrainer(BackdoorDetectionTrainer):
    """Alias for backward compatibility"""
    pass
