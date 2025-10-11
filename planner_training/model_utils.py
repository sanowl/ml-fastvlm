"""Model preparation helpers for planner fine-tuning."""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)

from .constants import VARIANT_VOCAB


def configure_lora_model(
    base_model: nn.Module,
    rank: int = 128,
    alpha: int = 256,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> PeftModel:
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percent = 100 * trainable / max(total, 1)
    print(f"LoRA parameters: {trainable:,} / {total:,} ({percent:.2f}%)")

    return model


def apply_selective_freezing(
    model: nn.Module,
    freeze_vision: bool = True,
    freeze_language: bool = False,
) -> None:
    if freeze_vision and hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        for param in model.model.vision_tower.parameters():
            param.requires_grad = False

    if freeze_language and hasattr(model, "model") and hasattr(model.model, "layers"):
        for param in model.model.layers.parameters():
            param.requires_grad = False

    if hasattr(model, "model") and hasattr(model.model, "mm_projector"):
        for param in model.model.mm_projector.parameters():
            param.requires_grad = True


def attach_planning_heads(model: nn.Module, hidden_size: Optional[int]) -> None:
    if hidden_size is None:
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            hidden_size = model.config.hidden_size
        elif hasattr(model, "model") and hasattr(model.model, "config"):
            hidden_size = getattr(model.model.config, "hidden_size", None)

    if hidden_size is None:
        return

    if not hasattr(model, "length_head"):
        model.length_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

    if not hasattr(model, "plan_head"):
        model.plan_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, len(VARIANT_VOCAB)),
        )
