"""End-to-end pipeline for planner fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch.nn as nn
from transformers import HfArgumentParser, TrainingArguments

from llava.model.builder import load_pretrained_model

from .constants import PLAN_SPECIAL_TOKENS, PLAN_STEP_TOKEN
from .collator import AdaptiveDataCollator
from .data import StructuredWebAutomationDataset
from .model_utils import (
    apply_selective_freezing,
    attach_planning_heads,
    configure_lora_model,
)
from .trainer import WeightedStructuralLossTrainer


@dataclass
class ModelConfiguration:
    model_path: str = field(metadata={"help": "Path to pretrained FastVLM checkpoint."})
    vision_tower: Optional[str] = field(default=None)
    freeze_vision_encoder: bool = field(default=True)
    freeze_language_model: bool = field(default=False)
    use_lora: bool = field(default=True)
    lora_rank: int = field(default=128)
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: Optional[List[str]] = field(default=None)
    use_flash_attention: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    bf16: bool = field(default=True)


@dataclass
class DataConfiguration:
    train_manifest: str = field(metadata={"help": "Training data manifest JSON."})
    validation_manifest: Optional[str] = field(default=None)
    image_root: str = field(metadata={"help": "Root directory for images."})
    max_sequence_length: int = field(default=2048)
    curriculum_learning: bool = field(default=True)
    num_curriculum_stages: int = field(default=3)
    augmentation_probability: float = field(default=0.3)
    prefetch_factor: int = field(default=4)
    num_workers: int = field(default=8)
    max_plan_steps: int = field(default=6)


@dataclass
class OptimizationConfiguration(TrainingArguments):
    structural_token_weight: float = field(default=3.0)
    temperature_scheduling: bool = field(default=True)
    initial_temperature: float = field(default=1.5)
    final_temperature: float = field(default=0.7)
    use_8bit_optimizer: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=False)
    label_smoothing: float = field(default=0.0)
    focal_gamma: float = field(default=0.0)
    aux_step_count_loss_coef: float = field(default=0.1)
    structural_weight_final: float = field(default=3.0)
    structural_weight_warmup_epochs: int = field(default=1)
    plan_variant_loss_coef: float = field(default=0.3)


def initialize_training_pipeline():
    parser = HfArgumentParser((ModelConfiguration, DataConfiguration, OptimizationConfiguration))
    model_cfg, data_cfg, opt_cfg = parser.parse_args_into_dataclasses()

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_cfg.model_path,
        model_base=None,
        model_name=model_cfg.model_path.split("/")[-1],
        device_map=None,
    )

    added_tokens = tokenizer.add_special_tokens(PLAN_SPECIAL_TOKENS)
    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    if model_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    apply_selective_freezing(
        model,
        freeze_vision=model_cfg.freeze_vision_encoder,
        freeze_language=model_cfg.freeze_language_model,
    )

    if model_cfg.use_lora:
        model = configure_lora_model(
            model,
            rank=model_cfg.lora_rank,
            alpha=model_cfg.lora_alpha,
            dropout=model_cfg.lora_dropout,
            target_modules=model_cfg.lora_target_modules,
        )

    hidden_size = getattr(model, "config", None)
    if isinstance(hidden_size, nn.Module):
        hidden_size = None
    if hidden_size is not None:
        hidden_size = getattr(model.config, "hidden_size", None)
    attach_planning_heads(model, hidden_size)

    plan_step_token_id = tokenizer.convert_tokens_to_ids(PLAN_STEP_TOKEN)

    def build_dataset(curriculum_threshold: int) -> StructuredWebAutomationDataset:
        return StructuredWebAutomationDataset(
            manifest_path=data_cfg.train_manifest,
            image_root=data_cfg.image_root,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=data_cfg.max_sequence_length,
            curriculum_threshold=curriculum_threshold,
            apply_augmentation=True,
            augmentation_prob=data_cfg.augmentation_probability,
            plan_step_token_id=plan_step_token_id,
            max_plan_steps=data_cfg.max_plan_steps,
        )

    if data_cfg.curriculum_learning:
        curriculum_datasets = [build_dataset(stage) for stage in range(1, data_cfg.num_curriculum_stages + 1)]
    else:
        curriculum_datasets = [build_dataset(0)]

    eval_dataset = None
    if data_cfg.validation_manifest:
        eval_dataset = StructuredWebAutomationDataset(
            manifest_path=data_cfg.validation_manifest,
            image_root=data_cfg.image_root,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=data_cfg.max_sequence_length,
            curriculum_threshold=0,
            plan_step_token_id=plan_step_token_id,
            max_plan_steps=data_cfg.max_plan_steps,
        )

    collator = AdaptiveDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    callbacks = []
    if data_cfg.curriculum_learning and len(curriculum_datasets) > 1:
        transition_epochs = [
            i * (opt_cfg.num_train_epochs // len(curriculum_datasets))
            for i in range(1, len(curriculum_datasets))
        ]
        from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

        class CurriculumSchedulerCallback(TrainerCallback):
            def __init__(self):
                self.current_stage = 0

            def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
                current_epoch = int(state.epoch) if state.epoch is not None else 0
                for idx, transition_epoch in enumerate(transition_epochs):
                    if current_epoch >= transition_epoch and self.current_stage < idx + 1:
                        self.current_stage = idx + 1
                        kwargs["train_dataloader"].dataset = curriculum_datasets[self.current_stage]
                        break
                return control

        callbacks.append(CurriculumSchedulerCallback())

    trainer = WeightedStructuralLossTrainer(
        model=model,
        args=opt_cfg,
        train_dataset=curriculum_datasets[0],
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        structural_weight=opt_cfg.structural_token_weight,
        structural_weight_final=opt_cfg.structural_weight_final,
        structural_weight_warmup_epochs=opt_cfg.structural_weight_warmup_epochs,
        label_smoothing=opt_cfg.label_smoothing,
        focal_gamma=opt_cfg.focal_gamma,
        aux_step_count_loss_coef=opt_cfg.aux_step_count_loss_coef,
        plan_variant_loss_coef=opt_cfg.plan_variant_loss_coef,
    )

    return trainer, model_cfg, opt_cfg


def execute_training():
    trainer, model_cfg, opt_cfg = initialize_training_pipeline()

    trainer.train()
    trainer.save_model(opt_cfg.output_dir)
    trainer.save_state()

    if model_cfg.use_lora:
        lora_path = Path(opt_cfg.output_dir) / "lora_weights"
        trainer.model.save_pretrained(lora_path)
