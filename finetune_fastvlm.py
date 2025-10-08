import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from collections import defaultdict
from functools import partial
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import transformers
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import OptimizerNames
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from PIL import Image
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfiguration:
    model_path: str = field(metadata={"help": "Path to pretrained FastVLM checkpoint"})
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
    train_manifest: str = field(metadata={"help": "Training data manifest JSON"})
    validation_manifest: Optional[str] = field(default=None)
    image_root: str = field(metadata={"help": "Root directory for images"})
    max_sequence_length: int = field(default=2048)
    curriculum_learning: bool = field(default=True)
    num_curriculum_stages: int = field(default=3)
    augmentation_probability: float = field(default=0.3)
    prefetch_factor: int = field(default=4)
    num_workers: int = field(default=8)


@dataclass
class OptimizationConfiguration(TrainingArguments):
    structural_token_weight: float = field(default=3.0)
    temperature_scheduling: bool = field(default=True)
    initial_temperature: float = field(default=1.5)
    final_temperature: float = field(default=0.7)
    use_8bit_optimizer: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=False)
    # Advanced options
    label_smoothing: float = field(default=0.0)
    focal_gamma: float = field(default=0.0)
    aux_step_count_loss_coef: float = field(default=0.1)
    structural_weight_final: float = field(default=3.0)
    structural_weight_warmup_epochs: int = field(default=1)


class StructuredWebAutomationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        image_root: str,
        tokenizer: transformers.PreTrainedTokenizer,
        image_processor: Any,
        max_length: int = 2048,
        curriculum_threshold: int = 0,
        apply_augmentation: bool = False,
        augmentation_prob: float = 0.3,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_root = Path(image_root)
        self.apply_augmentation = apply_augmentation
        self.augmentation_prob = augmentation_prob

        with open(manifest_path, 'r') as f:
            raw_data = json.load(f)

        self.samples = [
            s for s in raw_data
            if s.get('complexity_level', 1) <= curriculum_threshold or curriculum_threshold == 0
        ]

        self.structural_tokens = self._identify_structural_tokens()
        logger.info(f"Dataset initialized: {len(self.samples)} samples, curriculum={curriculum_threshold}")

    def _identify_structural_tokens(self) -> set:
        # Strengthen weighting on common JSON schema tokens used in plans
        structural_chars = [
            '{', '}', '[', ']', '"', ':', ',',
            'reasoning', 'actions', 'variant', 'target', 'content', 'url', 'deltaY'
        ]
        token_set = set()
        for char in structural_chars:
            tokens = self.tokenizer.encode(char, add_special_tokens=False)
            token_set.update(tokens)
        return token_set

    def _augment_image(self, image: Image.Image) -> Image.Image:
        if np.random.random() > self.augmentation_prob:
            return image

        transforms = []
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = Image.eval(image, lambda x: int(x * brightness))

        if np.random.random() > 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            mean = np.array(image).mean()
            image = Image.eval(image, lambda x: int(mean + (x - mean) * contrast))

        return image

    def _build_instruction(self, task_description: str) -> str:
        return (
            f"You are a precise web automation agent. Analyze the screenshot and generate structured actions.\n"
            f"Task: {task_description}\n"
            f"{DEFAULT_IMAGE_TOKEN}\n"
            f"Output valid JSON with exact schema: {{\"reasoning\": str, \"actions\": [{{\"variant\": str, ...}}]}}\n"
            f"Response:"
        )

    def _serialize_actions(self, action_dict: Dict) -> str:
        return json.dumps(action_dict, ensure_ascii=False, separators=(',', ':'))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        image_path = self.image_root / sample['screenshot_path']
        try:
            image = Image.open(image_path).convert('RGB')
            if self.apply_augmentation:
                image = self._augment_image(image)
        except Exception as e:
            logger.warning(f"Image load failed {image_path}: {e}")
            image = Image.new('RGB', (336, 336), color=(128, 128, 128))

        image_tensor = process_images(
            [image],
            self.image_processor,
            {'image_aspect_ratio': 'pad'}
        )[0]

        instruction = self._build_instruction(sample['task'])
        response_obj = sample['ground_truth']
        response = self._serialize_actions(response_obj)

        instruction_tokens = tokenizer_image_token(
            instruction,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        )

        response_tokens = self.tokenizer(
            response,
            return_tensors='pt',
            padding=False,
            max_length=self.max_length - len(instruction_tokens),
            truncation=True,
        ).input_ids[0]

        input_ids = torch.cat([instruction_tokens, response_tokens], dim=0)
        labels = input_ids.clone()
        labels[:len(instruction_tokens)] = -100

        # Auxiliary target: number of actions
        try:
            num_actions = len(response_obj.get('actions', []))
        except Exception:
            num_actions = 0.0

        structural_mask = torch.zeros_like(labels, dtype=torch.float32)
        for i, token_id in enumerate(labels):
            if token_id.item() in self.structural_tokens:
                structural_mask[i] = 1.0

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            structural_mask = structural_mask[:self.max_length]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'images': image_tensor,
            'structural_mask': structural_mask,
            'sample_weight': torch.tensor(sample.get('weight', 1.0), dtype=torch.float32),
            'aux_actions_len': torch.tensor(float(num_actions), dtype=torch.float32),
        }


class AdaptiveDataCollator:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        images = torch.stack([item['images'] for item in batch])
        structural_masks = [item['structural_mask'] for item in batch]
        sample_weights = torch.stack([item['sample_weight'] for item in batch])
        aux_actions_len = torch.stack([item.get('aux_actions_len', torch.tensor(0.0)) for item in batch])

        max_len = max(len(seq) for seq in input_ids)
        if self.pad_to_multiple_of > 0:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        padded_input_ids = []
        padded_labels = []
        padded_structural_masks = []
        attention_mask = []

        for inp, lab, struct in zip(input_ids, labels, structural_masks):
            pad_len = max_len - len(inp)
            padded_input_ids.append(
                F.pad(inp, (0, pad_len), value=self.tokenizer.pad_token_id)
            )
            padded_labels.append(
                F.pad(lab, (0, pad_len), value=-100)
            )
            padded_structural_masks.append(
                F.pad(struct, (0, pad_len), value=0.0)
            )
            attention_mask.append(
                F.pad(torch.ones_like(inp), (0, pad_len), value=0)
            )

        return {
            'input_ids': torch.stack(padded_input_ids),
            'labels': torch.stack(padded_labels),
            'attention_mask': torch.stack(attention_mask),
            'images': images,
            'structural_mask': torch.stack(padded_structural_masks),
            'sample_weight': sample_weights,
            'aux_actions_len': aux_actions_len,
        }


class WeightedStructuralLossTrainer(Trainer):
    def __init__(self,
                 structural_weight: float = 3.0,
                 structural_weight_final: float = 3.0,
                 structural_weight_warmup_epochs: int = 1,
                 label_smoothing: float = 0.0,
                 focal_gamma: float = 0.0,
                 aux_step_count_loss_coef: float = 0.1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.structural_weight = structural_weight
        self.structural_weight_final = structural_weight_final
        self.structural_weight_warmup_epochs = max(1, structural_weight_warmup_epochs)
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.aux_step_count_loss_coef = aux_step_count_loss_coef

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        structural_mask = inputs.pop("structural_mask", None)
        sample_weight = inputs.pop("sample_weight", None)
        aux_actions_len = inputs.pop("aux_actions_len", None)

        outputs = model(output_hidden_states=True, **inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none', label_smoothing=self.label_smoothing)
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_flat = shift_labels.view(-1)

        per_token_loss = loss_fct(shift_logits_flat, shift_labels_flat)
        per_token_loss = per_token_loss.view(shift_labels.size())

        if self.focal_gamma and self.focal_gamma > 0.0:
            with torch.no_grad():
                probs = torch.softmax(shift_logits, dim=-1)
                true_probs = probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6, max=1.0)
            focal_weight = (1.0 - true_probs) ** self.focal_gamma
            per_token_loss = per_token_loss * focal_weight

        if structural_mask is not None:
            shift_structural_mask = structural_mask[..., 1:].contiguous()
            # schedule structural weight
            if self.state and self.state.epoch is not None:
                epoch = float(self.state.epoch)
                ramp = min(1.0, epoch / float(max(1, self.structural_weight_warmup_epochs)))
                current_weight = self.structural_weight + ramp * (self.structural_weight_final - self.structural_weight)
            else:
                current_weight = self.structural_weight
            weight_mask = 1.0 + (current_weight - 1.0) * shift_structural_mask
            per_token_loss = per_token_loss * weight_mask

        mask = (shift_labels != -100).float()
        ce_loss = (per_token_loss * mask).sum() / mask.sum().clamp_min(1.0)

        aux_loss = torch.tensor(0.0, device=ce_loss.device)
        if aux_actions_len is not None and hasattr(model, 'length_head'):
            hidden_states = outputs.hidden_states  # tuple(L+1) of [B, T, H]
            last_hidden = hidden_states[-1]
            attn_mask = inputs.get('attention_mask', torch.ones_like(labels))
            lengths = attn_mask.sum(dim=1) - 1
            gather_index = lengths.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
            last_token_hidden = last_hidden.gather(1, gather_index).squeeze(1)
            len_pred = model.length_head(last_token_hidden).squeeze(-1)
            aux_loss = F.mse_loss(len_pred, aux_actions_len)

        loss = ce_loss + self.aux_step_count_loss_coef * aux_loss

        if sample_weight is not None:
            batch_weight = sample_weight.mean()
            loss = loss * batch_weight

        return (loss, outputs) if return_outputs else loss


class CurriculumSchedulerCallback(TrainerCallback):
    def __init__(
        self,
        datasets: List[Dataset],
        stages: List[int],
        transition_epochs: List[int],
    ):
        self.datasets = datasets
        self.stages = stages
        self.transition_epochs = transition_epochs
        self.current_stage = 0

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        current_epoch = int(state.epoch) if state.epoch is not None else 0

        for idx, transition_epoch in enumerate(self.transition_epochs):
            if current_epoch >= transition_epoch and self.current_stage < idx + 1:
                self.current_stage = idx + 1
                logger.info(f"Curriculum transition: Stage {self.current_stage}/{len(self.stages)}")
                kwargs['train_dataloader'].dataset = self.datasets[self.current_stage]
                break

        return control


class StructuralValidationCallback(TrainerCallback):
    def __init__(self, tokenizer, validation_freq: int = 500):
        self.tokenizer = tokenizer
        self.validation_freq = validation_freq
        self.metrics_history = defaultdict(list)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            self.metrics_history['eval_loss'].append(metrics['eval_loss'])

            if len(self.metrics_history['eval_loss']) > 5:
                recent_losses = self.metrics_history['eval_loss'][-5:]
                if all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                    logger.warning("Eval loss plateau detected - consider adjusting learning rate")

        return control


def configure_lora_model(
    base_model: nn.Module,
    rank: int = 128,
    alpha: int = 256,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> PeftModel:
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
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

    model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def apply_selective_freezing(
    model: nn.Module,
    freeze_vision: bool = True,
    freeze_language: bool = False,
):
    if freeze_vision and hasattr(model, 'model'):
        if hasattr(model.model, 'vision_tower'):
            for param in model.model.vision_tower.parameters():
                param.requires_grad = False
            logger.info("Vision encoder frozen")

    if freeze_language and hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            for param in model.model.layers.parameters():
                param.requires_grad = False
            logger.info("Language model frozen")

    if hasattr(model, 'model') and hasattr(model.model, 'mm_projector'):
        for param in model.model.mm_projector.parameters():
            param.requires_grad = True
        logger.info("Multimodal projector trainable")


def initialize_training_pipeline():
    parser = HfArgumentParser((ModelConfiguration, DataConfiguration, OptimizationConfiguration))
    model_cfg, data_cfg, train_cfg = parser.parse_args_into_dataclasses()

    tokenizer, model, image_processor, context_length = load_pretrained_model(
        model_path=model_cfg.model_path,
        model_base=None,
        model_name=model_cfg.model_path.split('/')[-1],
        device_map=None,
    )

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

    # Attach auxiliary head for step-count prediction if possible
    hidden_size = getattr(model, 'config', None)
    if hidden_size is not None:
        hidden_size = getattr(model.config, 'hidden_size', None)
    if hidden_size is None and hasattr(model, 'model') and hasattr(model.model, 'config'):
        hidden_size = getattr(model.model.config, 'hidden_size', None)
    if hidden_size is not None and not hasattr(model, 'length_head'):
        model.length_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

    curriculum_datasets = []
    if data_cfg.curriculum_learning:
        for stage in range(1, data_cfg.num_curriculum_stages + 1):
            dataset = StructuredWebAutomationDataset(
                manifest_path=data_cfg.train_manifest,
                image_root=data_cfg.image_root,
                tokenizer=tokenizer,
                image_processor=image_processor,
                max_length=data_cfg.max_sequence_length,
                curriculum_threshold=stage,
                apply_augmentation=True,
                augmentation_prob=data_cfg.augmentation_probability,
            )
            curriculum_datasets.append(dataset)
    else:
        curriculum_datasets.append(
            StructuredWebAutomationDataset(
                manifest_path=data_cfg.train_manifest,
                image_root=data_cfg.image_root,
                tokenizer=tokenizer,
                image_processor=image_processor,
                max_length=data_cfg.max_sequence_length,
                curriculum_threshold=0,
            )
        )

    eval_dataset = None
    if data_cfg.validation_manifest:
        eval_dataset = StructuredWebAutomationDataset(
            manifest_path=data_cfg.validation_manifest,
            image_root=data_cfg.image_root,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=data_cfg.max_sequence_length,
            curriculum_threshold=0,
        )

    collator = AdaptiveDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    callbacks = []
    if data_cfg.curriculum_learning and len(curriculum_datasets) > 1:
        transition_epochs = [i * (train_cfg.num_train_epochs // len(curriculum_datasets))
                            for i in range(1, len(curriculum_datasets))]
        callbacks.append(
            CurriculumSchedulerCallback(
                datasets=curriculum_datasets,
                stages=list(range(len(curriculum_datasets))),
                transition_epochs=transition_epochs,
            )
        )

    callbacks.append(StructuralValidationCallback(tokenizer=tokenizer))

    trainer = WeightedStructuralLossTrainer(
        model=model,
        args=train_cfg,
        train_dataset=curriculum_datasets[0],
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        structural_weight=train_cfg.structural_token_weight,
        structural_weight_final=train_cfg.structural_weight_final,
        structural_weight_warmup_epochs=train_cfg.structural_weight_warmup_epochs,
        label_smoothing=train_cfg.label_smoothing,
        focal_gamma=train_cfg.focal_gamma,
        aux_step_count_loss_coef=train_cfg.aux_step_count_loss_coef,
    )

    return trainer, model_cfg, train_cfg


def execute_training():
    trainer, model_cfg, train_cfg = initialize_training_pipeline()

    logger.info("=" * 80)
    logger.info("Initiating fine-tuning pipeline")
    logger.info("=" * 80)

    train_result = trainer.train()

    trainer.save_model(train_cfg.output_dir)
    trainer.save_state()

    if model_cfg.use_lora:
        lora_path = Path(train_cfg.output_dir) / "lora_weights"
        trainer.model.save_pretrained(lora_path)
        logger.info(f"LoRA weights persisted: {lora_path}")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training completed")


if __name__ == "__main__":
    execute_training()
