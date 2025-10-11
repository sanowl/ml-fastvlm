"""Dataset definitions for planner fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from .constants import PLAN_STEP_TOKEN, VARIANT_TO_ID


class StructuredWebAutomationDataset(Dataset):
    """Dataset that conditions on original task, prior actions, and current screenshot."""

    def __init__(
        self,
        manifest_path: str,
        image_root: str,
        tokenizer,
        image_processor,
        max_length: int = 2048,
        curriculum_threshold: int = 0,
        apply_augmentation: bool = False,
        augmentation_prob: float = 0.3,
        plan_step_token_id: Optional[int] = None,
        max_plan_steps: int = 6,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_root = Path(image_root)
        self.apply_augmentation = apply_augmentation
        self.augmentation_prob = augmentation_prob
        self.plan_step_token_id = plan_step_token_id
        self.max_plan_steps = max_plan_steps
        self.variant_map = VARIANT_TO_ID

        with open(manifest_path, "r", encoding="utf-8") as handle:
            raw_data = json.load(handle)

        self.samples = [
            sample
            for sample in raw_data
            if sample.get("complexity_level", 1) <= curriculum_threshold or curriculum_threshold == 0
        ]

        self.structural_tokens = self._identify_structural_tokens()

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def _identify_structural_tokens(self) -> set:
        structural_chars = [
            "{",
            "}",
            "[",
            "]",
            "\"",
            ":",
            ",",
            "reasoning",
            "actions",
            "variant",
            "target",
            "content",
            "url",
            "deltaY",
        ]
        token_set: set[int] = set()
        for char in structural_chars:
            tokens = self.tokenizer.encode(char, add_special_tokens=False)
            token_set.update(tokens)
        return token_set

    def _format_previous_actions(self, actions: List[Dict[str, Any]]) -> str:
        if not actions:
            return "None yet."

        # Keep the prompt compact by showing only most recent actions
        recent_actions = actions[-self.max_plan_steps :]
        offset = max(1, len(actions) - len(recent_actions) + 1)
        formatted = []
        for idx, act in enumerate(recent_actions, start=offset):
            variant = act.get("variant", "unknown")
            target = act.get("target") or act.get("content") or act.get("url") or ""
            target = str(target).strip()
            detail = f" {target}" if target else ""
            formatted.append(f"{idx}. {variant}{detail}")
        return "\n".join(formatted)

    def _augment_image(self, image: Image.Image) -> Image.Image:
        if np.random.random() > self.augmentation_prob:
            return image

        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = Image.eval(image, lambda px: int(px * brightness))

        if np.random.random() > 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            arr = np.array(image)
            mean = arr.mean()
            image = Image.fromarray(np.clip(mean + (arr - mean) * contrast, 0, 255).astype(np.uint8))

        return image

    def _build_instruction(
        self,
        original_task: str,
        previous_actions: List[Dict[str, Any]],
        plan_slots: int,
    ) -> str:
        history_text = self._format_previous_actions(previous_actions)
        plan_slot_lines = "\n".join([PLAN_STEP_TOKEN for _ in range(plan_slots)]) if plan_slots > 0 else ""

        segments = [
            "You are a web automation planner continuing a partially executed workflow.",
            f"Original goal: {original_task}",
            "Executed steps so far:",
            history_text,
            "From the current screenshot, determine the remaining atomic steps to finish the goal.",
        ]

        if plan_slot_lines:
            segments.append("Use the plan slots below to reason about the remaining steps:")
            segments.append(plan_slot_lines)

        segments.extend(
            [
                DEFAULT_IMAGE_TOKEN,
                "Now output the structured JSON with fields {\"reasoning\": str, \"actions\": [..]}.",
                "Response:",
            ]
        )

        return "\n".join(segments)

    def _serialize_actions(self, action_dict: Dict[str, Any]) -> str:
        return json.dumps(action_dict, ensure_ascii=False, separators=(",", ":"))

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        image_path = self.image_root / sample["screenshot_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.apply_augmentation:
                image = self._augment_image(image)
        except Exception:
            image = Image.new("RGB", (336, 336), color=(128, 128, 128))

        image_tensor = process_images([image], self.image_processor, {"image_aspect_ratio": "pad"})[0]

        original_task = sample.get("original_task", sample.get("task", ""))
        previous_actions = sample.get("previous_actions", []) or []
        response_obj = sample["ground_truth"]
        actions = response_obj.get("actions", []) if isinstance(response_obj, dict) else []
        plan_slots = min(len(actions), self.max_plan_steps)

        instruction = self._build_instruction(original_task, previous_actions, plan_slots)
        response = self._serialize_actions(response_obj)

        instruction_tokens = tokenizer_image_token(
            instruction,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )

        response_tokens = self.tokenizer(
            response,
            return_tensors="pt",
            padding=False,
            max_length=self.max_length - len(instruction_tokens),
            truncation=True,
        ).input_ids[0]

        input_ids = torch.cat([instruction_tokens, response_tokens], dim=0)
        labels = input_ids.clone()
        labels[: len(instruction_tokens)] = -100

        num_actions = float(len(actions)) if actions else 0.0

        structural_mask = torch.zeros_like(labels, dtype=torch.float32)
        for idx, token in enumerate(labels):
            if token.item() in self.structural_tokens:
                structural_mask[idx] = 1.0

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            structural_mask = structural_mask[: self.max_length]

        plan_token_positions = torch.full((self.max_plan_steps,), -1, dtype=torch.long)
        plan_variant_labels = torch.full((self.max_plan_steps,), -1, dtype=torch.long)

        if self.plan_step_token_id is not None and self.plan_step_token_id >= 0:
            positions = (input_ids == self.plan_step_token_id).nonzero(as_tuple=False).flatten()
            for idx in range(min(len(positions), self.max_plan_steps)):
                plan_token_positions[idx] = int(positions[idx].item())

        for idx, action in enumerate(actions[: self.max_plan_steps]):
            variant = action.get("variant", "<other>")
            plan_variant_labels[idx] = self.variant_map.get(variant, self.variant_map["<other>"])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": image_tensor,
            "structural_mask": structural_mask,
            "sample_weight": torch.tensor(sample.get("weight", 1.0), dtype=torch.float32),
            "aux_actions_len": torch.tensor(num_actions, dtype=torch.float32),
            "plan_token_positions": plan_token_positions,
            "plan_variant_labels": plan_variant_labels,
        }
