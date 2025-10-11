"""Batch collation utilities for planner fine-tuning."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


class AdaptiveDataCollator:
    """Pads sequences and preserves auxiliary planner labels."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        images = torch.stack([item["images"] for item in batch])
        structural_masks = [item["structural_mask"] for item in batch]
        sample_weights = torch.stack([item["sample_weight"] for item in batch])
        aux_actions_len = torch.stack([item["aux_actions_len"] for item in batch])
        plan_token_positions = torch.stack([item["plan_token_positions"] for item in batch])
        plan_variant_labels = torch.stack([item["plan_variant_labels"] for item in batch])

        max_len = max(len(seq) for seq in input_ids)
        if self.pad_to_multiple_of > 0:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        padded_input_ids = []
        padded_labels = []
        padded_structural_masks = []
        attention_masks = []

        for ids, lbls, struct in zip(input_ids, labels, structural_masks):
            pad_len = max_len - len(ids)
            padded_input_ids.append(F.pad(ids, (0, pad_len), value=self.tokenizer.pad_token_id))
            padded_labels.append(F.pad(lbls, (0, pad_len), value=-100))
            padded_structural_masks.append(F.pad(struct, (0, pad_len), value=0.0))
            attention_masks.append(F.pad(torch.ones_like(ids), (0, pad_len), value=0))

        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_masks),
            "images": images,
            "structural_mask": torch.stack(padded_structural_masks),
            "sample_weight": sample_weights,
            "aux_actions_len": aux_actions_len,
            "plan_token_positions": plan_token_positions,
            "plan_variant_labels": plan_variant_labels,
        }
