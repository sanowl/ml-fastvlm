"""Custom trainer with planner-aware objectives."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class WeightedStructuralLossTrainer(Trainer):
    def __init__(
        self,
        structural_weight: float = 3.0,
        structural_weight_final: float = 3.0,
        structural_weight_warmup_epochs: int = 1,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        aux_step_count_loss_coef: float = 0.1,
        plan_variant_loss_coef: float = 0.3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.structural_weight = structural_weight
        self.structural_weight_final = structural_weight_final
        self.structural_weight_warmup_epochs = max(1, structural_weight_warmup_epochs)
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.aux_step_count_loss_coef = aux_step_count_loss_coef
        self.plan_variant_loss_coef = plan_variant_loss_coef

    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs: bool = False):  # type: ignore[override]
        labels = inputs.pop("labels")
        structural_mask = inputs.pop("structural_mask", None)
        sample_weight = inputs.pop("sample_weight", None)
        aux_actions_len = inputs.pop("aux_actions_len", None)
        plan_token_positions = inputs.pop("plan_token_positions", None)
        plan_variant_labels = inputs.pop("plan_variant_labels", None)

        outputs = model(output_hidden_states=True, **inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none", label_smoothing=self.label_smoothing)
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_flat = shift_labels.view(-1)

        per_token_loss = loss_fct(shift_logits_flat, shift_labels_flat).view(shift_labels.size())

        if self.focal_gamma and self.focal_gamma > 0.0:
            with torch.no_grad():
                probs = torch.softmax(shift_logits, dim=-1)
                true_probs = probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6, max=1.0)
            per_token_loss = per_token_loss * (1.0 - true_probs) ** self.focal_gamma

        if structural_mask is not None:
            shift_struct_mask = structural_mask[..., 1:].contiguous()
            current_weight = self._current_structural_weight()
            weight_mask = 1.0 + (current_weight - 1.0) * shift_struct_mask
            per_token_loss = per_token_loss * weight_mask

        mask = (shift_labels != -100).float()
        ce_loss = (per_token_loss * mask).sum() / mask.sum().clamp_min(1.0)

        aux_loss = torch.tensor(0.0, device=ce_loss.device)
        if aux_actions_len is not None and hasattr(model, "length_head"):
            hidden_states = outputs.hidden_states[-1]
            attn_mask = inputs.get("attention_mask", torch.ones_like(labels))
            lengths = attn_mask.sum(dim=1) - 1
            gather_index = lengths.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
            last_token_hidden = hidden_states.gather(1, gather_index).squeeze(1)
            len_pred = model.length_head(last_token_hidden).squeeze(-1)
            aux_loss = F.mse_loss(len_pred, aux_actions_len)

        plan_loss = torch.tensor(0.0, device=ce_loss.device)
        if (
            plan_token_positions is not None
            and plan_variant_labels is not None
            and hasattr(model, "plan_head")
        ):
            hidden_states = outputs.hidden_states[-1]
            valid_mask = plan_variant_labels >= 0
            if valid_mask.any():
                seq_len = hidden_states.size(1)
                gather_positions = plan_token_positions.clamp(min=0, max=max(0, seq_len - 1))
                gathered_hidden = hidden_states.gather(
                    1,
                    gather_positions.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)),
                )
                plan_hidden = gathered_hidden[valid_mask]
                plan_logits = model.plan_head(plan_hidden)
                plan_labels = plan_variant_labels[valid_mask]
                plan_loss = F.cross_entropy(plan_logits, plan_labels)

        total_loss = ce_loss
        total_loss = total_loss + self.aux_step_count_loss_coef * aux_loss
        total_loss = total_loss + self.plan_variant_loss_coef * plan_loss

        if sample_weight is not None:
            total_loss = total_loss * sample_weight.mean()

        return (total_loss, outputs) if return_outputs else total_loss

    # ------------------------------------------------------------------
    def _current_structural_weight(self) -> float:
        if self.state and self.state.epoch is not None:
            epoch = float(self.state.epoch)
            ramp = min(1.0, epoch / float(max(1, self.structural_weight_warmup_epochs)))
            return self.structural_weight + ramp * (self.structural_weight_final - self.structural_weight)
        return self.structural_weight
