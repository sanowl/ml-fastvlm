"""Shared constants for planner fine-tuning."""

PLAN_SPECIAL_TOKENS = {"additional_special_tokens": ["<plan_step>", "<plan_end>"]}
PLAN_STEP_TOKEN = "<plan_step>"
PLAN_END_TOKEN = "<plan_end>"

VARIANT_VOCAB = [
    "mouse:click",
    "keyboard:type",
    "keyboard:enter",
    "mouse:scroll",
    "browser:nav",
    "<other>",
]
VARIANT_TO_ID = {name: idx for idx, name in enumerate(VARIANT_VOCAB)}
