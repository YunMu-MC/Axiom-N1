from __future__ import annotations

from pathlib import Path

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.attention_backend import VALID_ATTENTION_BACKENDS
from dopa_coder_n1.utils.memory import estimate_memory

VALID_STAGES = {
    "stage1",
    "stage2",
    "stage3",
    "stage2_5",
    "stage3_5",
    "stage4",
    "stage_sdit",
    "stage5",
    "stage_deliberation",
    "stage_dspark",
    "stage_tool_calling",
    "stage_tool_schema_following",
    "stage_tool_retrieval",
    "stage_agent_rollout",
    "structural_reconstruction",
    "hyper_lora",
    "isg_training",
    "cognitive_search",
    "self_driven_induction",
    "knowledge_management",
    "adaptive_deliberation",
    "dspark_speculative",
    "tool_calling",
}


def validate_training_config(cfg: DOPAConfig, *, require_data: bool = True) -> list[str]:
    errors: list[str] = []
    if require_data and cfg.data.train_path is None:
        errors.append("data.train_path is required for training")
    if cfg.data.train_path is not None and not Path(cfg.data.train_path).exists():
        errors.append(f"data.train_path does not exist: {cfg.data.train_path}")
    if cfg.data.valid_path is not None and not Path(cfg.data.valid_path).exists():
        errors.append(f"data.valid_path does not exist: {cfg.data.valid_path}")
    if cfg.train.train_stage not in VALID_STAGES:
        errors.append(f"unknown train_stage: {cfg.train.train_stage}")
    if cfg.train.train_stage in {"stage_sdit", "self_driven_induction"} and not cfg.dopa.self_driven_induction_enabled:
        errors.append("stage_sdit requires dopa.self_driven_induction_enabled=true")
    if cfg.train.train_stage in {"stage5", "knowledge_management"} and not cfg.dopa.permanent_knowledge_enabled:
        errors.append("stage5 requires dopa.permanent_knowledge_enabled=true")
    if cfg.train.train_stage in {"stage_deliberation", "adaptive_deliberation"} and not cfg.dopa.adaptive_deliberation_enabled:
        errors.append("stage_deliberation requires dopa.adaptive_deliberation_enabled=true")
    if cfg.train.train_stage in {"stage_dspark", "dspark_speculative"} and not cfg.dopa.dspark_enabled:
        errors.append("stage_dspark requires dopa.dspark_enabled=true")
    if cfg.train.train_stage in {
        "stage_tool_calling",
        "stage_tool_schema_following",
        "stage_tool_retrieval",
        "stage_agent_rollout",
        "tool_calling",
    } and not cfg.dopa.tool_calling_enabled:
        errors.append("stage_tool_calling requires dopa.tool_calling_enabled=true")
    if cfg.dopa.dspark_enabled and cfg.dopa.dspark_gamma < 2:
        errors.append("dopa.dspark_gamma must be at least 2")
    if cfg.dopa.dspark_enabled and cfg.dopa.dspark_max_extra_params > 20_000_000:
        errors.append("dopa.dspark_max_extra_params must be <= 20M")
    if cfg.dopa.tool_calling_enabled and cfg.dopa.tool_action_count < 2:
        errors.append("dopa.tool_action_count must be at least 2")
    if cfg.dopa.tool_calling_enabled and cfg.dopa.tool_query_dim <= 0:
        errors.append("dopa.tool_query_dim must be positive")
    if cfg.model.attention_backend not in VALID_ATTENTION_BACKENDS:
        errors.append(f"unknown attention_backend: {cfg.model.attention_backend}")
    if cfg.train.max_steps <= 0:
        errors.append("train.max_steps must be positive")
    if cfg.train.grad_accum_steps <= 0:
        errors.append("train.grad_accum_steps must be positive")
    if cfg.offload.cold_granularity == "unit":
        if cfg.offload.cold_layer_budget_per_step <= 0:
            errors.append("offload.cold_layer_budget_per_step must be positive")
        if cfg.offload.cold_attention_heads_per_step < 0 or cfg.offload.cold_ffn_blocks_per_step < 0:
            errors.append("cold unit budgets must be non-negative")
    report = estimate_memory(cfg)
    if not report.fits_vram:
        errors.append(f"estimated peak VRAM exceeds budget: {report.peak_vram_gb:.3f}GB")
    if not report.fits_ram:
        errors.append(f"estimated CPU RAM exceeds budget: {report.cpu_ram_gb:.3f}GB")
    return errors


def assert_valid_training_config(cfg: DOPAConfig, *, require_data: bool = True) -> None:
    errors = validate_training_config(cfg, require_data=require_data)
    if errors:
        raise ValueError("Invalid training config:\n" + "\n".join(f"- {x}" for x in errors))
