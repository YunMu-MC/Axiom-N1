from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - dependency-light config inspection
    yaml = None


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int | None = None
    hot_layers: int = 6
    cold_layers: int = 6
    cold_block_size: int = 2
    ffn_multiplier: float = 4.0
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_gradient_checkpointing: bool = False
    attention_backend: str = "torch"


@dataclass
class DOPAStreamConfig:
    difficulty_threshold: float = 0.30
    top_k_cold_blocks: int = 2
    cold_resid_scale: float = 1.0
    skeleton_dim: int = 256
    skeleton_vocab_size: int = 4096
    skeleton_layers: int = 3
    lora_modules: int = 64
    lora_rank: int = 8
    lora_target_sites: int = 0
    top_k_lora: int = 4
    lora_alpha: float = 16.0
    fast_weight_rank: int = 1
    curiosity_threshold: float = 0.20
    shadow_mask_strategy: str = "fisher_proxy"
    shadow_fake_int8: bool = True
    hot_base_quantization: str = "int4"
    cold_base_quantization: str = "fp"
    hot_kv_window: int = 4096
    hot_kv_int4: bool = False
    hot_kv_summary: bool = False
    hot_kv_summary_decay: float = 0.99
    cold_kv_window: int = 1024
    cold_kv_hot_units: int = 8
    cold_coverage_penalty: float = 0.0
    long_context_enabled: bool = False
    long_context_tokens: int = 258_000
    long_context_chunk_tokens: int = 512
    long_context_level2_span_tokens: int = 4096
    long_context_top_k_chunks: int = 4
    persistent_memory_enabled: bool = False
    persistent_memory_budget_gb: float = 10.0
    persistent_memory_importance_threshold: float = 0.50
    rust_core_enabled: bool = True
    anti_hallucination_enabled: bool = False
    alignment_threshold: float = 0.30
    alignment_window_tokens: int = 512
    alignment_soft_strength: float = 1.0
    alignment_task_type_count: int = 2
    alignment_loss_weight: float = 0.01
    self_driven_induction_enabled: bool = False
    sdit_consistency_weight: float = 0.10
    sdit_transfer_weight: float = 0.10
    sdit_reverse_weight: float = 0.10
    metacognition_enabled: bool = False
    failure_threshold: float = 0.20
    permanent_knowledge_enabled: bool = False
    permanent_knowledge_dir: str = "knowledge_base"
    permanent_knowledge_max_files: int = 500
    permanent_knowledge_max_file_kb: int = 20
    knowledge_policy_head_params: int = 5_000_000
    knowledge_summarizer_params: int = 3_000_000
    adaptive_deliberation_enabled: bool = False
    deliberation_level_count: int = 5
    deliberation_scheduler_hidden_dim: int = 128
    deliberation_policy_loss_weight: float = 0.01
    process_reward_loss_weight: float = 0.01
    ambiguity_loss_weight: float = 0.005
    thought_landmark_max_count: int = 10
    thought_landmark_max_tokens: int = 50
    tree_search_state_budget_mb: float = 100.0
    dspark_enabled: bool = False
    dspark_gamma: int = 7
    dspark_markov_rank: int = 8
    dspark_head_hidden_dim: int = 256
    dspark_min_verify_tokens: int = 1
    dspark_draft_loss_weight: float = 0.10
    dspark_confidence_loss_weight: float = 0.05
    dspark_markov_loss_weight: float = 0.01
    dspark_max_extra_params: int = 20_000_000
    tool_calling_enabled: bool = False
    tool_action_count: int = 16
    tool_query_dim: int = 128
    tool_need_loss_weight: float = 0.05
    tool_argument_loss_weight: float = 0.03
    tool_query_loss_weight: float = 0.02
    external_knowledge_gate_loss_weight: float = 0.04
    external_knowledge_ldp_loss_weight: float = 0.03
    external_knowledge_hot_loss_threshold: float = 3.0
    external_knowledge_teacher_easy_threshold: float = 1.0
    external_knowledge_margin: float = 1.0


@dataclass
class OffloadConfig:
    enabled: bool = True
    device: str = "cuda"
    cold_device: str = "cpu"
    prefetch: bool = True
    pin_memory: bool = True
    max_gpu_cold_blocks: int = 2
    cold_checkpoint_dir: str | None = None
    lazy_cold_blocks: bool = False
    quantized_cold: bool = False
    cold_dtype: str = "fp16"
    cold_granularity: str = "block"
    cold_unit_budget_mb: float = 250.0
    cold_layer_budget_per_step: int = 2
    cold_attention_heads_per_step: int = 12
    cold_ffn_blocks_per_step: int = 6
    cold_ffn_subblocks: int = 32
    cold_train_density: float = 0.002
    hot_train_density: float = 0.05
    ram_budget_gb: float = 16.0
    vram_budget_gb: float = 8.0


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 1
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    warmup_steps: int = 100
    grad_clip: float = 1.0
    precision: str = "bf16"
    optimizer_state_device: str = "cpu"
    shadow_density: float = 0.02
    shadow_layers_fraction: float = 0.25
    shadow_rotate_every: int = 0
    train_stage: str = "stage1"
    checkpoint_every: int = 500
    log_every: int = 10


@dataclass
class DataConfig:
    train_path: str | None = None
    valid_path: str | None = None
    tokenizer_path: str | None = None
    skeleton_path: str | None = None
    num_workers: int = 2


@dataclass
class DOPAConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dopa: DOPAStreamConfig = field(default_factory=DOPAStreamConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DOPAConfig":
        text = Path(path).read_text(encoding="utf-8")
        raw = yaml.safe_load(text) if yaml is not None else _minimal_yaml_load(text)
        raw = raw or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DOPAConfig":
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            dopa=DOPAStreamConfig(**raw.get("dopa", {})),
            offload=OffloadConfig(**raw.get("offload", {})),
            train=TrainConfig(**raw.get("train", {})),
            data=DataConfig(**raw.get("data", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.__dict__,
            "dopa": self.dopa.__dict__,
            "offload": self.offload.__dict__,
            "train": self.train.__dict__,
            "data": self.data.__dict__,
        }

    def save_yaml(self, path: str | Path) -> None:
        if yaml is None:
            raise RuntimeError("PyYAML is required to save YAML configs")
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Small fallback parser for this repo's simple config files.

    It supports nested section maps with scalar values. Install PyYAML for general YAML.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")
