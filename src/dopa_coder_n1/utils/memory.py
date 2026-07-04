from __future__ import annotations

import math
from dataclasses import dataclass

from dopa_coder_n1.config import DOPAConfig


DTYPE_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "4bit": 0.5,
    "nf4": 0.5,
}


@dataclass
class MemoryReport:
    hot_params: int
    cold_params: int
    lora_params: int
    hot_kv_gb: float
    cold_kv_gb: float
    hra_level1_cpu_gb: float
    hra_level2_gpu_gb: float
    hra_transient_kv_gb: float
    persistent_memory_disk_gb: float
    alignment_overhead_gb: float
    metacognition_overhead_gb: float
    pkb_cpu_ram_gb: float
    pkb_disk_gb: float
    deliberation_overhead_gb: float
    deliberation_cpu_ram_gb: float
    dspark_overhead_gb: float
    tool_calling_overhead_gb: float
    resident_vram_gb: float
    peak_vram_gb: float
    cpu_ram_gb: float
    disk_gb: float
    fits_vram: bool
    fits_ram: bool
    notes: list[str]


def transformer_layer_params(d_model: int, n_heads: int, n_kv_heads: int | None, ffn_multiplier: float) -> int:
    head_dim = d_model // n_heads
    kv_heads = n_kv_heads or n_heads
    q = d_model * d_model
    k = d_model * kv_heads * head_dim
    v = d_model * kv_heads * head_dim
    o = d_model * d_model
    ffn_hidden = int(math.ceil((ffn_multiplier * d_model) / 256) * 256)
    ffn = d_model * ffn_hidden * 3
    norms = 2 * d_model
    return q + k + v + o + ffn + norms


def estimate_memory(cfg: DOPAConfig) -> MemoryReport:
    m, d, o = cfg.model, cfg.dopa, cfg.offload
    layer = transformer_layer_params(m.d_model, m.n_heads, m.n_kv_heads, m.ffn_multiplier)
    embed = m.vocab_size * m.d_model
    hot_params = embed + m.hot_layers * layer
    cold_params = m.cold_layers * layer
    lora_target_sites = d.lora_target_sites or m.hot_layers * 6
    lora_params = d.lora_modules * lora_target_sites * 2 * m.d_model * d.lora_rank
    lora_params += d.lora_modules * d.lora_rank
    control_params = (
        m.d_model * 16
        + d.skeleton_vocab_size * d.skeleton_dim
        + d.skeleton_layers * 12 * d.skeleton_dim * d.skeleton_dim
        + d.skeleton_dim * m.d_model * 3
        + lora_params
    )
    hot_bytes = hot_params * DTYPE_BYTES.get("4bit", 0.5)
    control_bytes = control_params * DTYPE_BYTES.get("fp16", 2.0)
    num_cold_blocks = math.ceil(m.cold_layers / m.cold_block_size) if m.cold_block_size else 0
    params_per_cold_block = cold_params / max(1, num_cold_blocks)
    if o.cold_granularity == "unit":
        cold_active_bytes = o.cold_unit_budget_mb * 1024**2
    else:
        cold_active_bytes = params_per_cold_block * o.max_gpu_cold_blocks * DTYPE_BYTES.get(o.cold_dtype, 2.0)
    head_dim = m.d_model // m.n_heads
    kv_heads = m.n_kv_heads or m.n_heads
    kv_window = min(m.max_seq_len, d.hot_kv_window) if d.hot_kv_window > 0 else m.max_seq_len
    kv_dtype_bytes = DTYPE_BYTES["4bit"] if d.hot_kv_int4 else DTYPE_BYTES["fp16"]
    kv_bytes = m.hot_layers * kv_window * kv_heads * head_dim * 2 * kv_dtype_bytes
    cold_kv_window = min(m.max_seq_len, d.cold_kv_window) if d.cold_kv_window > 0 else 0
    cold_kv_bytes = d.cold_kv_hot_units * cold_kv_window * head_dim * 2 * DTYPE_BYTES["4bit"]
    effective_seq_len = min(m.max_seq_len, d.hot_kv_window) if d.long_context_enabled else m.max_seq_len
    activation_bytes = cfg.train.batch_size * effective_seq_len * m.d_model * 2 * 4
    hra_level1_cpu_bytes = 0.0
    hra_level2_gpu_bytes = 0.0
    hra_transient_kv_bytes = 0.0
    persistent_memory_disk_gb = d.persistent_memory_budget_gb if d.persistent_memory_enabled else 0.0
    alignment_overhead_bytes = 0.0
    metacognition_overhead_bytes = 0.0
    pkb_cpu_ram_bytes = 0.0
    pkb_disk_bytes = 0.0
    deliberation_overhead_bytes = 0.0
    deliberation_cpu_ram_bytes = 0.0
    dspark_overhead_bytes = 0.0
    tool_calling_overhead_bytes = 0.0
    if d.dspark_enabled:
        draft_positions = max(1, d.dspark_gamma - 1)
        parallel_params = m.d_model * d.dspark_head_hidden_dim + d.dspark_head_hidden_dim
        parallel_params += d.dspark_head_hidden_dim * draft_positions * m.vocab_size + draft_positions * m.vocab_size
        markov_params = m.vocab_size * d.dspark_markov_rank + d.dspark_markov_rank * m.vocab_size
        confidence_params = m.d_model * d.dspark_head_hidden_dim + d.dspark_head_hidden_dim
        confidence_params += d.dspark_head_hidden_dim * draft_positions + draft_positions
        dspark_overhead_bytes = min(
            d.dspark_max_extra_params,
            parallel_params + markov_params + confidence_params,
        ) * DTYPE_BYTES["fp16"]
    if d.tool_calling_enabled:
        need_params = 2 * m.d_model + m.d_model * d.tool_action_count + d.tool_action_count
        argument_params = 2 * m.d_model + m.d_model + 1
        query_params = 2 * m.d_model + m.d_model * d.tool_query_dim + d.tool_query_dim
        tool_calling_overhead_bytes = (need_params + argument_params + query_params) * DTYPE_BYTES["fp16"]
    if d.adaptive_deliberation_enabled:
        scheduler_params = (
            2 * m.d_model
            + m.d_model * d.deliberation_scheduler_hidden_dim
            + d.deliberation_scheduler_hidden_dim
            + d.deliberation_scheduler_hidden_dim
            + d.deliberation_scheduler_hidden_dim
            + d.deliberation_scheduler_hidden_dim * 1
            + 1
            + 2 * m.d_model
            + m.d_model * d.deliberation_level_count
            + d.deliberation_level_count
        )
        ambiguity_params = 2 * m.d_model + m.d_model + 1
        reward_params = 2 * m.d_model + m.d_model + 1
        deliberation_overhead_bytes = (scheduler_params + ambiguity_params + reward_params) * DTYPE_BYTES["fp16"]
        deliberation_cpu_ram_bytes = d.tree_search_state_budget_mb * 1024**2
    if d.metacognition_enabled:
        failure_gate_params = 3 * m.d_model + 1
        metacognition_overhead_bytes = failure_gate_params * DTYPE_BYTES["fp16"]
    if d.permanent_knowledge_enabled:
        pkb_disk_bytes = d.permanent_knowledge_max_files * d.permanent_knowledge_max_file_kb * 1024
        pkb_index_bytes = d.permanent_knowledge_max_files * 128 * DTYPE_BYTES["fp32"]
        pkb_cpu_ram_bytes = pkb_index_bytes + d.knowledge_summarizer_params + d.knowledge_policy_head_params
    if d.anti_hallucination_enabled:
        scorer_params = 2 * m.d_model + 1
        task_type_params = d.alignment_task_type_count * m.d_model
        token_bias_bytes = m.vocab_size * DTYPE_BYTES["fp32"]
        intent_vector_bytes = m.d_model * DTYPE_BYTES["fp16"]
        alignment_overhead_bytes = (scorer_params + task_type_params) * DTYPE_BYTES["fp16"]
        alignment_overhead_bytes += token_bias_bytes + intent_vector_bytes
    if d.long_context_enabled:
        long_tokens = min(d.long_context_tokens, m.max_seq_len)
        level1 = math.ceil(long_tokens / max(1, d.long_context_chunk_tokens))
        level2 = math.ceil(long_tokens / max(1, d.long_context_level2_span_tokens))
        hra_level1_cpu_bytes = level1 * m.d_model * 2
        hra_level2_gpu_bytes = level2 * m.d_model * 2
        hra_transient_kv_bytes = d.long_context_top_k_chunks * d.long_context_chunk_tokens * m.d_model * 2
    resident_vram = (
        hot_bytes
        + control_bytes
        + kv_bytes
        + cold_kv_bytes
        + hra_level2_gpu_bytes
        + alignment_overhead_bytes
        + metacognition_overhead_bytes
        + deliberation_overhead_bytes
        + dspark_overhead_bytes
        + tool_calling_overhead_bytes
    ) / 1024**3
    peak_vram = (
        hot_bytes
        + control_bytes
        + kv_bytes
        + cold_kv_bytes
        + hra_level2_gpu_bytes
        + alignment_overhead_bytes
        + metacognition_overhead_bytes
        + deliberation_overhead_bytes
        + dspark_overhead_bytes
        + tool_calling_overhead_bytes
        + hra_transient_kv_bytes
        + cold_active_bytes
        + activation_bytes
    ) / 1024**3
    cold_storage_bytes = cold_params * (0.5 if o.quantized_cold else DTYPE_BYTES.get(o.cold_dtype, 2.0))
    pkb_disk_gb = pkb_disk_bytes / 1024**3
    pkb_cpu_ram_gb = pkb_cpu_ram_bytes / 1024**3
    deliberation_cpu_ram_gb = deliberation_cpu_ram_bytes / 1024**3
    disk_gb = cold_storage_bytes / 1024**3 + persistent_memory_disk_gb + pkb_disk_gb
    cpu_ram_gb = (
        (0.5 if o.lazy_cold_blocks else disk_gb)
        + hra_level1_cpu_bytes / 1024**3
        + pkb_cpu_ram_gb
        + deliberation_cpu_ram_gb
    )
    notes = []
    if not o.lazy_cold_blocks and disk_gb > o.ram_budget_gb * 0.7:
        notes.append("cold shell should be lazy/NVMe-backed; CPU RAM budget is too small")
    if not o.quantized_cold and disk_gb > o.ram_budget_gb:
        notes.append("FP16 cold shell exceeds 16GB RAM; use quantized_cold or smaller config")
    if o.cold_granularity != "unit" and peak_vram > o.vram_budget_gb:
        notes.append("block-level cold loading exceeds budget; use cold_granularity=unit")
    elif peak_vram > o.vram_budget_gb:
        notes.append("estimated peak VRAM exceeds budget; reduce max_gpu_cold_blocks/seq_len/d_model")
    return MemoryReport(
        hot_params=int(hot_params),
        cold_params=int(cold_params),
        lora_params=int(lora_params),
        hot_kv_gb=kv_bytes / 1024**3,
        cold_kv_gb=cold_kv_bytes / 1024**3,
        hra_level1_cpu_gb=hra_level1_cpu_bytes / 1024**3,
        hra_level2_gpu_gb=hra_level2_gpu_bytes / 1024**3,
        hra_transient_kv_gb=hra_transient_kv_bytes / 1024**3,
        persistent_memory_disk_gb=persistent_memory_disk_gb,
        alignment_overhead_gb=alignment_overhead_bytes / 1024**3,
        metacognition_overhead_gb=metacognition_overhead_bytes / 1024**3,
        pkb_cpu_ram_gb=pkb_cpu_ram_gb,
        pkb_disk_gb=pkb_disk_gb,
        deliberation_overhead_gb=deliberation_overhead_bytes / 1024**3,
        deliberation_cpu_ram_gb=deliberation_cpu_ram_gb,
        dspark_overhead_gb=dspark_overhead_bytes / 1024**3,
        tool_calling_overhead_gb=tool_calling_overhead_bytes / 1024**3,
        resident_vram_gb=resident_vram,
        peak_vram_gb=peak_vram,
        cpu_ram_gb=cpu_ram_gb,
        disk_gb=disk_gb,
        fits_vram=peak_vram <= o.vram_budget_gb,
        fits_ram=cpu_ram_gb <= o.ram_budget_gb,
        notes=notes,
    )
