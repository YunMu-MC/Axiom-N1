from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.training.validation import validate_training_config


def test_load_tiny_config():
    cfg = DOPAConfig.from_yaml("configs/tiny.yaml")
    assert cfg.model.d_model == 128
    assert cfg.dopa.lora_modules == 8


def test_64b_target_config_matches_paper_depth():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    assert cfg.model.hot_layers == 3
    assert cfg.model.cold_layers == 93
    assert cfg.model.hot_layers + cfg.model.cold_layers == 96
    assert cfg.model.d_model == 8192
    assert cfg.model.n_heads == 64
    assert cfg.model.n_kv_heads == 64
    assert cfg.model.attention_backend == "triton_int4"
    assert cfg.dopa.hot_kv_int4
    assert cfg.dopa.hot_kv_summary


def test_validation_rejects_unknown_attention_backend():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    cfg.model.attention_backend = "missing_kernel"
    errors = validate_training_config(cfg, require_data=False)
    assert any("unknown attention_backend" in error for error in errors)



def test_64b_config_enables_258k_context_and_10gb_memory_stream():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    assert cfg.model.max_seq_len == 258_000
    assert cfg.dopa.long_context_enabled
    assert cfg.dopa.long_context_tokens == 258_000
    assert cfg.dopa.long_context_chunk_tokens == 512
    assert cfg.dopa.long_context_top_k_chunks == 4
    assert cfg.dopa.persistent_memory_enabled
    assert cfg.dopa.persistent_memory_budget_gb == 10.0



def test_64b_config_enables_anti_hallucination_iia_gate():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    assert cfg.dopa.anti_hallucination_enabled
    assert cfg.dopa.alignment_threshold == 0.30
    assert cfg.dopa.alignment_window_tokens == 512
    assert cfg.dopa.alignment_soft_strength > 0.0



def test_64b_config_enables_self_driven_learning_and_pkb():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    assert cfg.dopa.self_driven_induction_enabled
    assert cfg.dopa.metacognition_enabled
    assert cfg.dopa.permanent_knowledge_enabled
    assert cfg.dopa.permanent_knowledge_max_files == 500
    assert cfg.dopa.permanent_knowledge_max_file_kb == 20



def test_64b_config_enables_adaptive_deliberation():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    assert cfg.dopa.adaptive_deliberation_enabled
    assert cfg.dopa.deliberation_level_count == 5
    assert cfg.dopa.thought_landmark_max_count == 10
    assert cfg.dopa.thought_landmark_max_tokens == 50
    assert cfg.dopa.tree_search_state_budget_mb <= 100.0



def test_64b_config_enables_dspark_speculative_decoding():
    cfg = DOPAConfig.from_yaml("configs/coder_n1_64b.yaml")
    assert cfg.dopa.dspark_enabled
    assert cfg.dopa.dspark_gamma == 7
    assert cfg.dopa.dspark_markov_rank <= 16
    assert cfg.dopa.dspark_max_extra_params <= 20_000_000
