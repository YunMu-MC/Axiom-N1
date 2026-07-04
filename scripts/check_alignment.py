from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.shadow import ShadowLinear
from dopa_coder_n1.model.rust_core import RustCoreBackend
from dopa_coder_n1.utils.memory import estimate_memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Check implementation alignment with DoAP V2 paper invariants.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "coder_n1_64b.yaml"))
    parser.add_argument(
        "--runtime-config",
        default=str(ROOT / "configs" / "tiny_unit.yaml"),
        help="Small config used for real module instantiation checks.",
    )
    args = parser.parse_args()

    cfg = DOPAConfig.from_yaml(args.config)
    runtime_cfg = DOPAConfig.from_yaml(args.runtime_config)
    runtime_model = DOPACoderN1(runtime_cfg)
    hot_shadows = [m for m in runtime_model.hot_layers.modules() if isinstance(m, ShadowLinear)]
    mem = estimate_memory(cfg)
    lora_target_sites = cfg.dopa.lora_target_sites or cfg.model.hot_layers * 6
    rust_core = RustCoreBackend.default()
    rust_core_available = rust_core.available()
    rust_tokenizer_ok = False
    rust_skeleton_ok = False
    if rust_core_available:
        try:
            rust_tokenizer_ok = rust_core.encode_bytes("abc", add_bos=True, add_eos=True) == [1, 101, 102, 103, 2]
            with tempfile.TemporaryDirectory() as tmp:
                skeleton_probe = Path(tmp) / "skeleton.json"
                skeleton_probe.write_text('{"kind":"probe"}', encoding="utf-8")
                rust_skeleton_ok = len(
                    rust_core.encode_skeleton_json(skeleton_probe, vocab_size=512, max_len=8)
                ) == 8
        except Exception:
            rust_tokenizer_ok = False
            rust_skeleton_ok = False
    report = {
        "config": str(Path(args.config).resolve()),
        "runtime_config": str(Path(args.runtime_config).resolve()),
        "depth_ok": cfg.model.hot_layers == 3 and cfg.model.cold_layers == 93,
        "fine_grained_cold_ok": cfg.offload.cold_granularity == "unit",
        "hot_shadow_count_runtime": len(hot_shadows),
        "hot_base_int4_runtime": bool(hot_shadows) and all(m.base_quantization == "int4" for m in hot_shadows),
        "shadow_fake_int8_runtime": bool(hot_shadows) and all(m.fake_int8 for m in hot_shadows),
        "lora_params": mem.lora_params,
        "lora_target_sites": lora_target_sites,
        "attention_backend": cfg.model.attention_backend,
        "peak_vram_gb": mem.peak_vram_gb,
        "fits_vram": mem.fits_vram,
        "long_context_ok": cfg.model.max_seq_len == 258_000 and cfg.dopa.long_context_enabled and cfg.dopa.long_context_top_k_chunks == 4,
        "persistent_memory_ok": cfg.dopa.persistent_memory_enabled and cfg.dopa.persistent_memory_budget_gb <= 10.0,
        "rust_core_ok": rust_core_available,
        "rust_tokenizer_ok": rust_tokenizer_ok,
        "rust_skeleton_ok": rust_skeleton_ok,
        "anti_hallucination_ok": cfg.dopa.anti_hallucination_enabled and cfg.dopa.alignment_threshold == 0.30 and cfg.dopa.alignment_window_tokens == 512,
        "self_driven_learning_ok": (
            cfg.dopa.self_driven_induction_enabled
            and cfg.dopa.sdit_consistency_weight > 0
            and cfg.dopa.sdit_transfer_weight > 0
            and cfg.dopa.sdit_reverse_weight > 0
        ),
        "metacognition_ok": cfg.dopa.metacognition_enabled and 0 < mem.metacognition_overhead_gb < 0.0001,
        "permanent_knowledge_ok": (
            cfg.dopa.permanent_knowledge_enabled
            and cfg.dopa.permanent_knowledge_max_files == 500
            and cfg.dopa.permanent_knowledge_max_file_kb == 20
            and mem.pkb_disk_gb <= 0.01
            and mem.pkb_cpu_ram_gb < 0.01
        ),
        "knowledge_policy_head_runtime": hasattr(runtime_model, "knowledge_policy_head"),
        "adaptive_deliberation_ok": (
            cfg.dopa.adaptive_deliberation_enabled
            and cfg.dopa.deliberation_level_count == 5
            and cfg.dopa.thought_landmark_max_count == 10
            and cfg.dopa.thought_landmark_max_tokens == 50
            and cfg.dopa.tree_search_state_budget_mb <= 100.0
            and 0 < mem.deliberation_overhead_gb < 0.004
            and 0 < mem.deliberation_cpu_ram_gb <= 0.10
        ),
        "adaptive_deliberation_runtime": (
            hasattr(runtime_model, "deliberation_scheduler")
            and hasattr(runtime_model, "ambiguity_detector")
            and hasattr(runtime_model, "process_reward_head")
        ),
        "dspark_ok": (
            cfg.dopa.dspark_enabled
            and cfg.dopa.dspark_gamma == 7
            and cfg.dopa.dspark_markov_rank <= 16
            and cfg.dopa.dspark_max_extra_params <= 20_000_000
            and 0 < mem.dspark_overhead_gb < 0.04
        ),
        "dspark_runtime": hasattr(runtime_model, "dspark_heads") and hasattr(runtime_model, "dspark_scheduler"),
        "tool_calling_ok": (
            cfg.dopa.tool_calling_enabled
            and cfg.dopa.tool_action_count >= 8
            and cfg.dopa.tool_query_dim >= 64
            and 0 < mem.tool_calling_overhead_gb < 0.01
        ),
        "tool_calling_runtime": hasattr(runtime_model, "tool_calling_heads"),
    }
    report["strict_core_ok"] = (
        report["depth_ok"]
        and report["fine_grained_cold_ok"]
        and report["hot_base_int4_runtime"]
        and report["shadow_fake_int8_runtime"]
        and report["lora_params"] >= 100_000_000
        and report["fits_vram"]
        and report["long_context_ok"]
        and report["persistent_memory_ok"]
        and report["rust_core_ok"]
        and report["rust_tokenizer_ok"]
        and report["rust_skeleton_ok"]
        and report["anti_hallucination_ok"]
        and report["self_driven_learning_ok"]
        and report["metacognition_ok"]
        and report["permanent_knowledge_ok"]
        and report["knowledge_policy_head_runtime"]
        and report["adaptive_deliberation_ok"]
        and report["adaptive_deliberation_runtime"]
        and report["dspark_ok"]
        and report["dspark_runtime"]
        and report["tool_calling_ok"]
        and report["tool_calling_runtime"]
    )
    print(json.dumps(report, indent=2))
    if not report["strict_core_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
