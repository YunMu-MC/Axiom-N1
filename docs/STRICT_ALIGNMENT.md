# Strict DoAP V2 Alignment Notes

This file records the implementation deltas made to move the codebase closer
to `doap_v2_paper.pdf`.

## Tightened

- An anti-hallucination Intent-Implementation Alignment Gate now compares
  pooled user intent against the implementation window, emits an alignment
  score, and can softly penalize drift-biased logits. The 64B config enables
  it with threshold 0.30 and a 512-token implementation window.

- The 64B config now enables the supplement's 258K context path via
  Hierarchical Retrieval Attention (512-token chunks, 4096-token hot window,
  top-4 transient cold chunks) plus a 10GB persistent memory stream.
- HRA and persistent-memory text ranking use a local Rust backend
  (`rust/dopa_core`) with Python/PyTorch wrappers for model integration.
- CPU text preprocessing now uses the same Rust core for UTF-8 byte
  tokenization and skeleton byte encoding when the release binary is present,
  while retaining Python fallback behavior for portability.
- Difficulty routing is token-level: `DifficultyGate` now returns `[B, T]`
  scores instead of a single last-token score.
- Coarse cold routing can consume `[B, T, blocks]` weights, so cold
  contribution can be masked per token.
- Fine-grained cold routing aggregates over active hard tokens when selecting
  cold attention heads and FFN sub-blocks.
- LoRA Bank is now variable-rank and owns multiple target-site factor sets per
  module. The 64B config sets `lora_target_sites: 24`, giving about 201M LoRA
  parameters, close to the paper's bank-size accounting.
- Shadow parameters now use Fisher-proxy mask initialization and fake-INT8
  straight-through dequantization, with `shadow_int8` and `shadow_scale`
  buffers for runtime/storage accounting.
- Stage aliases now map the original runnable names to paper-style names:
  `structural_reconstruction`, `hyper_lora`, `isg_training`,
  `cognitive_search`.
- A formal `model/cognitive.py` boundary now exists for ISG and curiosity
  search/distillation integration.

- Self-Driven Inductive Training is wired as `stage_sdit`, with
  consistency, rule-transfer, and reverse-prediction losses.
- A metacognition failure gate reads Hot Core hidden states and exposes
  `failure_probability` / `failure_overconfidence` for post-hoc learning.
- Stage 5 knowledge-management training has a 5-way operation policy
  head for `[KN_CREATE]`, `[KN_UPDATE]`, `[KN_MERGE]`, `[KN_DELETE]`, and
  `[KN_QUERY]`.
- A permanent JSON knowledge base implements `[KN_CREATE]`, `[KN_UPDATE]`,
  `[KN_MERGE]`, `[KN_DELETE]`, and `[KN_QUERY]` with 128-dim embeddings,
  top-3 retrieval, 500-file / 20KB-per-file limits, and failure-driven
  knowledge creation.

- Adaptive Deliberation adds a five-level metacognitive scheduler
  (`low`, `medium`, `high`, `ultra`, `x_open`), a token-distribution ambiguity
  detector, a process reward head, CPU-side thought landmarks, and a serial
  tree-search state container. Training uses the separate `stage_deliberation`
  alias so existing `stage2_5` intent-alignment behavior remains intact.

- DSpark speculative decoding is integrated as a Hot-Core-mounted
  semi-autoregressive drafter: a parallel draft head proposes `gamma - 1`
  tokens, a low-rank Markov head injects local dependency correction, a
  confidence head predicts per-position prefix survival, and a load-aware
  scheduler trims verification length before Cold Shell validation. Training is
  isolated behind `stage_dspark`.

- Tool-call learning is now represented inside the model with a dedicated
  `ToolCallingHeads` module: a tool-need/action classifier, an argument-validity
  verifier, and a normalized tool-retrieval query embedding head. Training is
  isolated behind `stage_tool_calling` and its schema/retrieval/rollout aliases.

## Still Not Fully Paper-Exact

- Hot Core ShadowLinear base weights are now stored as packed INT4 buffers.
  Forward execution still dequantizes before PyTorch linear rather than using a
  fused INT4 GEMM kernel, so this is storage-accurate but not a full GPTQ matmul
  runtime yet.
- LoRA factors are counted per target site, but the current runtime still
  injects a merged hidden-space delta instead of applying separate LoRA updates
  directly inside each Hot Core Q/K/V/O/FFN matrix.
- The LoRA Bank and hypernetwork still move with `model.to(device)` in normal
  PyTorch execution. The paper's CPU-resident bank plus tiny generated GPU
  expert path remains a future runtime optimization.
- PyTorch/Triton tensor kernels remain in Python-bound modules because moving
  GPU graph execution to a Rust CLI would add process overhead and break
  autograd; Rust is currently reserved for CPU text/retrieval hot paths.
- The Agent runtime, tool executors, permissions, and audit logs remain a
  separate project boundary. This repository trains the model to produce and
  recover from tool calls; it does not embed the runtime sandbox itself.
- ISG and cognitive search now have interfaces, but no autonomous web retrieval
  policy is executed during training by default.
- Shadow masks use base-weight magnitude as a Fisher proxy. True Fisher
  selection requires accumulating task gradients/statistics before mask
  materialization.
