# Axiom N1

Axiom N1 is a full engineering implementation of the DOPA Coder N1 architecture:
a dual-stream, on-demand-parameter code model trained from scratch.

It is not a fine-tuning wrapper. It does not load a pretrained LLM. The same model code
supports tiny smoke tests, single-GPU development runs, and a 64B-style configuration.

## Code-Only Release Boundary

This GitHub repository is prepared for code-only publication. Local dialogue corpora,
API-generated distillation records, checkpoints, run folders, downloaded wheels, caches,
and private `.env` files are excluded by `.gitignore`.

The Python import package remains `dopa_coder_n1` so existing scripts and tests keep their
stable module path while the public project name is Axiom N1.

## What Is Implemented

- Causal Transformer hot core with RMSNorm, RoPE, grouped-query attention, and SwiGLU.
- Cold shell partitioned into demand-loaded cold blocks.
- Fine-grained Cold Shell executes selected attention/FFN units in cold-layer order.
- Incremental Hot Core decoding with optional 4-bit packed sliding-window KV cache.
- Incremental fine-grained cold attention heads with a small 4-bit selective KV cache.
- Formal attention backend boundary for `torch`, `int4_reference`, and `triton_int4` packed-KV execution.
- Automatic shadow-parameter injection with separate hot/cold densities.
- CPU-offloaded AdamW optimizer state for local low-VRAM runs.
- Difficulty gate and layer demand predictor for sparse cold block activation.
- Hot/cold gated fusion.
- Structured skeleton compiler.
- Hypernetwork that compiles skeleton embeddings into sparse LoRA-bank coefficients.
- Shared LoRA bank with top-k module materialization.
- Curiosity gate and session-only rank-1 fast weights for fleeting knowledge.
- Shadow linear injection for sparse additive delta training.
- Stage-aware training losses for Stage 1, Stage 2, Stage 3, and Stage 3.5.
- Byte tokenizer and packed code dataset for true from-scratch training.
- Rust-accelerated CPU text paths for retrieval ranking, UTF-8 byte
  tokenization, and skeleton byte encoding, with Python fallbacks.
- Tool-call learning heads for Agent runtimes: tool-need classification,
  argument-validity checking, and tool-registry retrieval queries.
- Train, generate, inspect, and toy-data scripts.

## Important Boundary

This repo is the complete implementation scaffold. Training a real 64B model from scratch
still requires:

- a large code corpus,
- a real training cluster or a long-running offload setup,
- many checkpoints,
- CUDA/Triton runtime support for the fastest kernel path,
- careful evaluation and safety filtering.

The included `configs/coder_n1_64b.yaml` expresses the paper-scale target: 3 hot
layers plus 93 cold layers, about 2B hot parameters and 62B cold parameters.
The `configs\tiny.yaml` and `configs\tiny_unit.yaml` files exist only to verify
the exact same code path on local hardware.

## Quick Start

Install dependencies in your Python environment:

```powershell
cd axiom-n1
python -m venv .venv
$env:PIP_CACHE_DIR=".pip-cache"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

Or use the bundled installer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_deps.ps1
powershell -ExecutionPolicy Bypass -File scripts\install_deps.ps1 -Cuda
```

Create toy data and run a short training job:

```powershell
.\.venv\Scripts\python.exe scripts\make_toy_data.py
.\.venv\Scripts\python.exe scripts\train.py --config configs\tiny.yaml --out-dir runs\tiny
.\.venv\Scripts\python.exe scripts\generate.py --checkpoint runs\tiny\last.pt --prompt "def solve():`n    "
```

Run the formal staged trainer and short evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\train_stages.py --config configs\tiny_unit.yaml --out-dir runs\staged_tiny --stages stage1,stage2 --steps 1,1 --device cpu
.\.venv\Scripts\python.exe scripts\evaluate.py --checkpoint runs\staged_tiny\stage2\last.pt --data data\toy.jsonl --max-batches 2 --out runs\staged_tiny\eval.json
```

Run the final end-to-end local pipeline:

```powershell
.\.venv\Scripts\python.exe scripts\final_train.py --config configs\tiny_unit.yaml --out-dir runs\final_tiny --stages stage1,stage2 --steps 1,1 --device cpu
```

For real data, prepare a train/valid split first or let `final_train.py` do it:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --input PATH\TO\code_corpus --out-dir data\prepared --valid-ratio 0.02
.\.venv\Scripts\python.exe scripts\final_train.py --config configs\local_8gb_16gb.yaml --raw-data PATH\TO\code_corpus --out-dir runs\local_final --stages stage1,stage2,stage3,stage3_5 --steps 1000,1000,1000,1000
```

After PyTorch is installed, this single command verifies forward, backward, generation,
and checkpoint save:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_torch.py
```

Check whether the final int4 kernel backend can run on the current machine:

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe scripts\check_kernel.py --config configs\tiny_unit.yaml
```

Build the Rust CPU core used by retrieval and preprocessing fast paths:

```powershell
cd rust\dopa_core
cargo build --release
cd ..\..
.\.venv\Scripts\python.exe scripts\check_alignment.py --config configs\coder_n1_64b.yaml
```

Inspect parameter groups:

```powershell
python scripts\inspect_model.py --config configs\tiny.yaml
python scripts\inspect_model.py --config configs\coder_n1_64b.yaml
```

Estimate whether a config fits an 8GB VRAM / 16GB RAM machine:

```powershell
.\.venv\Scripts\python.exe scripts\estimate_memory.py --config configs\local_8gb_16gb.yaml
```

Run the fine-grained cold-shell path:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\tiny_unit.yaml --out-dir runs\tiny_unit
.\.venv\Scripts\python.exe scripts\export_cold_blocks.py --config configs\tiny_unit.yaml --out-dir checkpoints\tiny_units --int4
```

Exporting `configs\coder_n1_64b.yaml` in int4 format is the full target path and
requires roughly 29GB of SSD space for cold-unit files.

After training materialized fine-grained cold units, write them back to SSD:

```powershell
.\.venv\Scripts\python.exe scripts\writeback_cold_units.py --checkpoint runs\staged_tiny\stage2\last.pt --out-dir checkpoints\cold_shell_local --format int4
```

Use incremental decoding:

```powershell
.\.venv\Scripts\python.exe scripts\generate.py --checkpoint runs\tiny_unit\last.pt --prompt "def solve():" --incremental
```

`configs\tiny_unit.yaml`, `configs\local_8gb_16gb.yaml`, and
`configs\coder_n1_64b.yaml` enable `dopa.hot_kv_int4: true` and
`dopa.hot_kv_summary: true`, so the persistent Hot Core cache is stored as packed
int4 between incremental steps and evicted tokens are folded into one EMA summary
KV token per layer.

Fine-grained cold attention heads also maintain a bounded packed int4 cache for
the most recently selected cold units, controlled by `dopa.cold_kv_window` and
`dopa.cold_kv_hot_units`.

The packed-KV attention path is routed through `model.attention_backend`.
`triton_int4` is the final configured backend for packed int4 KV. On a CUDA/Triton
machine it uses a Triton kernel to dequantize packed K/V tensors before PyTorch SDPA.
On CPU-only or no-Triton environments it automatically falls back to `int4_reference`,
which keeps the same model path runnable and testable.

## Training Stages

Stage 1: language surface modeling. The hot core and LM head learn next-token prediction.

Stage 2: structural reconstruction. Gates and cold routing learn when deeper capacity is
needed while penalizing unnecessary cold usage.

Stage 3: skeleton/hypernetwork/LoRA-bank meta-training. Skeleton embeddings compile into
sparse specialist parameters.

Stage 3.5: inductive skeleton and curiosity calibration. The model learns when it lacks
knowledge and how to form temporary experts.

Stage Tool Calling: schema following, tool retrieval, and rollout supervision.
Use JSONL fields `tool_need_label`, `tool_argument_valid_label`, and
`tool_query_target` to train the tool-call heads. The runtime sandbox and tool
executors are expected to live in a separate Agent project.

## Data Format

Plain `.txt`, `.py`, and `.md` files are accepted.

For skeleton-aware training, use JSONL:

```json
{"text": "def solve():\n    ...", "skeleton": {"name": "grid_bfs", "steps": [{"op": "graph_search"}]}}
```

## Project Layout

- `src/dopa_coder_n1/model/dopa.py` - full DOPA model assembly.
- `src/dopa_coder_n1/model/layers.py` - Transformer primitives.
- `src/dopa_coder_n1/model/cold_offload.py` - cold shell manager.
- `src/dopa_coder_n1/model/lora_bank.py` - LoRA bank and hypernetwork.
- `src/dopa_coder_n1/model/skeleton.py` - skeleton tokenizer/compiler.
- `src/dopa_coder_n1/model/shadow.py` - shadow parameter injection.
- `src/dopa_coder_n1/training` - optimizer, staged losses, checkpointing.
- `scripts` - train/generate/inspect helpers.
- `configs` - tiny, dev, and 64B-style configs.
