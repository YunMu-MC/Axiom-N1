# DOPA Coder N1 架构说明

这个项目是完整工程实现，不是只写一个缩小实验模型。`tiny.yaml` 只是为了本地验证同一套代码路径能运行，真正扩大模型时不需要换代码，只需要换配置、数据和训练环境。

## 模块对应关系

- Hot Core：`DOPACoderN1.hot_layers`，标准 causal Transformer。
- Cold Shell：`ColdBlockManager` 管理 `ColdBlock`，支持 CPU 驻留、按需搬运到 GPU。
- Difficulty Gate：根据 hot hidden state 输出当前 token 的难度。
- Layer Demand Predictor：输出 cold block 的 sparse top-k 权重。
- Gated Fusion：融合 hot hidden 和 cold hidden。
- Skeleton Compiler：把结构化 skeleton JSON 编译成向量。
- Hypernetwork：把 skeleton 向量编译成 LoRA bank 的 sparse 系数。
- LoRA Bank：共享低秩专家库，按 top-k 系数临时组合。
- Fast Weights：会话级临时 rank-1 参数，用于搜索/临时知识注入。
- Shadow Parameters：冻结 base linear，仅训练稀疏 additive delta。

## 从 0 训练路线

1. Stage 1：用大规模代码语料训练 hot core 的语言能力。
2. Stage 2：引入 cold shell，训练 difficulty gate、layer demand predictor 和 fusion。
3. Stage 3：构建 skeleton 数据集，训练 skeleton compiler、hypernetwork 和 LoRA bank。
4. Stage 3.5：加入执行反馈、失败修复、curiosity gate 和临时专家。
5. Scale-up：启用 offload、gradient checkpointing、混合精度、分块 checkpoint。

## 关于 64B

`configs/coder_n1_64b.yaml` 表达论文中的目标结构：

- 3 层 hot core，约 2B 参数
- 93 层 cold shell，约 62B 参数
- 总深度 96 层，cold shell 每 2 层一个训练/采样块
- d_model=8192，64 个 attention heads，FFN multiplier=2.0
- 64 个 LoRA 模块
- top-k cold block 和 top-k LoRA 激活

但“完整实现”和“已经训练完成 64B 权重”是两件事。这里交付的是完整工程代码；训练出可用 64B 权重需要真实语料、长期训练、显存/内存/硬盘 IO 工程和评测闭环。

## 8GB 显存 / 16GB 运存约束

你的机器预算更接近 `configs/local_8gb_16gb.yaml`。这份配置启用：

- `lazy_cold_blocks: true`，启动时不实例化完整 cold shell。
- `quantized_cold: true`，冷块按量化存储估算。
- `cold_granularity: unit`，冷壳按 attention head / FFN sub-block 粒度加载。
- `cold_attention_heads_per_step: 12` 和 `cold_ffn_blocks_per_step: 6`，对应补充 A 的训练预算。
- `num_workers: 0`，避免 DataLoader 额外复制内存。
- `max_seq_len: 2048`，先控制 KV cache 和 activation。

论文第 7.3 节写 cold shell FP16 约 124GB，这不可能放进 16GB 运存。要在这台机器上接近论文路线，必须走 NVMe/磁盘冷块流式加载，不能让 cold shell 常驻 RAM。

检查预算：

```powershell
.\.venv\Scripts\python.exe scripts\estimate_memory.py --config configs\local_8gb_16gb.yaml
.\.venv\Scripts\python.exe scripts\estimate_memory.py --config configs\coder_n1_64b.yaml
```

正规阶段训练：

```powershell
.\.venv\Scripts\python.exe scripts\train_stages.py --config configs\tiny_unit.yaml --out-dir runs\staged_tiny --stages stage1,stage2 --steps 1,1 --device cpu
```

最终一键训练流水线：

```powershell
.\.venv\Scripts\python.exe scripts\final_train.py --config configs\tiny_unit.yaml --out-dir runs\final_tiny --stages stage1,stage2 --steps 1,1 --device cpu
```

短期测评：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --checkpoint runs\staged_tiny\stage2\last.pt --data data\toy.jsonl --max-batches 2 --out runs\staged_tiny\eval.json
```

导出冷块：

```powershell
.\.venv\Scripts\python.exe scripts\export_cold_blocks.py --config configs\local_8gb_16gb.yaml --out-dir checkpoints\cold_shell_local --int4
```

训练后把已 materialized 的 cold unit 回写：

```powershell
.\.venv\Scripts\python.exe scripts\writeback_cold_units.py --checkpoint runs\staged_tiny\stage2\last.pt --out-dir checkpoints\cold_shell_local --format int4
```

## 补充 A/B 对应到代码

- attention head unit / FFN sub-block：`src/dopa_coder_n1/model/fine_cold.py`
- SSD lazy unit store：`ColdUnitStore`
- fine-grained LDP：`FineGrainedLayerDemandPredictor`，先选 cold layer，再在层内选择 head/FFN unit，并用 `cold_coverage_penalty` 做访问均衡。
- 顺序 Cold Shell：`FineGrainedColdShell.forward()` 会按 cold layer 从小到大执行稀疏 attention/FFN 残差，而不是把冷单元当全局专家池求和。
- Stage 2 sparse cold usage penalty：fine-grained cold selection 的 softmax weights 会进入 `training/stages.py`
- int4/int8 unit 导出/加载：`scripts/export_cold_blocks.py`，论文路径推荐 `--int4`
- Hot Core 4-bit sliding-window KV + summary token：`src/dopa_coder_n1/model/kv_cache.py`，增量解码时可通过 `dopa.hot_kv_int4` 持久保存 packed int4 cache，通过 `dopa.hot_kv_summary` 把被驱逐 token 融合进 EMA summary。
- Cold selective KV cache：`ColdSelectiveKVState`，增量解码时为最近被选中的 cold attention head unit 保留 packed int4 窗口。
- 正规 attention backend：`src/dopa_coder_n1/model/attention_backend.py`，通过 `model.attention_backend` 选择 `torch`、`int4_reference` 或 `triton_int4`。最终配置使用 `triton_int4`：有 CUDA/Triton 时用 `src/dopa_coder_n1/model/triton_int4.py` 里的 int4 KV dequant kernel，没有 kernel 环境时自动回退到 `int4_reference`。
- 旋转 Shadow Mask：`rotate_shadow_masks()`
- 自动 Shadow 参数注入：`inject_dopa_shadow_linears()` 会按 `hot_train_density` / `cold_train_density` 区分 hot/cold 密度，lazy cold unit 创建时也会注入。
- CPU optimizer offload：`CPUAdamW` 把 Adam 动量和 master 参数放在 CPU。
- 正规 staged runner：`scripts/train_stages.py` 串联 P1/P2/Stage3/Stage3.5 风格阶段，并写 `metrics.jsonl` 和 `stages_summary.json`。
- 最终流水线：`scripts/final_train.py` 串联数据准备、分阶段训练、短期测评、cold unit 回写和 Markdown/JSON final report。
- 短期测评：`scripts/evaluate.py` 输出 loss、perplexity、difficulty、cold routing 统计和生成样例。
- cold unit 回写：`scripts/writeback_cold_units.py` 把 checkpoint 中已加载/训练过的 fine-grained unit 写回 int4/int8/fp32 文件。

当前 `generate()` 已支持 `--incremental` 增量解码，Hot Core 的 KV cache 会按 `dopa.hot_kv_window` 裁剪，避免随生成长度无限增长。打开 `dopa.hot_kv_int4` 后，cache 在每步之间以 packed int4 驻留，进入 attention backend 前保持 packed 格式。打开 `dopa.hot_kv_summary` 后，窗口外历史被压入每层一个 summary KV token，并作为注意力前缀参与后续 token 计算。Cold attention head unit 在增量解码时会使用 `dopa.cold_kv_hot_units` 和 `dopa.cold_kv_window` 维护一个小型 packed int4 selective KV cache。当前最终 backend 已接入 `triton_int4` kernel 解包路径；后续如果要继续压性能，可以把 SDPA 部分进一步替换成 fused dequant attention kernel。
