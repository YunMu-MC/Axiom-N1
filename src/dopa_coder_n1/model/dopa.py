from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.model.alignment import IntentImplementationAlignmentGate
from dopa_coder_n1.model.cold_offload import ColdBlock, ColdBlockManager
from dopa_coder_n1.model.attention_backend import build_attention_backend
from dopa_coder_n1.model.deliberation import AdaptiveDeliberationScheduler, AmbiguityDetector
from dopa_coder_n1.model.dspark import DSparkHeads, VerificationScheduler
from dopa_coder_n1.model.fine_cold import FineGrainedColdShell, FineGrainedLayerDemandPredictor
from dopa_coder_n1.model.gates import CuriosityGate, DifficultyGate, HotColdFusion, LayerDemandPredictor
from dopa_coder_n1.model.layers import RMSNorm, TransformerBlock
from dopa_coder_n1.model.kv_cache import ColdSelectiveKVState, LayerKVCache, PackedLayerKV, pack_layer_kv
from dopa_coder_n1.model.metacognition import FailurePredictionGate
from dopa_coder_n1.model.lora_bank import HyperNetwork, LoRABank
from dopa_coder_n1.model.shadow import inject_dopa_shadow_linears
from dopa_coder_n1.model.skeleton import SkeletonBatch, SkeletonCompiler
from dopa_coder_n1.model.tool_calling import ToolCallingHeads

HotLayerCache = tuple[torch.Tensor, torch.Tensor] | LayerKVCache | PackedLayerKV


@dataclass
class DOPAOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    aux: dict[str, torch.Tensor] | None = None
    hot_kv_cache: list[HotLayerCache] | None = None
    cold_kv_cache: ColdSelectiveKVState | None = None


class DOPACoderN1(nn.Module):
    """Full DOPA Coder N1 architecture.

    This is the same code path for tiny local validation and large 64B-style configs:
    scale is controlled by config, not by switching to a different model.
    """

    def __init__(self, cfg: DOPAConfig):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        d = cfg.dopa
        n_kv = m.n_kv_heads
        self.attention_backend = build_attention_backend(m.attention_backend)
        self.token_embedding = nn.Embedding(m.vocab_size, m.d_model)
        self.use_fine_cold = cfg.offload.cold_granularity == "unit"
        self.hot_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=m.d_model,
                    n_heads=m.n_heads,
                    n_kv_heads=n_kv,
                    max_seq_len=m.max_seq_len,
                    ffn_multiplier=m.ffn_multiplier,
                    rope_theta=m.rope_theta,
                    dropout=m.dropout,
                    attention_backend=self.attention_backend,
                )
                for _ in range(m.hot_layers)
            ]
        )
        num_cold_blocks = math.ceil(m.cold_layers / m.cold_block_size) if m.cold_layers else 0

        def make_cold_block(block_index: int) -> ColdBlock:
            start = block_index * m.cold_block_size
            end = min(start + m.cold_block_size, m.cold_layers)
            layers = [
                TransformerBlock(
                    d_model=m.d_model,
                    n_heads=m.n_heads,
                    n_kv_heads=n_kv,
                    max_seq_len=m.max_seq_len,
                    ffn_multiplier=m.ffn_multiplier,
                    rope_theta=m.rope_theta,
                    dropout=m.dropout,
                    attention_backend=self.attention_backend,
                )
                for _ in range(start, end)
            ]
            return ColdBlock(layers)

        if cfg.offload.lazy_cold_blocks:
            cold_blocks: list[ColdBlock | None] = []
        else:
            cold_blocks = [make_cold_block(i) for i in range(num_cold_blocks)]
        active_device = cfg.offload.device
        cold_device = cfg.offload.cold_device if cfg.offload.enabled else cfg.offload.device
        if self.use_fine_cold:
            self.cold_manager = None
            self.fine_cold_shell = FineGrainedColdShell(
                d_model=m.d_model,
                n_heads=m.n_heads,
                cold_layers=m.cold_layers,
                max_seq_len=m.max_seq_len,
                ffn_multiplier=m.ffn_multiplier,
                ffn_subblocks=cfg.offload.cold_ffn_subblocks,
                rope_theta=m.rope_theta,
                dropout=m.dropout,
                checkpoint_dir=cfg.offload.cold_checkpoint_dir,
                cold_device=cold_device,
                active_device=active_device,
                storage_dtype=cfg.offload.cold_dtype,
                shadow_density=cfg.offload.cold_train_density,
                attention_backend=self.attention_backend,
            )
        else:
            self.fine_cold_shell = None
            self.cold_manager = ColdBlockManager(
                blocks=cold_blocks,
                block_factory=make_cold_block,
                num_blocks=num_cold_blocks,
                cold_device=cold_device,
                active_device=active_device,
                max_active=cfg.offload.max_gpu_cold_blocks,
                checkpoint_dir=cfg.offload.cold_checkpoint_dir,
                lazy=cfg.offload.lazy_cold_blocks,
                storage_dtype=cfg.offload.cold_dtype,
            )
        self.difficulty_gate = DifficultyGate(m.d_model)
        if self.use_fine_cold:
            self.layer_demand = FineGrainedLayerDemandPredictor(
                d_model=m.d_model,
                cold_layers=m.cold_layers,
                n_heads=m.n_heads,
                ffn_subblocks=cfg.offload.cold_ffn_subblocks,
                layer_budget=cfg.offload.cold_layer_budget_per_step,
                head_budget=cfg.offload.cold_attention_heads_per_step,
                ffn_budget=cfg.offload.cold_ffn_blocks_per_step,
                coverage_penalty=d.cold_coverage_penalty,
            )
        else:
            self.layer_demand = LayerDemandPredictor(m.d_model, num_cold_blocks, d.top_k_cold_blocks)
        self.fusion = HotColdFusion(m.d_model)
        self.skeleton_compiler = SkeletonCompiler(
            vocab_size=d.skeleton_vocab_size,
            skeleton_dim=d.skeleton_dim,
            layers=d.skeleton_layers,
        )
        self.hyper = HyperNetwork(
            skeleton_dim=d.skeleton_dim,
            hidden_dim=m.d_model,
            lora_modules=d.lora_modules,
            top_k=d.top_k_lora,
        )
        self.skeleton_to_model = nn.Linear(d.skeleton_dim, m.d_model)
        lora_target_sites = d.lora_target_sites or m.hot_layers * 6
        self.lora_bank = LoRABank(
            d.lora_modules,
            m.d_model,
            d.lora_rank,
            d.lora_alpha,
            target_sites=lora_target_sites,
        )
        self.curiosity_gate = CuriosityGate(m.d_model)
        self.failure_gate = FailurePredictionGate(m.d_model)
        self.knowledge_policy_head = nn.Sequential(
            nn.LayerNorm(m.d_model),
            nn.Linear(m.d_model, 5),
        )
        self.deliberation_scheduler = AdaptiveDeliberationScheduler(
            m.d_model,
            hidden_dim=d.deliberation_scheduler_hidden_dim,
            level_count=d.deliberation_level_count,
        )
        self.ambiguity_detector = AmbiguityDetector(m.d_model)
        self.process_reward_head = nn.Sequential(
            nn.LayerNorm(m.d_model),
            nn.Linear(m.d_model, 1),
        )
        self.dspark_heads = DSparkHeads(
            d_model=m.d_model,
            vocab_size=m.vocab_size,
            gamma=d.dspark_gamma,
            markov_rank=d.dspark_markov_rank,
            hidden_dim=d.dspark_head_hidden_dim,
        )
        self.dspark_scheduler = VerificationScheduler(
            gamma=d.dspark_gamma,
            min_verify_tokens=d.dspark_min_verify_tokens,
        )
        self.tool_calling_heads = ToolCallingHeads(
            d_model=m.d_model,
            action_count=d.tool_action_count,
            query_dim=d.tool_query_dim,
        )
        self.intent_alignment_gate = IntentImplementationAlignmentGate(
            m.d_model,
            m.vocab_size,
            threshold=d.alignment_threshold,
            soft_strength=d.alignment_soft_strength,
            window_tokens=d.alignment_window_tokens,
            task_type_count=d.alignment_task_type_count,
        )
        self.final_norm = RMSNorm(m.d_model)
        self.lm_head = nn.Linear(m.d_model, m.vocab_size, bias=False)
        if m.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.fast_u: nn.Parameter | None = None
        self.fast_v: nn.Parameter | None = None
        self.fast_scale = 0.0
        self.apply(self._init_weights)
        self.shadow_linears = inject_dopa_shadow_linears(
            self,
            hot_density=cfg.offload.hot_train_density,
            cold_density=cfg.offload.cold_train_density,
            mask_strategy=cfg.dopa.shadow_mask_strategy,
            fake_int8=cfg.dopa.shadow_fake_int8,
            hot_base_quantization=cfg.dopa.hot_base_quantization,
            cold_base_quantization=cfg.dopa.cold_base_quantization,
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_hot(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.hot_layers:
            if self.cfg.model.use_gradient_checkpointing and self.training:
                def layer_forward(y: torch.Tensor, active_layer: TransformerBlock = layer) -> torch.Tensor:
                    return active_layer(y, positions=positions)[0]

                x = checkpoint(layer_forward, x, use_reentrant=False)
            else:
                x, _ = layer(x, positions=positions)
        return x

    def _run_hot_incremental(
        self,
        x: torch.Tensor,
        position: int,
        kv_cache: list[HotLayerCache] | None = None,
    ) -> tuple[torch.Tensor, list[HotLayerCache]]:
        new_cache: list[HotLayerCache] = []
        pos = torch.full((x.size(1),), position, device=x.device, dtype=torch.long)
        for idx, layer in enumerate(self.hot_layers):
            layer_cache = kv_cache[idx] if kv_cache is not None and idx < len(kv_cache) else None
            x, cache = layer(x, positions=pos, kv_cache=layer_cache)
            previous = self.attention_backend.decode_cache(layer_cache, device=x.device, dtype=x.dtype)
            cache = self._trim_hot_cache(cache, previous=previous)
            if self.cfg.dopa.hot_kv_int4:
                new_cache.append(pack_layer_kv(cache))
            else:
                new_cache.append(cache)
        return x, new_cache

    def _trim_hot_cache(
        self,
        cache: tuple[torch.Tensor, torch.Tensor],
        previous: tuple[torch.Tensor, torch.Tensor] | LayerKVCache | None = None,
    ) -> LayerKVCache:
        window = self.cfg.dopa.hot_kv_window
        k, v = cache
        k_summary = previous.k_summary if isinstance(previous, LayerKVCache) else None
        v_summary = previous.v_summary if isinstance(previous, LayerKVCache) else None
        has_summary = isinstance(previous, LayerKVCache) and previous.has_summary
        if window > 0 and k.size(2) > window:
            overflow = k.size(2) - window
            if self.cfg.dopa.hot_kv_summary:
                evicted_k = k[:, :, :overflow].mean(dim=2, keepdim=True).detach()
                evicted_v = v[:, :, :overflow].mean(dim=2, keepdim=True).detach()
                if k_summary is None or v_summary is None or not has_summary:
                    k_summary = evicted_k
                    v_summary = evicted_v
                else:
                    decay = self.cfg.dopa.hot_kv_summary_decay
                    k_summary = decay * k_summary.to(evicted_k.device, evicted_k.dtype) + (1.0 - decay) * evicted_k
                    v_summary = decay * v_summary.to(evicted_v.device, evicted_v.dtype) + (1.0 - decay) * evicted_v
                has_summary = True
            k = k[:, :, -window:].contiguous()
            v = v[:, :, -window:].contiguous()
        elif not self.cfg.dopa.hot_kv_summary:
            k_summary = None
            v_summary = None
            has_summary = False
        return LayerKVCache(k=k, v=v, k_summary=k_summary, v_summary=v_summary, has_summary=has_summary)

    def compile_skeleton(
        self, skeleton: SkeletonBatch | None, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if skeleton is None:
            skel = torch.zeros(batch_size, self.cfg.dopa.skeleton_dim, device=device)
        else:
            token_ids = skeleton.token_ids.to(device)
            skel = self.skeleton_compiler(SkeletonBatch(token_ids=token_ids, adjacency=skeleton.adjacency))
        lora_coeffs, lora_scores = self.hyper(skel)
        return skel, lora_coeffs, lora_scores

    def apply_specialist(
        self,
        hidden: torch.Tensor,
        skel_embedding: torch.Tensor,
        lora_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        skel_bias = self.skeleton_to_model(skel_embedding).unsqueeze(1)
        hidden = hidden + skel_bias
        hidden = hidden + self.lora_bank(hidden, lora_coeffs)
        return hidden

    def encode_intent(self, intent_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(intent_ids.size(1), device=intent_ids.device)
        hidden = self.token_embedding(intent_ids) * math.sqrt(self.cfg.model.d_model)
        return self._run_hot(hidden, positions=positions)

    def apply_fast_weights(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.fast_u is None or self.fast_v is None or self.fast_scale == 0.0:
            return hidden
        v = self.fast_v.to(hidden.device, hidden.dtype)
        u = self.fast_u.to(hidden.device, hidden.dtype)
        return hidden + self.fast_scale * torch.einsum("btd,d,e->bte", hidden, v, u)

    def materialize_fast_weights(self, memory: torch.Tensor, scale: float = 0.05) -> None:
        """Install a session-only rank-r fast expert from a distilled memory vector."""
        if memory.ndim == 1:
            memory = memory.unsqueeze(0)
        vec = F.normalize(memory.mean(dim=0), dim=0)
        self.fast_u = nn.Parameter(vec.detach().clone(), requires_grad=False)
        self.fast_v = nn.Parameter(vec.detach().clone(), requires_grad=False)
        self.fast_scale = scale

    def clear_fast_weights(self) -> None:
        self.fast_u = None
        self.fast_v = None
        self.fast_scale = 0.0

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        skeleton: SkeletonBatch | None = None,
        force_cold: bool = False,
        return_aux: bool = False,
        intent_ids: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
        requested_deliberation_level: str | int | torch.Tensor | None = None,
        return_hot_logits: bool = False,
    ) -> DOPAOutput:
        device = input_ids.device
        batch, seq = input_ids.shape
        positions = torch.arange(seq, device=device)
        hidden = self.token_embedding(input_ids) * math.sqrt(self.cfg.model.d_model)
        hidden = self._run_hot(hidden, positions=positions)
        hot_hidden = hidden

        difficulty = self.difficulty_gate(hidden)
        if force_cold:
            active = torch.ones_like(difficulty, dtype=torch.bool)
        else:
            active = difficulty >= self.cfg.dopa.difficulty_threshold
        cold_weights = None
        cold_logits = None
        cold_selection = None
        if self.use_fine_cold:
            cold_selection = self.layer_demand(hidden, active=active)
            cold_logits = cold_selection.logits
            cold_weights = cold_selection.weights.unsqueeze(0).expand(batch, -1)
            cold_hidden = self.fine_cold_shell(hidden, cold_selection, positions=positions)
        else:
            cold_weights, cold_logits = self.layer_demand(hidden, active=active)
            cold_hidden = self.cold_manager.forward_selected(hidden, cold_weights, positions=positions)
        cold_hidden = cold_hidden.to(hidden.device)
        cold_hidden = cold_hidden * active.to(cold_hidden.dtype).unsqueeze(-1)
        hidden = self.fusion(hidden, cold_hidden)

        skel_embedding, lora_coeffs, lora_scores = self.compile_skeleton(skeleton, batch, hidden.device)
        hidden = self.apply_specialist(hidden, skel_embedding, lora_coeffs)
        hidden = self.apply_fast_weights(hidden)
        confidence = self.curiosity_gate(hidden)
        failure_probability = None
        failure_overconfidence = None
        if self.cfg.dopa.metacognition_enabled:
            failure_probability = self.failure_gate(hot_hidden)
            failure_overconfidence = failure_probability < self.cfg.dopa.failure_threshold
        knowledge_policy_logits = None
        if self.cfg.dopa.permanent_knowledge_enabled:
            knowledge_policy_logits = self.knowledge_policy_head(hot_hidden.mean(dim=1))
        deliberation = None
        process_reward = None
        if self.cfg.dopa.adaptive_deliberation_enabled:
            failure_for_deliberation = (
                failure_probability
                if failure_probability is not None
                else torch.zeros_like(confidence)
            )
            deliberation = self.deliberation_scheduler(
                hot_hidden,
                curiosity_confidence=confidence,
                failure_probability=failure_for_deliberation,
                task_complexity=difficulty.mean(dim=-1),
                requested_level=requested_deliberation_level,
            )
            process_reward = torch.sigmoid(self.process_reward_head(hot_hidden.mean(dim=1))).squeeze(-1)
        dspark = None
        dspark_schedule = None
        if self.cfg.dopa.dspark_enabled:
            dspark = self.dspark_heads(hot_hidden[:, -1], previous_tokens=input_ids[:, -1])
            dspark_schedule = self.dspark_scheduler(dspark.confidence)
        tool_calling = None
        if self.cfg.dopa.tool_calling_enabled:
            tool_calling = self.tool_calling_heads(hot_hidden)
        alignment = None
        if self.cfg.dopa.anti_hallucination_enabled and intent_ids is not None:
            intent_hidden = self.encode_intent(intent_ids.to(input_ids.device))
            alignment = self.intent_alignment_gate(
                intent_hidden,
                hot_hidden,
                task_type_ids=task_type_ids,
            )

        dspark = None
        dspark_schedule = None
        if self.cfg.dopa.dspark_enabled:
            dspark = self.dspark_heads(hot_hidden[:, -1], previous_tokens=input_ids[:, -1])
            dspark_schedule = self.dspark_scheduler(dspark.confidence)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        if alignment is not None:
            logits = self.intent_alignment_gate.apply_logit_bias(logits, alignment.score)
        ambiguity_score = None
        ask_clarification = None
        if self.cfg.dopa.adaptive_deliberation_enabled:
            ambiguity_score = self.ambiguity_detector(hidden, token_logits=logits)
            ask_clarification = self.ambiguity_detector.should_ask_clarification(ambiguity_score)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        aux = None
        if return_aux:
            aux = {
                "difficulty": difficulty,
                "cold_weights": cold_weights if cold_weights is not None else torch.zeros_like(difficulty[:, None]),
                "cold_logits": cold_logits,
                "cold_unit_count": torch.tensor(
                    len(cold_selection.units) if cold_selection is not None else 0,
                    device=hidden.device,
                ),
                "cold_visit_mean": self._cold_visit_mean(hidden.device),
                "lora_coeffs": lora_coeffs,
                "lora_scores": lora_scores,
                "curiosity_confidence": confidence,
            }
            if failure_probability is not None:
                aux["failure_probability"] = failure_probability
                aux["failure_overconfidence"] = failure_overconfidence
            if knowledge_policy_logits is not None:
                aux["knowledge_policy_logits"] = knowledge_policy_logits
            if deliberation is not None:
                aux["deliberation_logits"] = deliberation.logits
                aux["deliberation_level"] = deliberation.selected_level
                aux["task_complexity"] = deliberation.complexity_score
                aux["deliberation_confidence"] = deliberation.confidence
                aux["ambiguity_score"] = ambiguity_score
                aux["ask_clarification"] = ask_clarification
                aux["process_reward"] = process_reward
            if dspark is not None:
                aux["dspark_draft_logits"] = dspark.draft_logits
                aux["dspark_markov_logits"] = dspark.markov_logits
                aux["dspark_corrected_logits"] = dspark.corrected_logits
                aux["dspark_confidence"] = dspark.confidence
                aux["dspark_draft_tokens"] = dspark.draft_tokens
                aux["dspark_verify_lengths"] = dspark_schedule.verify_lengths
                aux["dspark_verify_mask"] = dspark_schedule.mask
                aux["dspark_prefix_survival"] = dspark_schedule.prefix_survival
            if tool_calling is not None:
                aux["tool_need_logits"] = tool_calling.need_logits
                aux["tool_argument_validity"] = tool_calling.argument_validity
                aux["tool_query_embedding"] = tool_calling.query_embedding
            if return_hot_logits:
                aux["hot_logits"] = self.lm_head(self.final_norm(hot_hidden))
            if alignment is not None:
                aux["alignment_score"] = alignment.score
                aux["alignment_triggered"] = alignment.triggered
                aux["intent_vector"] = alignment.intent_vector
                aux["implementation_vector"] = alignment.implementation_vector
        return DOPAOutput(logits=logits, loss=loss, hidden=hidden, aux=aux)

    @torch.no_grad()
    def forward_incremental(
        self,
        input_ids: torch.Tensor,
        position: int,
        hot_kv_cache: list[HotLayerCache] | None = None,
        cold_kv_cache: ColdSelectiveKVState | None = None,
        skeleton: SkeletonBatch | None = None,
        force_cold: bool = False,
        return_aux: bool = False,
        requested_deliberation_level: str | int | torch.Tensor | None = None,
    ) -> DOPAOutput:
        batch, seq = input_ids.shape
        if seq != 1:
            raise ValueError("forward_incremental expects exactly one token per batch")
        hidden = self.token_embedding(input_ids) * math.sqrt(self.cfg.model.d_model)
        hidden, new_hot_cache = self._run_hot_incremental(hidden, position=position, kv_cache=hot_kv_cache)
        hot_hidden = hidden

        difficulty = self.difficulty_gate(hidden)
        if force_cold:
            active = torch.ones_like(difficulty, dtype=torch.bool)
        else:
            active = difficulty >= self.cfg.dopa.difficulty_threshold
        cold_weights = None
        cold_logits = None
        cold_selection = None
        if self.use_fine_cold:
            cold_selection = self.layer_demand(hidden, active=active)
            cold_logits = cold_selection.logits
            cold_weights = cold_selection.weights.unsqueeze(0).expand(batch, -1)
            if cold_kv_cache is None and self.cfg.dopa.cold_kv_hot_units > 0 and self.cfg.dopa.cold_kv_window > 0:
                cold_kv_cache = ColdSelectiveKVState(
                    max_units=self.cfg.dopa.cold_kv_hot_units,
                    window=self.cfg.dopa.cold_kv_window,
                )
            cold_hidden = self.fine_cold_shell(
                hidden,
                cold_selection,
                positions=torch.tensor([position], device=hidden.device),
                cold_kv_state=cold_kv_cache,
            )
        else:
            cold_weights, cold_logits = self.layer_demand(hidden, active=active)
            cold_hidden = self.cold_manager.forward_selected(hidden, cold_weights, positions=torch.tensor([position], device=hidden.device))
        cold_hidden = cold_hidden.to(hidden.device)
        cold_hidden = cold_hidden * active.to(cold_hidden.dtype).unsqueeze(-1)
        hidden = self.fusion(hidden, cold_hidden)
        skel_embedding, lora_coeffs, lora_scores = self.compile_skeleton(skeleton, batch, hidden.device)
        hidden = self.apply_specialist(hidden, skel_embedding, lora_coeffs)
        hidden = self.apply_fast_weights(hidden)
        confidence = self.curiosity_gate(hidden)
        failure_probability = None
        failure_overconfidence = None
        if self.cfg.dopa.metacognition_enabled:
            failure_probability = self.failure_gate(hot_hidden)
            failure_overconfidence = failure_probability < self.cfg.dopa.failure_threshold
        knowledge_policy_logits = None
        if self.cfg.dopa.permanent_knowledge_enabled:
            knowledge_policy_logits = self.knowledge_policy_head(hot_hidden.mean(dim=1))
        deliberation = None
        process_reward = None
        if self.cfg.dopa.adaptive_deliberation_enabled:
            failure_for_deliberation = (
                failure_probability
                if failure_probability is not None
                else torch.zeros_like(confidence)
            )
            deliberation = self.deliberation_scheduler(
                hot_hidden,
                curiosity_confidence=confidence,
                failure_probability=failure_for_deliberation,
                task_complexity=difficulty.mean(dim=-1),
                requested_level=requested_deliberation_level,
            )
            process_reward = torch.sigmoid(self.process_reward_head(hot_hidden.mean(dim=1))).squeeze(-1)
        dspark = None
        dspark_schedule = None
        if self.cfg.dopa.dspark_enabled:
            dspark = self.dspark_heads(hot_hidden[:, -1], previous_tokens=input_ids[:, -1])
            dspark_schedule = self.dspark_scheduler(dspark.confidence)
        tool_calling = None
        if self.cfg.dopa.tool_calling_enabled:
            tool_calling = self.tool_calling_heads(hot_hidden)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        ambiguity_score = None
        ask_clarification = None
        if self.cfg.dopa.adaptive_deliberation_enabled:
            ambiguity_score = self.ambiguity_detector(hidden, token_logits=logits)
            ask_clarification = self.ambiguity_detector.should_ask_clarification(ambiguity_score)
        aux = None
        if return_aux:
            aux = {
                "difficulty": difficulty,
                "cold_weights": cold_weights if cold_weights is not None else torch.zeros_like(difficulty[:, None]),
                "cold_logits": cold_logits,
                "cold_unit_count": torch.tensor(
                    len(cold_selection.units) if cold_selection is not None else 0,
                    device=hidden.device,
                ),
                "cold_visit_mean": self._cold_visit_mean(hidden.device),
                "lora_coeffs": lora_coeffs,
                "lora_scores": lora_scores,
                "curiosity_confidence": confidence,
            }
            if failure_probability is not None:
                aux["failure_probability"] = failure_probability
                aux["failure_overconfidence"] = failure_overconfidence
            if knowledge_policy_logits is not None:
                aux["knowledge_policy_logits"] = knowledge_policy_logits
            if deliberation is not None:
                aux["deliberation_logits"] = deliberation.logits
                aux["deliberation_level"] = deliberation.selected_level
                aux["task_complexity"] = deliberation.complexity_score
                aux["deliberation_confidence"] = deliberation.confidence
                aux["ambiguity_score"] = ambiguity_score
                aux["ask_clarification"] = ask_clarification
                aux["process_reward"] = process_reward
            if tool_calling is not None:
                aux["tool_need_logits"] = tool_calling.need_logits
                aux["tool_argument_validity"] = tool_calling.argument_validity
                aux["tool_query_embedding"] = tool_calling.query_embedding
            if dspark is not None:
                aux["dspark_draft_logits"] = dspark.draft_logits
                aux["dspark_markov_logits"] = dspark.markov_logits
                aux["dspark_corrected_logits"] = dspark.corrected_logits
                aux["dspark_confidence"] = dspark.confidence
                aux["dspark_draft_tokens"] = dspark.draft_tokens
                aux["dspark_verify_lengths"] = dspark_schedule.verify_lengths
                aux["dspark_verify_mask"] = dspark_schedule.mask
                aux["dspark_prefix_survival"] = dspark_schedule.prefix_survival
        return DOPAOutput(
            logits=logits,
            hidden=hidden,
            aux=aux,
            hot_kv_cache=new_hot_cache,
            cold_kv_cache=cold_kv_cache,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        skeleton: SkeletonBatch | None = None,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_id: int | None = None,
        use_incremental: bool = False,
        intent_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.eval()
        ids = input_ids
        if use_incremental:
            return self.generate_incremental(
                ids,
                max_new_tokens=max_new_tokens,
                skeleton=skeleton,
                temperature=temperature,
                top_k=top_k,
                eos_id=eos_id,
            )
        for _ in range(max_new_tokens):
            x = ids[:, -self.cfg.model.max_seq_len :]
            out = self.forward(x, skeleton=skeleton, intent_ids=intent_ids)
            logits = out.logits[:, -1]
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    logits = torch.where(logits < values[:, [-1]], torch.full_like(logits, -float("inf")), logits)
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and torch.all(nxt.eq(eos_id)):
                break
        return ids

    @torch.no_grad()
    def generate_incremental(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        skeleton: SkeletonBatch | None = None,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        ids = input_ids
        cache = None
        cold_cache = None
        logits = None
        for pos in range(ids.size(1)):
            out = self.forward_incremental(
                ids[:, pos : pos + 1],
                position=pos,
                hot_kv_cache=cache,
                cold_kv_cache=cold_cache,
                skeleton=skeleton,
            )
            cache = out.hot_kv_cache
            cold_cache = out.cold_kv_cache
            logits = out.logits[:, -1]
        for _ in range(max_new_tokens):
            if logits is None:
                out = self.forward_incremental(
                    ids[:, -1:],
                    position=ids.size(1) - 1,
                    hot_kv_cache=cache,
                    cold_kv_cache=cold_cache,
                    skeleton=skeleton,
                )
                cache = out.hot_kv_cache
                cold_cache = out.cold_kv_cache
                logits = out.logits[:, -1]
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)), dim=-1)
                    next_logits = torch.where(
                        next_logits < values[:, [-1]],
                        torch.full_like(next_logits, -float("inf")),
                        next_logits,
                    )
                probs = torch.softmax(next_logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and torch.all(nxt.eq(eos_id)):
                break
            out = self.forward_incremental(
                nxt,
                position=ids.size(1) - 1,
                hot_kv_cache=cache,
                cold_kv_cache=cold_cache,
                skeleton=skeleton,
            )
            cache = out.hot_kv_cache
            cold_cache = out.cold_kv_cache
            logits = out.logits[:, -1]
        return ids

    def parameter_report(self) -> dict[str, int]:
        groups: dict[str, int] = {}
        for name, param in self.named_parameters():
            key = name.split(".")[0]
            groups[key] = groups.get(key, 0) + param.numel()
        groups["total"] = sum(p.numel() for p in self.parameters())
        groups["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return groups

    def freeze_base_for_stage(self, stage: str) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
        if stage == "stage2_5":
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.intent_alignment_gate.parameters():
                p.requires_grad_(True)
            return
        if stage == "stage_deliberation":
            for p in self.parameters():
                p.requires_grad_(False)
            for module in [self.deliberation_scheduler, self.ambiguity_detector, self.process_reward_head]:
                for p in module.parameters():
                    p.requires_grad_(True)
            return
        if stage == "stage_dspark":
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.dspark_heads.parameters():
                p.requires_grad_(True)
            return
        if stage == "stage_tool_calling":
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.tool_calling_heads.parameters():
                p.requires_grad_(True)
            return
        if stage in {"stage2", "stage3", "stage3_5"}:
            for module in [self.token_embedding, self.hot_layers, *self._cold_modules()]:
                for p in module.parameters():
                    p.requires_grad_(False)
        if stage == "stage1":
            for module in [
                self.difficulty_gate,
                self.layer_demand,
                self.fusion,
                self.skeleton_compiler,
                self.hyper,
                self.skeleton_to_model,
                self.lora_bank,
                self.curiosity_gate,
                self.failure_gate,
                self.knowledge_policy_head,
                self.deliberation_scheduler,
                self.ambiguity_detector,
                self.process_reward_head,
                self.dspark_heads,
                self.tool_calling_heads,
                self.intent_alignment_gate,
            ]:
                for p in module.parameters():
                    p.requires_grad_(False)

    def _cold_modules(self) -> list[nn.Module]:
        if self.use_fine_cold:
            return [self.fine_cold_shell]
        return [self.cold_manager]

    def _cold_visit_mean(self, device: torch.device) -> torch.Tensor:
        if self.use_fine_cold and hasattr(self.layer_demand, "visit_counts"):
            return self.layer_demand.visit_counts.float().mean().to(device)
        return torch.zeros((), device=device)
