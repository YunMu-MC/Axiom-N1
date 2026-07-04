from __future__ import annotations

from dataclasses import dataclass

import torch

from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.dspark import VerificationScheduler, accept_prefix_from_distributions
from dopa_coder_n1.model.skeleton import SkeletonBatch


@dataclass
class DSparkDecodeStats:
    cycles: int
    proposed_tokens: int
    verified_tokens: int
    accepted_tokens: int


@torch.no_grad()
def dspark_generate(
    model: DOPACoderN1,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    skeleton: SkeletonBatch | None = None,
    temperature: float = 0.0,
    eos_id: int | None = None,
    engine_load: float = 0.0,
) -> tuple[torch.Tensor, DSparkDecodeStats]:
    model.eval()
    if not model.cfg.dopa.dspark_enabled:
        return model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            skeleton=skeleton,
            temperature=temperature,
            eos_id=eos_id,
            use_incremental=True,
        ), DSparkDecodeStats(cycles=0, proposed_tokens=0, verified_tokens=0, accepted_tokens=0)

    ids = input_ids
    cycles = proposed = verified = accepted = 0
    scheduler = VerificationScheduler(gamma=model.cfg.dopa.dspark_gamma)
    while ids.size(1) < input_ids.size(1) + max_new_tokens:
        anchor = ids[:, -1:]
        anchor_out = model(anchor, skeleton=skeleton, return_aux=True)
        aux = anchor_out.aux or {}
        draft_logits = aux["dspark_corrected_logits"]
        confidence = aux["dspark_confidence"]
        draft_tokens = draft_logits.argmax(dim=-1)
        schedule = scheduler(confidence, engine_load=engine_load)
        verify_len = int(schedule.verify_lengths.max().item())
        verify_len = max(1, min(verify_len, max_new_tokens - (ids.size(1) - input_ids.size(1))))
        candidate = draft_tokens[:, :verify_len]
        proposed += int(draft_tokens.size(0) * draft_tokens.size(1))
        verified += int(candidate.numel())

        target_context = torch.cat([ids, candidate], dim=1)[:, -model.cfg.model.max_seq_len :]
        target = model(target_context, skeleton=skeleton)
        target_logits = target.logits[:, -verify_len - 1 : -1]
        target_log_probs = torch.log_softmax(target_logits, dim=-1)
        draft_log_probs = torch.log_softmax(draft_logits[:, :verify_len], dim=-1)
        result = accept_prefix_from_distributions(candidate, draft_log_probs, target_log_probs)
        take = int(result.accepted_lengths.max().item())
        if take == 0:
            next_token = target.logits[:, -1:].argmax(dim=-1)
            ids = torch.cat([ids, next_token], dim=1)
        else:
            ids = torch.cat([ids, candidate[:, :take]], dim=1)
            accepted += take
        cycles += 1
        if eos_id is not None and torch.all(ids[:, -1].eq(eos_id)):
            break
    return ids[:, : input_ids.size(1) + max_new_tokens], DSparkDecodeStats(
        cycles=cycles,
        proposed_tokens=proposed,
        verified_tokens=verified,
        accepted_tokens=accepted,
    )
