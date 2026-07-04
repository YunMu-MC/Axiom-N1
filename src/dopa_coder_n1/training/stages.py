from __future__ import annotations

import torch
from torch.nn import functional as F

from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.inductive_learning import SelfDrivenInductiveLoss
from dopa_coder_n1.training.hot_orchestration import external_knowledge_orchestration_loss


STAGE_ALIASES = {
    "stage1": "structural_reconstruction",
    "stage2": "hyper_lora",
    "stage2_5": "intent_alignment",
    "stage3": "isg_training",
    "stage3_5": "cognitive_search",
    "stage4": "cognitive_search",
    "stage_sdit": "self_driven_induction",
    "stage5": "knowledge_management",
    "stage_deliberation": "adaptive_deliberation",
    "stage_dspark": "dspark_speculative",
    "stage_tool_calling": "tool_calling",
    "stage_tool_schema_following": "tool_calling",
    "stage_tool_retrieval": "tool_calling",
    "stage_agent_rollout": "tool_calling",
}


def normalize_stage(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def stage_loss(model: DOPACoderN1, batch: dict, stage: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    stage = normalize_stage(stage)
    needs_hot_logits = (
        "teacher_token_losses" in batch
        and "hot_token_losses" not in batch
        and "external_knowledge_labels" not in batch
    )
    out = model(
        batch["input_ids"],
        labels=batch["labels"],
        skeleton=batch.get("skeleton"),
        intent_ids=batch.get("intent_ids", batch["input_ids"]),
        task_type_ids=batch.get("task_type_ids"),
        return_aux=True,
        return_hot_logits=needs_hot_logits,
    )
    if out.loss is None:
        raise RuntimeError("labels are required for training")
    loss = out.loss
    metrics = {"lm_loss": out.loss.detach()}
    aux = out.aux or {}
    orchestration_loss, orchestration_metrics = external_knowledge_orchestration_loss(batch, aux, model.cfg)
    if orchestration_loss is not None:
        loss = loss + orchestration_loss
        metrics.update(orchestration_metrics)
    if stage == "structural_reconstruction":
        if "cold_weights" in aux:
            cold_usage = aux["cold_weights"].sum(dim=-1).mean()
            metrics["cold_usage"] = cold_usage.detach()
    elif stage == "hyper_lora":
        cold_usage = aux["cold_weights"].sum(dim=-1).mean()
        loss = loss + 0.01 * cold_usage
        metrics["cold_usage"] = cold_usage.detach()
        if "cold_visit_mean" in aux:
            metrics["cold_visit_mean"] = aux["cold_visit_mean"].detach()
    elif stage == "isg_training":
        coeffs = aux["lora_coeffs"]
        entropy = -(coeffs.clamp_min(1e-8) * coeffs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        loss = loss - 0.001 * entropy
        metrics["lora_entropy"] = entropy.detach()
    elif stage == "intent_alignment":
        score = aux["alignment_score"]
        target = batch.get("alignment_labels", torch.ones_like(score)).to(score.device, score.dtype)
        alignment_bce = F.binary_cross_entropy(score, target)
        loss = loss + model.cfg.dopa.alignment_loss_weight * alignment_bce
        metrics["alignment_bce"] = alignment_bce.detach()
    elif stage == "adaptive_deliberation":
        if not model.cfg.dopa.adaptive_deliberation_enabled:
            raise RuntimeError("adaptive deliberation stage requires cfg.dopa.adaptive_deliberation_enabled")
        if "deliberation_logits" not in aux:
            raise RuntimeError("adaptive deliberation stage requires deliberation_logits")
        labels = batch.get("deliberation_level_labels")
        if labels is None:
            labels = torch.round(aux["task_complexity"] * (model.cfg.dopa.deliberation_level_count - 1)).long()
        else:
            labels = labels.to(out.logits.device).long().view(-1)
        policy_ce = F.cross_entropy(aux["deliberation_logits"], labels)
        reward_target = batch.get("process_reward_labels")
        if reward_target is None:
            reward_target = torch.ones_like(aux["process_reward"])
        else:
            reward_target = reward_target.to(out.logits.device, aux["process_reward"].dtype).view(-1)
        process_reward_loss = F.mse_loss(aux["process_reward"], reward_target)
        ambiguity_target = batch.get("ambiguity_labels")
        if ambiguity_target is None:
            ambiguity_target = torch.zeros_like(aux["ambiguity_score"])
        else:
            ambiguity_target = ambiguity_target.to(out.logits.device, aux["ambiguity_score"].dtype).view(-1)
        ambiguity_bce = F.binary_cross_entropy(aux["ambiguity_score"].clamp(1e-6, 1 - 1e-6), ambiguity_target)
        loss = (
            loss
            + model.cfg.dopa.deliberation_policy_loss_weight * policy_ce
            + model.cfg.dopa.process_reward_loss_weight * process_reward_loss
            + model.cfg.dopa.ambiguity_loss_weight * ambiguity_bce
        )
        metrics["deliberation_policy_ce"] = policy_ce.detach()
        metrics["process_reward_loss"] = process_reward_loss.detach()
        metrics["ambiguity_bce"] = ambiguity_bce.detach()
    elif stage == "cognitive_search":
        confidence = aux["curiosity_confidence"]
        calibration = F.binary_cross_entropy(confidence, torch.ones_like(confidence))
        loss = loss + 0.01 * calibration
        metrics["curiosity_bce"] = calibration.detach()
    elif stage == "dspark_speculative":
        if not model.cfg.dopa.dspark_enabled:
            raise RuntimeError("DSpark stage requires cfg.dopa.dspark_enabled")
        required = ("dspark_corrected_logits", "dspark_draft_logits", "dspark_markov_logits", "dspark_confidence")
        missing = [key for key in required if key not in aux]
        if missing:
            raise RuntimeError(f"DSpark stage is missing aux fields: {', '.join(missing)}")
        target = batch["labels"][:, 1 : 1 + aux["dspark_corrected_logits"].size(1)].to(out.logits.device)
        if target.size(1) < aux["dspark_corrected_logits"].size(1):
            pad = target[:, -1:].expand(-1, aux["dspark_corrected_logits"].size(1) - target.size(1))
            target = torch.cat([target, pad], dim=1)
        draft_ce = F.cross_entropy(
            aux["dspark_corrected_logits"].reshape(-1, aux["dspark_corrected_logits"].size(-1)),
            target.reshape(-1),
        )
        accept_labels = batch.get("dspark_accept_labels")
        if accept_labels is None:
            accept_labels = torch.ones_like(aux["dspark_confidence"])
        else:
            accept_labels = accept_labels.to(out.logits.device, aux["dspark_confidence"].dtype)
        confidence_bce = F.binary_cross_entropy(aux["dspark_confidence"], accept_labels)
        markov_kl = F.kl_div(
            F.log_softmax(aux["dspark_markov_logits"], dim=-1),
            F.softmax(aux["dspark_draft_logits"].detach(), dim=-1),
            reduction="batchmean",
        )
        loss = (
            loss
            + model.cfg.dopa.dspark_draft_loss_weight * draft_ce
            + model.cfg.dopa.dspark_confidence_loss_weight * confidence_bce
            + model.cfg.dopa.dspark_markov_loss_weight * markov_kl
        )
        metrics["dspark_draft_ce"] = draft_ce.detach()
        metrics["dspark_confidence_bce"] = confidence_bce.detach()
        metrics["dspark_markov_kl"] = markov_kl.detach()
    elif stage == "tool_calling":
        if not model.cfg.dopa.tool_calling_enabled:
            raise RuntimeError("tool calling stage requires cfg.dopa.tool_calling_enabled")
        required = ("tool_need_logits", "tool_argument_validity", "tool_query_embedding")
        missing = [key for key in required if key not in aux]
        if missing:
            raise RuntimeError(f"tool calling stage is missing aux fields: {', '.join(missing)}")
        need_labels = batch.get("tool_need_labels")
        if need_labels is None:
            need_labels = torch.zeros(aux["tool_need_logits"].size(0), dtype=torch.long, device=out.logits.device)
        else:
            need_labels = need_labels.to(out.logits.device).long().view(-1)
        need_ce = F.cross_entropy(aux["tool_need_logits"], need_labels)

        argument_labels = batch.get("tool_argument_valid_labels")
        if argument_labels is None:
            argument_labels = torch.ones_like(aux["tool_argument_validity"])
        else:
            argument_labels = argument_labels.to(out.logits.device, aux["tool_argument_validity"].dtype).view(-1)
        argument_bce = F.binary_cross_entropy(aux["tool_argument_validity"].clamp(1e-6, 1 - 1e-6), argument_labels)

        query_targets = batch.get("tool_query_targets")
        if query_targets is None:
            query_loss = torch.zeros((), device=out.logits.device, dtype=out.logits.dtype)
        else:
            query_targets = query_targets.to(out.logits.device, aux["tool_query_embedding"].dtype)
            query_targets = F.normalize(query_targets, dim=-1)
            target = torch.ones(aux["tool_query_embedding"].size(0), device=out.logits.device)
            query_loss = F.cosine_embedding_loss(aux["tool_query_embedding"], query_targets, target)

        loss = (
            loss
            + model.cfg.dopa.tool_need_loss_weight * need_ce
            + model.cfg.dopa.tool_argument_loss_weight * argument_bce
            + model.cfg.dopa.tool_query_loss_weight * query_loss
        )
        metrics["tool_need_ce"] = need_ce.detach()
        metrics["tool_argument_bce"] = argument_bce.detach()
        metrics["tool_query_cosine"] = query_loss.detach()
    elif stage == "self_driven_induction":
        if not model.cfg.dopa.self_driven_induction_enabled:
            raise RuntimeError("self-driven induction stage requires cfg.dopa.self_driven_induction_enabled")
        input_ids = batch["input_ids"]
        variant_a = batch.get("variant_a_input_ids")
        if variant_a is None:
            variant_a = input_ids.clone()
            variant_a[:, -1] = (variant_a[:, -1] + 1) % model.cfg.model.vocab_size
        variant_b = batch.get("variant_b_input_ids")
        if variant_b is None:
            variant_b = torch.flip(input_ids, dims=[1])
        rule_labels = batch.get("rule_labels", batch.get("labels", input_ids))
        variant_a = variant_a.to(input_ids.device)
        variant_b = variant_b.to(input_ids.device)
        rule_labels = rule_labels.to(input_ids.device)
        out_a = model(variant_a, skeleton=batch.get("skeleton"), return_aux=False)
        out_b = model(variant_b, skeleton=batch.get("skeleton"), return_aux=False)
        sdit = SelfDrivenInductiveLoss(
            consistency_weight=model.cfg.dopa.sdit_consistency_weight,
            transfer_weight=model.cfg.dopa.sdit_transfer_weight,
            reverse_weight=model.cfg.dopa.sdit_reverse_weight,
        )
        sdit_loss, sdit_metrics = sdit(
            out.hidden,
            out_a.hidden,
            out_b.hidden,
            reverse_logits=out.logits,
            rule_labels=rule_labels.to(out.logits.device),
        )
        loss = loss + sdit_loss
        for key, value in sdit_metrics.items():
            metrics[key] = value.detach()
    elif stage == "knowledge_management":
        if not model.cfg.dopa.permanent_knowledge_enabled:
            raise RuntimeError("knowledge management stage requires cfg.dopa.permanent_knowledge_enabled")
        if "knowledge_policy_logits" not in aux:
            raise RuntimeError("knowledge management stage requires knowledge_policy_logits")
        labels = batch.get("knowledge_action_labels")
        if labels is None:
            labels = torch.full(
                (aux["knowledge_policy_logits"].size(0),),
                4,
                dtype=torch.long,
                device=out.logits.device,
            )
        else:
            labels = labels.to(out.logits.device).long().view(-1)
        policy_ce = F.cross_entropy(aux["knowledge_policy_logits"], labels)
        loss = loss + 0.01 * policy_ce
        metrics["knowledge_policy_ce"] = policy_ce.detach()
    metrics["loss"] = loss.detach()
    return loss, metrics
