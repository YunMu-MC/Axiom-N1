from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.data.dialogue_cleaner import (
    CleanPolicy,
    ConversationRecord,
    StreamingDialogueCleaner,
    build_openassistant_conversations,
    clean_conversations,
    iter_openassistant_conversations_by_tree,
    language_group,
    normalize_helpsteer3_preference_row,
    normalize_hh_rlhf_row,
    normalize_prompt_target_row,
    normalize_wildchat_row,
    render_markdown_report,
    split_train_valid,
    write_jsonl,
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str
    split: str
    license_id: str
    converter: str
    url: str
    default_enabled: bool = True
    restricted: bool = False
    prompt_field: str = "prompt"
    target_field: str = "response"
    notes: str = ""


@dataclass(frozen=True)
class RustCleanedRecord:
    index: str
    language: str
    category: str
    quality_score: float
    quality_tags: tuple[str, ...]
    encoded_len: int
    source: str
    json_line: str


@dataclass(frozen=True)
class RustCleanBatchResult:
    accepted: list[RustCleanedRecord]
    rejected: Counter
    seen: int


class RustCleanerSession:
    def __init__(self, binary: Path, policy: CleanPolicy):
        self.binary = binary
        self.policy = policy
        self.process: subprocess.Popen | None = None

    def clean_batch(self, conversations: list[ConversationRecord]) -> RustCleanBatchResult:
        rows = build_cleaner_rows(conversations)
        if not rows:
            return RustCleanBatchResult(accepted=[], rejected=Counter(), seen=0)
        process = self._ensure_started()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Rust cleaner stream was not opened with stdin/stdout pipes.")
        try:
            process.stdin.write(f"{len(rows)}\n")
            process.stdin.write("\n".join(rows))
            process.stdin.write("\n")
            process.stdin.flush()
            output: list[str] = []
            for _ in rows:
                line = process.stdout.readline()
                if line == "":
                    raise RuntimeError(f"Rust cleaner stream ended early: {self._process_error_message()}")
                output.append(line)
        except OSError as exc:
            raise RuntimeError(f"Rust cleaner stream failed: {exc}") from exc
        return parse_cleaner_outputs("".join(output), seen=len(rows))

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if process.stdin is not None and process.poll() is None:
                process.stdin.close()
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def _ensure_started(self) -> subprocess.Popen:
        if self.process is not None and self.process.poll() is None:
            return self.process
        target_categories = self.policy.target_categories or frozenset()
        cmd = [
            str(self.binary),
            "clean-batch-stdin",
            str(self.policy.min_chars),
            str(self.policy.max_chars),
            f"{self.policy.min_score:.8f}",
            ",".join(sorted(self.policy.allowed_licenses)),
            ",".join(sorted(self.policy.allowed_languages)),
            ",".join(sorted(target_categories)),
        ]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        return self.process

    def _process_error_message(self) -> str:
        process = self.process
        if process is None:
            return "process not started"
        stderr = ""
        if process.poll() is not None and process.stderr is not None:
            stderr = process.stderr.read().strip()
        return stderr or f"exit code {process.returncode}"


@dataclass
class RustCleaner:
    mode: str
    binary: Path | None
    batch_size: int
    active: bool = False
    rejected: Counter = field(default_factory=Counter)
    accepted: int = 0
    batches: int = 0
    failed_batches: int = 0
    fallback_reasons: Counter = field(default_factory=Counter)
    last_error: str = ""
    disabled_reason: str = ""
    session: RustCleanerSession | None = field(default=None, repr=False)

    def clean_batch(
        self,
        conversations: list[ConversationRecord],
        *,
        policy: CleanPolicy,
        cache_dir: Path,
    ) -> RustCleanBatchResult | None:
        if not self.active or not conversations:
            return None
        if self.binary is None:
            raise RuntimeError("Rust cleaner is active but binary is missing.")
        try:
            if self.session is None:
                self.session = RustCleanerSession(self.binary, policy)
            result = self.session.clean_batch(conversations)
        except RuntimeError as exc:
            self.close()
            if self.mode != "auto":
                raise
            self.failed_batches += 1
            self.fallback_reasons["runtime_error"] += 1
            self.last_error = str(exc)[:500]
            self.disabled_reason = f"runtime_error: {self.last_error}"
            self.active = False
            print(
                f"Rust cleaner failed in auto mode; falling back to Python cleaner for this run: {exc}",
                file=sys.stderr,
            )
            return None
        self.accepted += len(result.accepted)
        self.rejected.update(result.rejected)
        self.batches += 1
        return result

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "active": self.active,
            "binary": str(self.binary) if self.binary else None,
            "batch_size": self.batch_size,
            "batches": self.batches,
            "failed_batches": self.failed_batches,
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "last_error": self.last_error,
            "disabled_reason": self.disabled_reason,
            "accepted": self.accepted,
            "rejected": dict(sorted(self.rejected.items())),
            "engine": "rust_full_cleaner",
        }


class RustPrefilterSession:
    def __init__(self, binary: Path, policy: CleanPolicy):
        self.binary = binary
        self.policy = policy
        self.process: subprocess.Popen | None = None

    def filter_batch(self, conversations: list[ConversationRecord]) -> dict[str, tuple[str, str]]:
        rows = build_prefilter_rows(conversations)
        if not rows:
            return {}
        process = self._ensure_started()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Rust prefilter stream was not opened with stdin/stdout pipes.")
        try:
            process.stdin.write(f"{len(rows)}\n")
            process.stdin.write("\n".join(rows))
            process.stdin.write("\n")
            process.stdin.flush()
            output: list[str] = []
            for _ in rows:
                line = process.stdout.readline()
                if line == "":
                    raise RuntimeError(f"Rust prefilter stream ended early: {self._process_error_message()}")
                output.append(line)
        except OSError as exc:
            raise RuntimeError(f"Rust prefilter stream failed: {exc}") from exc
        return parse_prefilter_verdicts("".join(output))

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if process.stdin is not None and process.poll() is None:
                process.stdin.close()
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def _ensure_started(self) -> subprocess.Popen:
        if self.process is not None and self.process.poll() is None:
            return self.process
        cmd = [
            str(self.binary),
            "filter-batch-stdin",
            str(self.policy.min_chars),
            str(self.policy.max_chars),
            ",".join(sorted(self.policy.allowed_licenses)),
            ",".join(sorted(self.policy.allowed_languages)),
        ]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        return self.process

    def _process_error_message(self) -> str:
        process = self.process
        if process is None:
            return "process not started"
        stderr = ""
        if process.poll() is not None and process.stderr is not None:
            stderr = process.stderr.read().strip()
        return stderr or f"exit code {process.returncode}"


@dataclass
class RustPrefilter:
    mode: str
    binary: Path | None
    batch_size: int
    active: bool = False
    rejected: Counter = field(default_factory=Counter)
    batches: int = 0
    failed_batches: int = 0
    fallback_reasons: Counter = field(default_factory=Counter)
    last_error: str = ""
    disabled_reason: str = ""
    session: RustPrefilterSession | None = field(default=None, repr=False)

    def filter_batch(
        self,
        conversations: list[ConversationRecord],
        *,
        policy: CleanPolicy,
        cache_dir: Path,
    ) -> tuple[list[ConversationRecord], Counter]:
        if not self.active or not conversations:
            return conversations, Counter()
        if self.binary is None:
            raise RuntimeError("Rust prefilter is active but binary is missing.")
        try:
            if self.session is None:
                self.session = RustPrefilterSession(self.binary, policy)
            verdicts = self.session.filter_batch(conversations)
        except RuntimeError as exc:
            self.close()
            if self.mode != "auto":
                raise
            self.failed_batches += 1
            self.fallback_reasons["runtime_error"] += 1
            self.last_error = str(exc)[:500]
            self.disabled_reason = f"runtime_error: {self.last_error}"
            self.active = False
            print(
                f"Rust fast filter failed in auto mode; disabling Rust prefilter for this run: {exc}",
                file=sys.stderr,
            )
            return conversations, Counter()
        accepted, rejected = apply_prefilter_verdicts(conversations, verdicts)
        self.rejected.update(rejected)
        self.batches += 1
        return accepted, rejected

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "active": self.active,
            "binary": str(self.binary) if self.binary else None,
            "batch_size": self.batch_size,
            "batches": self.batches,
            "failed_batches": self.failed_batches,
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "last_error": self.last_error,
            "disabled_reason": self.disabled_reason,
            "rejected": dict(sorted(self.rejected.items())),
        }


@dataclass
class StreamResumeState:
    total_bytes: int = 0
    written_records: int = 0
    written_language_bytes: dict[str, int] = field(default_factory=dict)
    written_language_records: dict[str, int] = field(default_factory=dict)
    categories: Counter = field(default_factory=Counter)
    sources: Counter = field(default_factory=Counter)
    source_stats: dict[str, dict[str, int | bool]] = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)
    next_shard_index: int = 0
    shard_count: int = 0


SOURCE_SPECS = {
    "oasst1": SourceSpec(
        name="OpenAssistant/oasst1",
        dataset="OpenAssistant/oasst1",
        config="default",
        split="train",
        license_id="apache-2.0",
        converter="openassistant",
        url="https://huggingface.co/datasets/OpenAssistant/oasst1",
        notes="Human assistant conversations, permissive Apache-2.0.",
    ),
    "oasst2": SourceSpec(
        name="OpenAssistant/oasst2",
        dataset="OpenAssistant/oasst2",
        config="default",
        split="train",
        license_id="apache-2.0",
        converter="openassistant",
        url="https://huggingface.co/datasets/OpenAssistant/oasst2",
        notes="Second OpenAssistant release; same tree converter and quality-label filtering.",
    ),
    "wildchat": SourceSpec(
        name="allenai/WildChat-1M",
        dataset="allenai/WildChat-1M",
        config="default",
        split="train",
        license_id="odc-by",
        converter="wildchat",
        url="https://huggingface.co/datasets/allenai/WildChat-1M",
        notes="Real user-chat traffic with redaction metadata; strict PII and quality gates stay enabled.",
    ),
    "aya_dataset": SourceSpec(
        name="CohereLabs/aya_dataset",
        dataset="CohereLabs/aya_dataset",
        config="default",
        split="train",
        license_id="apache-2.0",
        converter="prompt_target",
        url="https://huggingface.co/datasets/CohereLabs/aya_dataset",
        prompt_field="inputs",
        target_field="targets",
        notes="Multilingual human-annotated prompt/target data; useful for Chinese/English balance.",
    ),
    "aya_collection_translated_dolly": SourceSpec(
        name="CohereLabs/aya_collection:translated_dolly",
        dataset="CohereLabs/aya_collection",
        config="translated_dolly",
        split="train",
        license_id="apache-2.0",
        converter="prompt_target_en_zh",
        url="https://huggingface.co/datasets/CohereLabs/aya_collection",
        prompt_field="inputs",
        target_field="targets",
        notes="Aya Collection translated Dolly subset; accepts only Chinese/English rows after strict quality gates.",
    ),
    "aya_collection_flan_cot": SourceSpec(
        name="CohereLabs/aya_collection:translated_flan_cot",
        dataset="CohereLabs/aya_collection",
        config="translated_flan_cot",
        split="train",
        license_id="apache-2.0",
        converter="prompt_target_en_zh",
        url="https://huggingface.co/datasets/CohereLabs/aya_collection",
        prompt_field="inputs",
        target_field="targets",
        notes="Aya Collection translated chain-of-thought subset; accepts only Chinese/English rows.",
    ),
    "aya_collection_flan_qa": SourceSpec(
        name="CohereLabs/aya_collection:translated_flan_qa",
        dataset="CohereLabs/aya_collection",
        config="translated_flan_qa",
        split="train",
        license_id="apache-2.0",
        converter="prompt_target_en_zh",
        url="https://huggingface.co/datasets/CohereLabs/aya_collection",
        prompt_field="inputs",
        target_field="targets",
        notes="Aya Collection translated QA subset; useful for Chinese/English language-flow balance.",
    ),
    "helpsteer2": SourceSpec(
        name="nvidia/HelpSteer2",
        dataset="nvidia/HelpSteer2",
        config="default",
        split="train",
        license_id="cc-by-4.0",
        converter="prompt_target",
        url="https://huggingface.co/datasets/nvidia/HelpSteer2",
        prompt_field="prompt",
        target_field="response",
        notes="Human preference annotation source; final DOPA quality scorer remains authoritative.",
    ),
    "helpsteer3_preference": SourceSpec(
        name="nvidia/HelpSteer3",
        dataset="nvidia/HelpSteer3",
        config="preference",
        split="train",
        license_id="cc-by-4.0",
        converter="helpsteer3_preference",
        url="https://huggingface.co/datasets/nvidia/HelpSteer3",
        notes="Human preference source; keeps the preferred response only.",
    ),
    "hh_rlhf": SourceSpec(
        name="Anthropic/hh-rlhf",
        dataset="Anthropic/hh-rlhf",
        config="default",
        split="train",
        license_id="mit",
        converter="hh_rlhf",
        url="https://huggingface.co/datasets/Anthropic/hh-rlhf",
        default_enabled=False,
        notes="Optional safety/preference data; not default language-flow material.",
    ),
    "databricks_dolly_15k": SourceSpec(
        name="databricks/databricks-dolly-15k",
        dataset="databricks/databricks-dolly-15k",
        config="default",
        split="train",
        license_id="cc-by-sa-3.0",
        converter="dolly",
        url="https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        default_enabled=False,
        prompt_field="instruction",
        target_field="response",
        notes="Human-generated instruction/response data; small but useful as a quality-biased continuation source.",
    ),
    "stack_exchange_preferences": SourceSpec(
        name="HuggingFaceH4/stack-exchange-preferences",
        dataset="HuggingFaceH4/stack-exchange-preferences",
        config="default",
        split="train",
        license_id="cc-by-sa-4.0",
        converter="stack_exchange",
        url="https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences",
        notes="Stack Exchange human Q&A/preferences; keeps the chosen or highest-rated answer only.",
    ),
    "pmp_stack_exchange": SourceSpec(
        name="HuggingFaceH4/pmp-stack-exchange",
        dataset="HuggingFaceH4/pmp-stack-exchange",
        config="default",
        split="train",
        license_id="cc-by-sa-4.0",
        converter="stack_exchange",
        url="https://huggingface.co/datasets/HuggingFaceH4/pmp-stack-exchange",
        notes="Stack Exchange paired-preference source; converted to one best-answer assistant turn.",
    ),
    "openorca": SourceSpec(
        name="Open-Orca/OpenOrca",
        dataset="Open-Orca/OpenOrca",
        config="default",
        split="train",
        license_id="mit",
        converter="openorca",
        url="https://huggingface.co/datasets/Open-Orca/OpenOrca",
        default_enabled=False,
        notes="Synthetic teacher-response source; opt-in only, excluded from the default human-dialogue corpus.",
    ),
    "lmsys_chat_1m": SourceSpec(
        name="lmsys/lmsys-chat-1m",
        dataset="lmsys/lmsys-chat-1m",
        config="default",
        split="train",
        license_id="lmsys-chat-1m-agreement",
        converter="wildchat",
        url="https://huggingface.co/datasets/lmsys/lmsys-chat-1m",
        default_enabled=False,
        restricted=True,
        notes="Large real-chat source but has non-transfer restrictions; requires explicit opt-in.",
    ),
    "ultrachat_200k": SourceSpec(
        name="HuggingFaceH4/ultrachat_200k",
        dataset="HuggingFaceH4/ultrachat_200k",
        config="default",
        split="train_sft",
        license_id="mit",
        converter="wildchat",
        url="https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
        default_enabled=False,
        notes="Synthetic chat source; kept out of default high-quality real-dialogue corpus.",
    ),
}


def default_source_keys() -> list[str]:
    return [key for key, spec in SOURCE_SPECS.items() if spec.default_enabled and not spec.restricted]


def validate_selected_sources(selected: list[str], *, allow_restricted_license: bool) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    for key in selected:
        spec = SOURCE_SPECS[key]
        if spec.restricted and not allow_restricted_license:
            raise RuntimeError(
                f"Source {spec.name} is restricted; rerun with --allow-restricted-license "
                "only after you accept that dataset's agreement."
            )
        specs.append(spec)
    return specs


def build_clean_policy(
    *,
    min_score: float,
    target_categories: frozenset[str] | None,
    selected_specs: list[SourceSpec],
    allow_restricted_license: bool,
) -> CleanPolicy:
    policy = CleanPolicy(min_score=min_score, target_categories=target_categories)
    if not allow_restricted_license:
        return policy
    extra_licenses = {spec.license_id.lower() for spec in selected_specs if spec.restricted}
    if not extra_licenses:
        return policy
    return CleanPolicy(
        allowed_licenses=frozenset(set(policy.allowed_licenses) | extra_licenses),
        allowed_languages=policy.allowed_languages,
        min_turns=policy.min_turns,
        max_turns=policy.max_turns,
        min_chars=policy.min_chars,
        max_chars=policy.max_chars,
        min_score=policy.min_score,
        keep_rejected_samples=policy.keep_rejected_samples,
        target_categories=policy.target_categories,
    )


def source_catalog_payload() -> list[dict]:
    return [
        {
            "key": key,
            "name": spec.name,
            "dataset": spec.dataset,
            "config": spec.config,
            "split": spec.split,
            "license": spec.license_id,
            "default_enabled": spec.default_enabled,
            "restricted": spec.restricted,
            "url": spec.url,
            "notes": spec.notes,
        }
        for key, spec in SOURCE_SPECS.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and clean a small high-quality dialogue corpus trial for DOPA language flow."
    )
    parser.add_argument(
        "--out-dir",
        default="data/dialogue_corpus_100gb",
        help="Output directory for cleaned JSONL and reports.",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCE_SPECS),
        help="Source to fetch. Repeat for multiple sources. Defaults to unrestricted high-quality sources.",
    )
    parser.add_argument("--list-sources", action="store_true", help="Print the source catalog and exit.")
    parser.add_argument("--limit-per-source", type=int, default=300)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-accepted", type=int, default=300)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--target-gb",
        type=float,
        default=0.0,
        help="If >0, stream-clean until this many GiB are written or sources are exhausted.",
    )
    parser.add_argument("--shard-mb", type=float, default=1024.0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--fast-filter",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use the Rust hard-reject prefilter before Python scoring. auto uses it if built.",
    )
    parser.add_argument("--fast-filter-bin", default="", help="Path to dopa_dialogue_filter binary.")
    parser.add_argument("--fast-filter-batch-size", type=int, default=4096)
    parser.add_argument("--progress-min-seconds", type=float, default=5.0)
    parser.add_argument(
        "--resume-dedupe-scan",
        action="store_true",
        help="Scan existing shards and seed duplicate fingerprints on resume. Slower for large corpora.",
    )
    parser.add_argument(
        "--language-ratio",
        action="append",
        default=["en=0.5", "zh=0.5"],
        help="Target written-byte language ratio, e.g. en=0.5 and zh=0.5.",
    )
    parser.add_argument(
        "--target-category",
        action="append",
        help="Keep only a quality category. Repeat for multiple categories.",
    )
    parser.add_argument(
        "--backend",
        choices=["datasets", "rows-api", "hf-mirror-parquet", "auto"],
        default="datasets",
        help="Fetch backend. datasets uses Hugging Face datasets with streaming first.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint for datasets/hub access.",
    )
    parser.add_argument("--cache-dir", default=".cache/hf_dialogue")
    parser.add_argument(
        "--allow-restricted-license",
        action="store_true",
        help="Reserved for future sources with extra license agreements. Not used by default.",
    )
    args = parser.parse_args()

    if args.list_sources:
        print(json.dumps(source_catalog_payload(), ensure_ascii=False, indent=2, sort_keys=True))
        return

    selected = args.source or default_source_keys()
    selected_specs = validate_selected_sources(selected, allow_restricted_license=args.allow_restricted_license)
    out_dir = Path(args.out_dir)
    clean_dir = out_dir / "clean"
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    target_categories = frozenset(args.target_category) if args.target_category else None
    policy = build_clean_policy(
        min_score=args.min_score,
        target_categories=target_categories,
        selected_specs=selected_specs,
        allow_restricted_license=args.allow_restricted_license,
    )
    if args.target_gb > 0:
        run_streaming_clean(args, selected, policy, out_dir, report_dir)
        return

    conversations: list[ConversationRecord] = []
    fetch_manifest: list[dict] = []
    for key in selected:
        spec = SOURCE_SPECS[key]
        rows = fetch_hf_rows(
            spec,
            limit=args.limit_per_source,
            page_size=args.page_size,
            timeout=args.timeout,
            backend=args.backend,
            hf_endpoint=args.hf_endpoint,
            cache_dir=Path(args.cache_dir),
        )
        converted = convert_rows(spec, rows)
        conversations.extend(converted)
        fetch_manifest.append(
            {
                "source": spec.name,
                "dataset": spec.dataset,
                "url": spec.url,
                "license": spec.license_id,
                "split": spec.split,
                "requested_rows": args.limit_per_source,
                "fetched_rows": len(rows),
                "converted_conversations": len(converted),
            }
        )
        print(f"{spec.name}: fetched_rows={len(rows)} converted_conversations={len(converted)}")

    cleaned, report = clean_conversations(conversations, policy)
    cleaned = sorted(cleaned, key=lambda item: (-item.quality_score, item.source, item.source_id))
    if args.max_accepted > 0:
        cleaned = cleaned[: args.max_accepted]
        report.accepted = len(cleaned)
        report.categories.clear()
        report.sources.clear()
        for item in cleaned:
            report.categories[item.category] += 1
            report.sources[item.source] += 1

    train, valid = split_train_valid(cleaned, valid_ratio=args.valid_ratio, seed=args.seed)
    train_count = write_jsonl(clean_dir / "train.jsonl", (item.to_training_record() for item in train))
    valid_count = write_jsonl(clean_dir / "valid.jsonl", (item.to_training_record() for item in valid))
    write_jsonl(report_dir / "accepted_sample.jsonl", (item.to_training_record() for item in cleaned[:25]))
    (report_dir / "quality_report.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "policy": {
                    "min_score": args.min_score,
                    "valid_ratio": args.valid_ratio,
                    "max_accepted": args.max_accepted,
                    "allowed_licenses": sorted(policy.allowed_licenses),
                    "allowed_languages": sorted(policy.allowed_languages),
                    "target_categories": sorted(policy.target_categories) if policy.target_categories else None,
                    "no_raw_rows_saved": True,
                },
                "fetch_manifest": fetch_manifest,
                "clean_report": report.to_dict(),
                "outputs": {
                    "train_jsonl": str(clean_dir / "train.jsonl"),
                    "valid_jsonl": str(clean_dir / "valid.jsonl"),
                    "accepted_sample_jsonl": str(report_dir / "accepted_sample.jsonl"),
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (report_dir / "quality_report.md").write_text(
        render_markdown_report(report, train_count=train_count, valid_count=valid_count),
        encoding="utf-8",
    )
    print(f"cleaned={len(cleaned)} train={train_count} valid={valid_count} out={out_dir}")


def convert_rows(spec: SourceSpec, rows: list[dict]) -> list[ConversationRecord]:
    if spec.converter == "openassistant":
        return build_openassistant_conversations(rows, source=spec.name, license_id=spec.license_id)
    if spec.converter == "wildchat":
        conversations = [
            normalize_wildchat_row(row, source=spec.name, license_id=spec.license_id) for row in rows
        ]
        return [conversation for conversation in conversations if conversation is not None]
    if spec.converter == "prompt_target":
        conversations = [
            normalize_prompt_target_row(
                row,
                source=spec.name,
                license_id=spec.license_id,
                prompt_field=spec.prompt_field,
                target_field=spec.target_field,
            )
            for row in rows
        ]
        return [conversation for conversation in conversations if conversation is not None]
    if spec.converter == "prompt_target_en_zh":
        conversations = [
            normalize_prompt_target_row(
                row,
                source=spec.name,
                license_id=spec.license_id,
                prompt_field=spec.prompt_field,
                target_field=spec.target_field,
            )
            for row in rows
            if row_language_is_en_or_zh(row, prompt_field=spec.prompt_field, target_field=spec.target_field)
        ]
        return [conversation for conversation in conversations if conversation is not None]
    if spec.converter == "helpsteer3_preference":
        conversations = [
            normalize_helpsteer3_preference_row(row, source=spec.name, license_id=spec.license_id)
            for row in rows
        ]
        return [conversation for conversation in conversations if conversation is not None]
    if spec.converter == "hh_rlhf":
        conversations = [
            normalize_hh_rlhf_row(row, source=spec.name, license_id=spec.license_id) for row in rows
        ]
        return [conversation for conversation in conversations if conversation is not None]
    if spec.converter == "dolly":
        conversations = [
            normalize_dolly_row(row, source=spec.name, license_id=spec.license_id) for row in rows
        ]
        return [conversation for conversation in conversations if conversation is not None]
    if spec.converter == "stack_exchange":
        conversations = [
            normalize_stack_exchange_row(row, source=spec.name, license_id=spec.license_id) for row in rows
        ]
        return [conversation for conversation in conversations if conversation is not None]
    if spec.converter == "openorca":
        conversations = [
            normalize_openorca_row(row, source=spec.name, license_id=spec.license_id) for row in rows
        ]
        return [conversation for conversation in conversations if conversation is not None]
    raise ValueError(f"Unsupported converter: {spec.converter}")


def row_language_is_en_or_zh(row: Mapping[str, Any], *, prompt_field: str, target_field: str) -> bool:
    language_values = [
        row.get("language"),
        row.get("language_code"),
        row.get("lang"),
        row.get("locale"),
        row.get("script"),
    ]
    normalized = {str(value).strip().lower() for value in language_values if value is not None}
    en_zh_markers = {
        "en",
        "eng",
        "english",
        "zh",
        "zho",
        "cmn",
        "chinese",
        "zh-cn",
        "zh-hans",
        "zh-hant",
        "simplified chinese",
        "traditional chinese",
    }
    non_target_markers = {
        "ar",
        "ara",
        "de",
        "deu",
        "es",
        "spa",
        "fr",
        "fra",
        "hi",
        "hin",
        "id",
        "ind",
        "it",
        "ita",
        "ja",
        "jpn",
        "ko",
        "kor",
        "pt",
        "por",
        "ru",
        "rus",
    }
    if normalized & en_zh_markers:
        return True
    if normalized & non_target_markers:
        return False
    prompt = str(row.get(prompt_field) or "")
    target = str(row.get(target_field) or "")
    text = f"{prompt}\n{target}"
    if re.search(r"[\u4e00-\u9fff]", text):
        return True
    ascii_letters = sum(1 for char in text if ("a" <= char.lower() <= "z"))
    return ascii_letters >= 40 and ascii_letters >= max(1, len(text.strip())) * 0.35


def normalize_dolly_row(row: dict, *, source: str, license_id: str) -> ConversationRecord | None:
    instruction = str(row.get("instruction") or "").strip()
    response = str(row.get("response") or "").strip()
    if not instruction or not response:
        return None
    context = str(row.get("context") or "").strip()
    prompt = instruction if not context else f"{instruction}\n\nContext:\n{context}"
    category = str(row.get("category") or "").strip()
    source_id = str(row.get("id") or row.get("index") or "").strip()
    if not source_id:
        source_id = hashlib.sha256(f"{prompt}\n{response}".encode("utf-8")).hexdigest()[:16]
    return ConversationRecord(
        source=source,
        license_id=license_id.lower(),
        source_id=source_id,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        metadata={key: value for key, value in {"category": category, "language": "en"}.items() if value},
    )


def normalize_stack_exchange_row(row: Mapping[str, Any], *, source: str, license_id: str) -> ConversationRecord | None:
    prompt = _stack_exchange_prompt(row)
    answer, answer_meta = _best_stack_exchange_answer(row)
    if not prompt or not answer:
        return None
    source_id = _first_text(
        row,
        (
            "qid",
            "question_id",
            "post_id",
            "id",
            "prompt_id",
            "sample_id",
        ),
    )
    if not source_id:
        source_id = hashlib.sha256(f"{prompt}\n{answer}".encode("utf-8")).hexdigest()[:16]
    metadata: dict[str, str | int | float | bool | None] = {
        "language": (_first_text(row, ("language", "lang", "locale")) or "en").lower(),
        "site": _first_text(row, ("site", "site_name", "subreddit")),
        "tags": _stack_tags(row),
        "answer_score": answer_meta.get("score"),
        "answer_accepted": answer_meta.get("accepted"),
        "preference_source": answer_meta.get("source"),
    }
    return ConversationRecord(
        source=source,
        license_id=license_id.lower(),
        source_id=source_id,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        metadata={key: value for key, value in metadata.items() if value not in {"", None}},
    )


def normalize_openorca_row(row: Mapping[str, Any], *, source: str, license_id: str) -> ConversationRecord | None:
    question = _clean_source_text(_first_raw(row, ("question", "prompt", "instruction", "input")))
    response = _clean_source_text(_first_raw(row, ("response", "answer", "output", "completion")))
    if not question or not response:
        return None
    system_prompt = _clean_source_text(_first_raw(row, ("system_prompt", "system", "system_instruction")))
    prompt = question if not system_prompt else f"System:\n{system_prompt}\n\nUser:\n{question}"
    source_id = _first_text(row, ("id", "sample_id", "question_id"))
    if not source_id:
        source_id = hashlib.sha256(f"{prompt}\n{response}".encode("utf-8")).hexdigest()[:16]
    return ConversationRecord(
        source=source,
        license_id=license_id.lower(),
        source_id=source_id,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        metadata={
            "language": (_first_text(row, ("language", "lang", "locale")) or "en").lower(),
            "synthetic": True,
            "has_system_prompt": bool(system_prompt),
        },
    )


def _stack_exchange_prompt(row: Mapping[str, Any]) -> str:
    title = _clean_source_text(_first_raw(row, ("title", "question_title")))
    body = _clean_source_text(_first_raw(row, ("question", "body", "prompt", "instruction", "input")))
    if title and body and title.lower() not in body.lower():
        return f"{title}\n\n{body}"
    return body or title


def _best_stack_exchange_answer(row: Mapping[str, Any]) -> tuple[str, dict[str, str | int | float | bool | None]]:
    candidates: list[dict[str, Any]] = []
    raw_answers = row.get("answers")
    if isinstance(raw_answers, list):
        for index, item in enumerate(raw_answers):
            if isinstance(item, Mapping):
                text = _clean_source_text(_first_raw(item, ("text", "body", "answer", "response", "content")))
                if text:
                    candidates.append(
                        {
                            "text": text,
                            "score": _number_or_none(_first_raw(item, ("score", "upvotes", "pm_score", "rank"))),
                            "accepted": _truthy(_first_raw(item, ("is_accepted", "accepted", "chosen", "selected"))),
                            "source": f"answers[{index}]",
                        }
                    )
            else:
                text = _clean_source_text(item)
                if text:
                    candidates.append({"text": text, "score": None, "accepted": False, "source": f"answers[{index}]"})

    for key in ("chosen", "accepted", "accepted_answer", "best_answer", "answer", "response", "output", "completion"):
        text = _clean_source_text(_first_raw(row, (key,)))
        if text:
            candidates.append(
                {
                    "text": text,
                    "score": _number_or_none(_first_raw(row, (f"{key}_score", "score", "pm_score"))),
                    "accepted": key in {"chosen", "accepted", "accepted_answer", "best_answer"},
                    "source": key,
                }
            )

    pair_keys = (("response_j", "score_j"), ("response_k", "score_k"), ("answer_j", "score_j"), ("answer_k", "score_k"))
    for response_key, score_key in pair_keys:
        text = _clean_source_text(_first_raw(row, (response_key,)))
        if text:
            candidates.append(
                {
                    "text": text,
                    "score": _number_or_none(_first_raw(row, (score_key,))),
                    "accepted": False,
                    "source": response_key,
                }
            )

    if not candidates:
        return "", {}
    candidates.sort(
        key=lambda item: (
            bool(item.get("accepted")),
            _score_for_sort(item.get("score")),
            len(str(item.get("text") or "")),
        ),
        reverse=True,
    )
    best = candidates[0]
    score = best.get("score")
    if isinstance(score, float) and score.is_integer():
        score = int(score)
    meta = {
        "score": score,
        "accepted": bool(best.get("accepted")),
        "source": str(best.get("source") or ""),
    }
    return str(best["text"]), meta


def _first_raw(row: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return ""


def _first_text(row: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    return _clean_source_text(_first_raw(row, keys))


def _stack_tags(row: Mapping[str, Any]) -> str:
    tags = row.get("tags") or row.get("tag") or row.get("categories") or ""
    if isinstance(tags, str):
        return ",".join(part.strip() for part in re.split(r"[,;| ]+", tags) if part.strip())
    if isinstance(tags, (list, tuple, set)):
        return ",".join(str(item).strip() for item in tags if str(item).strip())
    return ""


def _clean_source_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and not value.strip():
        return ""
    if isinstance(value, Mapping):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, (list, tuple, set)):
        value = "\n".join(str(item) for item in value)
    text = html.unescape(str(value))
    text = re.sub(r"(?is)<(script|style).*?</\1>", "", text)
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"(?i)</\s*(p|li|div|pre|blockquote|h[1-6])\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line.strip()).strip()


def _number_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_for_sort(value: Any) -> float:
    number = _number_or_none(value)
    return -1_000_000.0 if number is None else number


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "accepted", "chosen", "preferred"}


def shard_sort_key(path: Path) -> tuple[int, int, str]:
    stem = path.stem
    if stem.startswith("shard-"):
        suffix = stem[len("shard-") :]
        if suffix.isdigit():
            return (0, int(suffix), path.name)
    return (1, 0, path.name)


def count_shard_files(shard_dir: Path) -> int:
    if not shard_dir.exists():
        return 0
    return sum(1 for path in shard_dir.glob("*.jsonl") if path.is_file() and path.stat().st_size > 0)


def next_stream_shard_index(shard_dir: Path) -> int:
    max_index = -1
    if not shard_dir.exists():
        return 0
    for path in shard_dir.glob("shard-*.jsonl"):
        suffix = path.stem[len("shard-") :]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def open_next_stream_shard(shard_dir: Path, shard_index: int):
    while True:
        path = shard_dir / f"shard-{shard_index:05d}.jsonl"
        try:
            return path.open("xb"), shard_index + 1
        except FileExistsError:
            shard_index += 1


def iter_existing_training_records(shard_dir: Path) -> Iterator[tuple[dict, int]]:
    if not shard_dir.exists():
        return
    for path in sorted(shard_dir.glob("*.jsonl"), key=shard_sort_key):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    print(f"Skipping malformed JSONL line {path}:{line_number}: {exc}", file=sys.stderr)
                    continue
                if isinstance(record, dict):
                    yield record, len(line.encode("utf-8"))


def load_stream_resume_state_from_report(report_dir: Path, shard_dir: Path) -> StreamResumeState | None:
    report_path = report_dir / "quality_report.json"
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read resume report {report_path}: {exc}", file=sys.stderr)
        return None
    target = payload.get("target") if isinstance(payload, dict) else None
    if not isinstance(target, dict):
        return None
    clean_report = payload.get("clean_report") if isinstance(payload, dict) else {}
    source_stats_payload = payload.get("source_stats") if isinstance(payload, dict) else {}
    state = StreamResumeState(
        total_bytes=int(target.get("written_bytes") or 0),
        written_records=int(target.get("written_records") or 0),
        written_language_bytes={
            str(key): int(value)
            for key, value in (target.get("language_written_bytes") or {}).items()
        },
        written_language_records={
            str(key): int(value)
            for key, value in (target.get("language_written_records") or {}).items()
        },
        next_shard_index=next_stream_shard_index(shard_dir),
        shard_count=count_shard_files(shard_dir),
    )
    if isinstance(clean_report, dict):
        state.categories.update({str(key): int(value) for key, value in (clean_report.get("categories") or {}).items()})
        state.sources.update({str(key): int(value) for key, value in (clean_report.get("sources") or {}).items()})
    if isinstance(source_stats_payload, dict):
        for source, stats in source_stats_payload.items():
            if isinstance(stats, dict):
                state.source_stats[str(source)] = {
                    "converted_conversations": int(stats.get("converted_conversations") or 0),
                    "accepted": int(stats.get("accepted") or 0),
                    "exhausted": bool(stats.get("exhausted", False)),
                }
    sample_path = report_dir / "accepted_sample.jsonl"
    if sample_path.exists():
        for record, _ in iter_existing_training_records(sample_path.parent):
            state.samples.append(record)
            if len(state.samples) >= 25:
                break
    return state if state.total_bytes > 0 or state.written_records > 0 else None


def scan_stream_resume_state(shard_dir: Path, cleaner: StreamingDialogueCleaner) -> StreamResumeState:
    state = StreamResumeState(
        total_bytes=sum(
            path.stat().st_size
            for path in shard_dir.glob("*.jsonl")
            if path.is_file() and path.stat().st_size > 0
        )
        if shard_dir.exists()
        else 0,
        next_shard_index=next_stream_shard_index(shard_dir),
        shard_count=count_shard_files(shard_dir),
    )
    for record, line_bytes in iter_existing_training_records(shard_dir):
        if not cleaner.remember_training_record(record):
            continue
        state.written_records += 1
        language = str(record.get("language") or "unknown")
        state.written_language_bytes[language] = state.written_language_bytes.get(language, 0) + line_bytes
        state.written_language_records[language] = state.written_language_records.get(language, 0) + 1
        category = str(record.get("category") or "unknown")
        source = str(record.get("source") or "unknown")
        state.categories[category] += 1
        state.sources[source] += 1
        if len(state.samples) < 25:
            state.samples.append(record)
    for source, accepted in state.sources.items():
        state.source_stats[source] = {
            "converted_conversations": 0,
            "accepted": int(accepted),
            "exhausted": False,
        }
    return state


def load_stream_resume_state(
    shard_dir: Path,
    cleaner: StreamingDialogueCleaner,
    *,
    report_dir: Path | None = None,
    seed_dedupe: bool = False,
) -> StreamResumeState:
    if report_dir is not None and not seed_dedupe:
        state = load_stream_resume_state_from_report(report_dir, shard_dir)
        if state is not None:
            return state
    return scan_stream_resume_state(shard_dir, cleaner)


def run_streaming_clean(
    args: argparse.Namespace,
    selected: list[str],
    policy: CleanPolicy,
    out_dir: Path,
    report_dir: Path,
) -> None:
    target_bytes = int(args.target_gb * 1024**3)
    shard_bytes = int(args.shard_mb * 1024**2)
    language_byte_targets = parse_language_ratios(args.language_ratio, target_bytes)
    shard_dir = out_dir / "clean" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    configure_hf_cache(Path(args.cache_dir), args.hf_endpoint)
    cleaner = StreamingDialogueCleaner(policy)
    rust_cleaner = prepare_rust_cleaner(args)
    resume = load_stream_resume_state(
        shard_dir,
        cleaner,
        report_dir=report_dir,
        seed_dedupe=bool(getattr(args, "resume_dedupe_scan", False)),
    )
    if resume.written_records:
        cleaner.report.accepted = resume.written_records
        cleaner.report.seen = resume.written_records
        cleaner.report.categories.update(resume.categories)
        cleaner.report.sources.update(resume.sources)
    source_stats: dict[str, dict[str, int | bool]] = {
        source: dict(stats) for source, stats in resume.source_stats.items()
    }
    samples: list[dict] = list(resume.samples)
    total_bytes = resume.total_bytes
    written_records = resume.written_records
    written_language_bytes: dict[str, int] = dict(resume.written_language_bytes)
    written_language_records: dict[str, int] = dict(resume.written_language_records)
    for lang in language_byte_targets:
        written_language_bytes.setdefault(lang, 0)
        written_language_records.setdefault(lang, 0)
    stream_skipped: dict[str, int] = {}
    shard_index = resume.next_shard_index
    shard_current = 0
    handle = None
    stopped_at_target = False
    last_progress_at = time.monotonic()
    if resume.written_records:
        print(
            f"resume records={resume.written_records} bytes_gib={resume.total_bytes / 1024**3:.3f} "
            f"shards={resume.shard_count} next_shard={resume.next_shard_index}",
            flush=True,
        )
        write_stream_report(
            report_dir,
            cleaner,
            source_stats,
            policy,
            args,
            total_bytes,
            written_records,
            count_shard_files(shard_dir),
            written_language_bytes,
            written_language_records,
            stream_skipped,
            rust_cleaner.to_dict(),
            stopped_at_target=total_bytes >= target_bytes,
        )
    if total_bytes >= target_bytes:
        print(
            f"stream_resume target already reached records={written_records} "
            f"bytes_gib={total_bytes / 1024**3:.3f} target_gib={args.target_gb:.3f} out={out_dir}",
            flush=True,
        )
        return
    try:
        for key in selected:
            spec = SOURCE_SPECS[key]
            stats = source_stats.setdefault(spec.name, {"converted_conversations": 0, "accepted": 0, "exhausted": False})
            if bool(stats.get("exhausted", False)):
                print(f"resume skip exhausted source={spec.name}", flush=True)
                continue
            source_exhausted = True
            source_iter = iter_source_conversations(
                spec,
                limit=args.limit_per_source,
                backend=args.backend,
                hf_endpoint=args.hf_endpoint,
                cache_dir=Path(args.cache_dir),
                page_size=args.page_size,
                timeout=args.timeout,
            )
            for batch in iter_batches(source_iter, max(1, args.fast_filter_batch_size)):
                stats["converted_conversations"] = int(stats["converted_conversations"]) + len(batch)
                rust_result = rust_cleaner.clean_batch(
                    batch,
                    policy=policy,
                    cache_dir=Path(args.cache_dir),
                )
                if rust_result is None:
                    accepted_items = iter_python_cleaned_records(batch, cleaner)
                else:
                    cleaner.report.seen += rust_result.seen
                    cleaner.report.rejected.update(rust_result.rejected)
                    accepted_items = iter_rust_cleaned_records(rust_result, cleaner)
                for accepted in accepted_items:
                    line = accepted.json_line
                    encoded = line.encode("utf-8")
                    encoded_len = accepted.encoded_len
                    lang = accepted.language
                    if language_byte_targets:
                        if lang not in language_byte_targets:
                            stream_skipped["language_not_target"] = stream_skipped.get("language_not_target", 0) + 1
                            continue
                        if written_language_bytes[lang] + encoded_len > language_byte_targets[lang]:
                            stream_skipped["language_quota_full"] = stream_skipped.get("language_quota_full", 0) + 1
                            continue
                    if total_bytes + encoded_len > target_bytes:
                        stopped_at_target = True
                        source_exhausted = False
                        break
                    if handle is None or shard_current + encoded_len > shard_bytes:
                        if handle is not None:
                            handle.close()
                        handle, shard_index = open_next_stream_shard(shard_dir, shard_index)
                        shard_current = 0
                    handle.write(encoded)
                    total_bytes += encoded_len
                    shard_current += encoded_len
                    written_records += 1
                    written_language_bytes[lang] = written_language_bytes.get(lang, 0) + encoded_len
                    written_language_records[lang] = written_language_records.get(lang, 0) + 1
                    stats["accepted"] = int(stats["accepted"]) + 1
                    if len(samples) < 25:
                        samples.append(json.loads(line))
                    if written_records % max(1, args.progress_every) == 0:
                        now = time.monotonic()
                        if should_emit_progress(
                            written_records=written_records,
                            progress_every=args.progress_every,
                            now=now,
                            last_progress_at=last_progress_at,
                            min_seconds=args.progress_min_seconds,
                        ):
                            last_progress_at = now
                            write_stream_report(
                                report_dir,
                                cleaner,
                                source_stats,
                                policy,
                                args,
                                total_bytes,
                                written_records,
                                count_shard_files(shard_dir),
                                written_language_bytes,
                                written_language_records,
                                stream_skipped,
                                rust_cleaner.to_dict(),
                                stopped_at_target=False,
                            )
                            print(
                                f"progress records={written_records} bytes_gib={total_bytes / 1024**3:.3f} "
                                f"accepted={cleaner.report.accepted} seen={cleaner.report.seen} "
                                f"en_gib={written_language_bytes.get('en', 0) / 1024**3:.3f} "
                                f"zh_gib={written_language_bytes.get('zh', 0) / 1024**3:.3f}",
                                flush=True,
                            )
                if stopped_at_target:
                    break
            stats["exhausted"] = source_exhausted
            gc.collect()
            if stopped_at_target:
                break
    finally:
        if handle is not None:
            handle.close()
        rust_cleaner.close()

    write_jsonl(report_dir / "accepted_sample.jsonl", samples)
    write_stream_report(
        report_dir,
        cleaner,
        source_stats,
        policy,
        args,
        total_bytes,
        written_records,
        count_shard_files(shard_dir),
        written_language_bytes,
        written_language_records,
        stream_skipped,
        rust_cleaner.to_dict(),
        stopped_at_target=stopped_at_target,
    )
    (report_dir / "quality_report.md").write_text(
        render_markdown_report(cleaner.report, train_count=written_records, valid_count=0),
        encoding="utf-8",
    )
    print(
        f"stream_cleaned records={written_records} bytes_gib={total_bytes / 1024**3:.3f} "
        f"target_gib={args.target_gb:.3f} shards={count_shard_files(shard_dir)} out={out_dir}",
        flush=True,
    )


def should_emit_progress(
    *,
    written_records: int,
    progress_every: int,
    now: float,
    last_progress_at: float,
    min_seconds: float,
) -> bool:
    if progress_every <= 0:
        return False
    if written_records <= 0 or written_records % progress_every != 0:
        return False
    if min_seconds <= 0:
        return True
    return now - last_progress_at >= min_seconds


def write_stream_report(
    report_dir: Path,
    cleaner: StreamingDialogueCleaner,
    source_stats: dict[str, dict[str, int | bool]],
    policy: CleanPolicy,
    args: argparse.Namespace,
    total_bytes: int,
    written_records: int,
    shards: int,
    written_language_bytes: dict[str, int],
    written_language_records: dict[str, int],
    stream_skipped: dict[str, int],
    fast_filter: dict,
    *,
    stopped_at_target: bool,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "target_gb": args.target_gb,
            "target_bytes": int(args.target_gb * 1024**3),
            "written_bytes": total_bytes,
            "written_gib": total_bytes / 1024**3,
            "written_records": written_records,
            "shards": shards,
            "stopped_at_target": stopped_at_target,
            "language_byte_targets": parse_language_ratios(args.language_ratio, int(args.target_gb * 1024**3)),
            "language_written_bytes": written_language_bytes,
            "language_written_gib": {
                lang: value / 1024**3 for lang, value in sorted(written_language_bytes.items())
            },
            "language_written_records": written_language_records,
            "stream_skipped": stream_skipped,
        },
        "policy": {
            "min_score": args.min_score,
            "allowed_licenses": sorted(policy.allowed_licenses),
            "allowed_languages": sorted(policy.allowed_languages),
            "target_categories": sorted(policy.target_categories) if policy.target_categories else None,
            "language_ratios": parse_language_ratio_values(args.language_ratio),
            "no_raw_rows_saved": True,
            "cache_dir": str(Path(args.cache_dir)),
            "hf_endpoint": args.hf_endpoint,
        },
        "fast_filter": fast_filter,
        "source_stats": source_stats,
        "clean_report": cleaner.report.to_dict(),
        "outputs": {
            "shard_dir": str(Path(args.out_dir) / "clean" / "shards"),
            "accepted_sample_jsonl": str(Path(args.out_dir) / "reports" / "accepted_sample.jsonl"),
        },
    }
    (report_dir / "quality_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_language_ratio_values(raw_values: list[str] | None) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for item in raw_values or []:
        if "=" not in item:
            raise ValueError(f"Invalid language ratio: {item}")
        lang, value = item.split("=", 1)
        lang = lang.strip().lower()
        if not lang:
            raise ValueError(f"Invalid language ratio: {item}")
        ratios[lang] = float(value)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Language ratio sum must be positive.")
    return {lang: value / total for lang, value in sorted(ratios.items())}


def parse_language_ratios(raw_values: list[str] | None, target_bytes: int) -> dict[str, int]:
    if target_bytes <= 0:
        return {}
    ratios = parse_language_ratio_values(raw_values)
    targets = {lang: int(target_bytes * ratio) for lang, ratio in ratios.items()}
    remainder = target_bytes - sum(targets.values())
    if remainder and targets:
        first = sorted(targets)[0]
        targets[first] += remainder
    return targets


def prepare_rust_cleaner(args: argparse.Namespace) -> RustCleaner:
    binary = find_rust_prefilter_binary(args.fast_filter_bin)
    active = False
    if args.fast_filter == "off":
        return RustCleaner(mode=args.fast_filter, binary=binary, batch_size=args.fast_filter_batch_size, active=False)
    if binary is not None and binary.exists():
        active = True
    elif args.fast_filter == "on":
        raise RuntimeError(
            "Rust cleaner requested with --fast-filter on, but binary was not found. "
            "Build it with: cargo build --release --manifest-path rust/dopa_dialogue_filter/Cargo.toml"
        )
    else:
        print("Rust cleaner binary not found; falling back to Python cleaning.", file=sys.stderr)
    return RustCleaner(mode=args.fast_filter, binary=binary, batch_size=args.fast_filter_batch_size, active=active)


def prepare_rust_prefilter(args: argparse.Namespace) -> RustPrefilter:
    binary = find_rust_prefilter_binary(args.fast_filter_bin)
    active = False
    if args.fast_filter == "off":
        return RustPrefilter(mode=args.fast_filter, binary=binary, batch_size=args.fast_filter_batch_size, active=False)
    if binary is not None and binary.exists():
        active = True
    elif args.fast_filter == "on":
        raise RuntimeError(
            "Rust fast filter requested with --fast-filter on, but binary was not found. "
            "Build it with: cargo build --release --manifest-path rust/dopa_dialogue_filter/Cargo.toml"
        )
    else:
        print("Rust fast filter binary not found; falling back to Python-only filtering.", file=sys.stderr)
    return RustPrefilter(mode=args.fast_filter, binary=binary, batch_size=args.fast_filter_batch_size, active=active)


def find_rust_prefilter_binary(raw_path: str) -> Path | None:
    if raw_path:
        return Path(raw_path)
    suffix = ".exe" if os.name == "nt" else ""
    candidate = ROOT / "rust" / "dopa_dialogue_filter" / "target" / "release" / f"dopa_dialogue_filter{suffix}"
    return candidate


def iter_batches(items: Iterator[ConversationRecord], batch_size: int) -> Iterator[list[ConversationRecord]]:
    batch: list[ConversationRecord] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass(frozen=True)
class AcceptedTrainingLine:
    json_line: str
    encoded_len: int
    language: str
    category: str
    source: str


def build_cleaner_rows(conversations: list[ConversationRecord]) -> list[str]:
    return list(iter_cleaner_rows(conversations))


def iter_cleaner_rows(conversations: list[ConversationRecord]) -> Iterator[str]:
    for index, conversation in enumerate(conversations):
        metadata = conversation.metadata if isinstance(conversation.metadata, Mapping) else {}
        language = str(metadata.get("language", ""))
        tree_id = metadata.get("message_tree_id")
        thread_key = "" if tree_id in {"", None} else f"{conversation.source}:{tree_id}"
        metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        text = _tsv_field(conversation.normalized_text())
        yield "\t".join(
            [
                str(index),
                _tsv_field(conversation.source),
                _tsv_field(conversation.source_id),
                _tsv_field(conversation.license_id),
                _tsv_field(language),
                str(len(conversation.messages)),
                _tsv_field(thread_key),
                _tsv_field(metadata_json),
                text,
            ]
        )


def parse_cleaner_outputs(body: str, *, seen: int) -> RustCleanBatchResult:
    accepted: list[RustCleanedRecord] = []
    rejected: Counter = Counter()
    for line in body.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 9)
        if len(parts) < 3:
            raise RuntimeError(f"Invalid Rust cleaner output line: {line}")
        index, action, reason = parts[:3]
        if action == "reject":
            rejected[reason] += 1
            continue
        if action != "accept" or len(parts) != 10:
            raise RuntimeError(f"Invalid Rust cleaner accept line: {line}")
        tags = tuple(item for item in parts[6].split(",") if item)
        json_line = parts[9]
        if not json_line.endswith("\n"):
            json_line += "\n"
        accepted.append(
            RustCleanedRecord(
                index=index,
                language=parts[3],
                category=parts[4],
                quality_score=float(parts[5]),
                quality_tags=tags,
                encoded_len=int(parts[7]),
                source=parts[8],
                json_line=json_line,
            )
        )
    return RustCleanBatchResult(accepted=accepted, rejected=rejected, seen=seen)


def iter_rust_cleaned_records(
    result: RustCleanBatchResult,
    cleaner: StreamingDialogueCleaner,
) -> Iterator[AcceptedTrainingLine]:
    for item in result.accepted:
        cleaner.report.accepted += 1
        cleaner.report.categories[item.category] += 1
        cleaner.report.sources[item.source] += 1
        yield AcceptedTrainingLine(
            json_line=item.json_line,
            encoded_len=item.encoded_len,
            language=item.language,
            category=item.category,
            source=item.source,
        )


def iter_python_cleaned_records(
    conversations: list[ConversationRecord],
    cleaner: StreamingDialogueCleaner,
) -> Iterator[AcceptedTrainingLine]:
    for conversation in conversations:
        cleaned = cleaner.accept(conversation)
        if cleaned is None:
            continue
        record = cleaned.to_training_record()
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        encoded_len = len(line.encode("utf-8"))
        yield AcceptedTrainingLine(
            json_line=line,
            encoded_len=encoded_len,
            language=str(record.get("language") or language_group(cleaned)),
            category=str(record.get("category") or cleaned.category),
            source=cleaned.source,
        )


def build_prefilter_rows(conversations: list[ConversationRecord]) -> list[str]:
    return list(iter_prefilter_rows(conversations))


def iter_prefilter_rows(conversations: list[ConversationRecord]) -> Iterator[str]:
    for index, conversation in enumerate(conversations):
        language = str(conversation.metadata.get("language", ""))
        text = _tsv_field(conversation.normalized_text())
        yield "\t".join(
            [
                str(index),
                _tsv_field(conversation.license_id),
                _tsv_field(language),
                str(len(conversation.messages)),
                text,
            ]
        )


def run_rust_prefilter_batch(
    binary: Path,
    conversations: list[ConversationRecord],
    *,
    policy: CleanPolicy,
    cache_dir: Path,
) -> dict[str, tuple[str, str]]:
    body = "\n".join(iter_prefilter_rows(conversations))
    if body:
        body += "\n"
    cmd = [
        str(binary),
        "filter-stdin",
        str(policy.min_chars),
        str(policy.max_chars),
        ",".join(sorted(policy.allowed_licenses)),
        ",".join(sorted(policy.allowed_languages)),
    ]
    proc = subprocess.run(
        cmd,
        input=body,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Rust prefilter failed: {_completed_process_message(proc)}")
    return parse_prefilter_verdicts(proc.stdout)


def _completed_process_message(proc: subprocess.CompletedProcess) -> str:
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    return stderr or stdout or f"exit code {proc.returncode}"


def parse_prefilter_verdicts(body: str) -> dict[str, tuple[str, str]]:
    verdicts: dict[str, tuple[str, str]] = {}
    for line in body.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise RuntimeError(f"Invalid Rust prefilter output line: {line}")
        verdicts[parts[0]] = (parts[1], parts[2])
    return verdicts


def apply_prefilter_verdicts(
    conversations: list[ConversationRecord],
    verdicts: dict[str, tuple[str, str]],
) -> tuple[list[ConversationRecord], Counter]:
    accepted: list[ConversationRecord] = []
    rejected: Counter = Counter()
    for index, conversation in enumerate(conversations):
        action, reason = verdicts.get(str(index), ("accept", "accepted"))
        if action == "accept":
            accepted.append(conversation)
        elif action == "reject":
            rejected[reason] += 1
        else:
            raise RuntimeError(f"Unknown Rust prefilter action for row {index}: {action}")
    return accepted, rejected


def _tsv_field(value: object) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", "\\n")


def iter_source_conversations(
    spec: SourceSpec,
    *,
    limit: int,
    backend: str,
    hf_endpoint: str,
    cache_dir: Path,
    page_size: int = 100,
    timeout: float = 60.0,
) -> Iterator[ConversationRecord]:
    if backend in {"rows-api", "hf-mirror-parquet"}:
        if backend == "rows-api":
            row_iter = iter_rows_api(spec, limit=limit, page_size=page_size, timeout=timeout)
        else:
            row_iter = iter_hf_mirror_parquet_rows(
                spec,
                limit=limit,
                hf_endpoint=hf_endpoint,
                cache_dir=cache_dir,
                timeout=timeout,
            )
        if spec.converter == "openassistant":
            if limit <= 0:
                raise RuntimeError(f"{backend} backend for OpenAssistant requires a positive --limit-per-source.")
            yield from convert_rows(spec, list(row_iter))
            return
        for row in row_iter:
            for conversation in convert_rows(spec, [row]):
                yield conversation
        return
    if backend == "auto":
        try:
            yield from iter_source_conversations(
                spec,
                limit=limit,
                backend="datasets",
                hf_endpoint=hf_endpoint,
                cache_dir=cache_dir,
                page_size=page_size,
                timeout=timeout,
            )
            return
        except Exception as exc:
            print(f"{spec.name}: datasets backend failed, falling back to streaming rows-api: {exc}", file=sys.stderr)
            fallback_backend = "hf-mirror-parquet" if hf_endpoint else "rows-api"
            yield from iter_source_conversations(
                spec,
                limit=limit,
                backend=fallback_backend,
                hf_endpoint=hf_endpoint,
                cache_dir=cache_dir,
                page_size=page_size,
                timeout=timeout,
            )
            return
    if spec.converter == "openassistant":
        yield from iter_openassistant_conversations_by_tree(
            iter_dataset_rows(spec, limit=limit, hf_endpoint=hf_endpoint, cache_dir=cache_dir),
            source=spec.name,
            license_id=spec.license_id,
        )
        return
    if spec.converter == "wildchat":
        for row in iter_dataset_rows(spec, limit=limit, hf_endpoint=hf_endpoint, cache_dir=cache_dir):
            conversation = normalize_wildchat_row(row, source=spec.name, license_id=spec.license_id)
            if conversation is not None:
                yield conversation
        return
    if spec.converter in {
        "prompt_target",
        "prompt_target_en_zh",
        "helpsteer3_preference",
        "hh_rlhf",
        "dolly",
        "stack_exchange",
        "openorca",
    }:
        for row in iter_dataset_rows(spec, limit=limit, hf_endpoint=hf_endpoint, cache_dir=cache_dir):
            for conversation in convert_rows(spec, [row]):
                yield conversation
        return
    raise ValueError(f"Unsupported converter: {spec.converter}")


def configure_hf_cache(cache_dir: Path, hf_endpoint: str) -> None:
    if str(cache_dir).lower().startswith("c:\\"):
        raise RuntimeError(f"Refusing to use C-drive cache for corpus downloads: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ["HF_HOME"] = str(cache_dir / "home")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    for key in ("HF_HOME", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def fetch_hf_rows(
    spec: SourceSpec,
    *,
    limit: int,
    page_size: int,
    timeout: float,
    backend: str,
    hf_endpoint: str,
    cache_dir: Path,
) -> list[dict]:
    if backend == "datasets":
        return _fetch_dataset_rows(spec, limit=limit, hf_endpoint=hf_endpoint, cache_dir=cache_dir)
    if backend == "auto":
        try:
            return _fetch_dataset_rows(spec, limit=limit, hf_endpoint=hf_endpoint, cache_dir=cache_dir)
        except Exception as exc:
            print(f"{spec.name}: datasets backend failed, falling back to rows-api: {exc}", file=sys.stderr)
    return _fetch_rows_api(spec, limit=limit, page_size=page_size, timeout=timeout)


def _fetch_rows_api(
    spec: SourceSpec,
    *,
    limit: int,
    page_size: int,
    timeout: float,
) -> list[dict]:
    return list(iter_rows_api(spec, limit=limit, page_size=page_size, timeout=timeout))


def iter_rows_api(
    spec: SourceSpec,
    *,
    limit: int,
    page_size: int,
    timeout: float,
) -> Iterator[dict]:
    emitted = 0
    offset = 0
    page_size = max(1, page_size)
    while limit <= 0 or emitted < limit:
        length = page_size if limit <= 0 else min(page_size, limit - emitted)
        params = urllib.parse.urlencode(
            {
                "dataset": spec.dataset,
                "config": spec.config,
                "split": spec.split,
                "offset": offset,
                "length": length,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{params}"
        payload = _fetch_json(url, timeout=timeout)
        page_rows = [item["row"] for item in payload.get("rows", []) if isinstance(item.get("row"), dict)]
        if not page_rows:
            break
        for row in page_rows:
            if limit > 0 and emitted >= limit:
                break
            emitted += 1
            yield row
        offset += len(page_rows)
        if len(page_rows) < length:
            break
        time.sleep(0.2)


def iter_hf_mirror_parquet_rows(
    spec: SourceSpec,
    *,
    limit: int,
    hf_endpoint: str,
    cache_dir: Path,
    timeout: float,
) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install pyarrow to use --backend hf-mirror-parquet.") from exc

    emitted = 0
    for filename in _hf_mirror_dataset_files(spec, hf_endpoint=hf_endpoint, timeout=timeout):
        parquet_path = _download_hf_mirror_file(
            spec,
            filename,
            hf_endpoint=hf_endpoint,
            cache_dir=cache_dir,
            timeout=timeout,
        )
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if limit > 0 and emitted >= limit:
                    return
                if isinstance(row, dict):
                    emitted += 1
                    yield row


def _hf_mirror_dataset_files(spec: SourceSpec, *, hf_endpoint: str, timeout: float) -> list[str]:
    endpoint = hf_endpoint.rstrip("/") or "https://hf-mirror.com"
    dataset_path = urllib.parse.quote(spec.dataset, safe="/")
    payload = _fetch_json(f"{endpoint}/api/datasets/{dataset_path}", timeout=timeout)
    siblings = payload.get("siblings") or []
    files = [
        str(item.get("rfilename") or "")
        for item in siblings
        if isinstance(item, dict) and str(item.get("rfilename") or "").endswith(".parquet")
    ]
    candidate_files = files
    if spec.config and spec.config != "default":
        config_markers = (
            f"/{spec.config}/",
            f"data/{spec.config}/",
            f"{spec.config}/",
        )
        config_files = [
            filename
            for filename in candidate_files
            if any(marker in f"/{filename}" or filename.startswith(marker) for marker in config_markers)
        ]
        if config_files:
            candidate_files = config_files
    split_marker = f"/{spec.split}-"
    split_files = [filename for filename in candidate_files if split_marker in f"/{filename}"]
    return sorted(split_files or candidate_files)


def _download_hf_mirror_file(
    spec: SourceSpec,
    filename: str,
    *,
    hf_endpoint: str,
    cache_dir: Path,
    timeout: float,
) -> Path:
    endpoint = hf_endpoint.rstrip("/") or "https://hf-mirror.com"
    dataset_safe = spec.dataset.replace("/", "__")
    local_path = cache_dir / "mirror_parquet" / dataset_safe / Path(filename)
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_filename = urllib.parse.quote(filename, safe="/")
    dataset_path = urllib.parse.quote(spec.dataset, safe="/")
    url = f"{endpoint}/datasets/{dataset_path}/resolve/main/{encoded_filename}"
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "dopa-coder-n1-dialogue-cleaner/0.1",
            "Accept": "application/octet-stream",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp_path.replace(local_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return local_path


def _fetch_dataset_rows(
    spec: SourceSpec,
    *,
    limit: int,
    hf_endpoint: str,
    cache_dir: Path,
) -> list[dict]:
    configure_hf_cache(cache_dir, hf_endpoint)
    return list(iter_dataset_rows(spec, limit=limit, hf_endpoint=hf_endpoint, cache_dir=cache_dir))


def iter_dataset_rows(
    spec: SourceSpec,
    *,
    limit: int,
    hf_endpoint: str,
    cache_dir: Path,
) -> Iterator[dict]:
    configure_hf_cache(cache_dir, hf_endpoint)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional corpus dependency: pip install datasets pyarrow") from exc

    dataset = load_dataset(
        spec.dataset,
        spec.config,
        split=spec.split,
        cache_dir=str(cache_dir),
        streaming=True,
    )
    for index, row in enumerate(dataset):
        if limit > 0 and index >= limit:
            break
        yield dict(row)


def _fetch_json(url: str, *, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "dopa-coder-n1-dialogue-cleaner/0.1",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} fetching {url}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error fetching {url}: {exc}") from exc


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
