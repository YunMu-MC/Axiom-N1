from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Iterator


Message = dict[str, str]


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)\b("
    r"sk-[A-Za-z0-9_-]{16,}|"
    r"(api[_-]?key|secret|token|password)\s*[:=]\s*[A-Za-z0-9_./+=-]{12,}|"
    r"AKIA[0-9A-Z]{16}"
    r")\b"
)
_URL_ONLY_RE = re.compile(r"^\s*(https?://\S+\s*){1,3}$", re.IGNORECASE)
_CODE_HINT_RE = re.compile(
    r"```|\bTraceback\b|\bpytest\b|\bunittest\b|\bdef\s+\w+|\bclass\s+\w+|\bfunction\b|"
    r"\bTypeError\b|\bValueError\b|\bcompiler\b|\bruntime error\b|\bstack trace\b|\bexception\b",
    re.IGNORECASE,
)
_TOOL_HINT_RE = re.compile(
    r"\b(tool|function call|schema|json schema|argument|sandbox|terminal|command)\b",
    re.IGNORECASE,
)
_SECURITY_HINT_RE = re.compile(
    r"\b(CVE|CWE|vulnerability|sanitize|injection|XSS|SQLi|SSRF|auth bypass|"
    r"access control|exploitability|security audit|patch diff|secure coding)\b",
    re.IGNORECASE,
)
_REASONING_HINT_RE = re.compile(r"\b(prove|derive|step by step|why|because|therefore)\b", re.IGNORECASE)
_ZH_DEBUG_HINT_RE = re.compile(r"报错|错误|异常|堆栈|调试|复现|修复|回归测试|单元测试|编译失败|运行失败")
_ZH_TOOL_HINT_RE = re.compile(r"工具调用|工具|命令|终端|接口|函数调用|JSON|模式|沙箱")
_ZH_SECURITY_HINT_RE = re.compile(r"漏洞|注入|越权|鉴权|权限绕过|安全审计|攻击面|补丁|修补|XSS|SSRF|SQL注入")
_ZH_REASONING_HINT_RE = re.compile(r"为什么|原因|步骤|推导|证明|因为|所以|怎么")
_MOJIBAKE_RE = re.compile(r"�|Ã.|â€|鈥|檚|搒|揷|榚|茅|铆|€|缁|涓€|璇|鐢|鍚|鏁|鐓|楠|姹")
_CHILD_PERSONA_RE = re.compile(
    r"\b(pretend|roleplay|act)\b.{0,80}\b(6|six|7|seven|8|eight)\s*(year[- ]old|yo)\s+"
    r"(girl|boy|child)|\b(call me mommy|hi mommy|baby girl)\b",
    re.IGNORECASE,
)
_UNSAFE_SECURITY_RE = re.compile(
    r"\b(passcode|phone|pin|password)\b.{0,500}\b("
    r"guess|brute force|generate combinations|test each combination|try every"
    r")\b|\b(guess|brute force|generate combinations|test each combination|try every)\b.{0,500}"
    r"\b(passcode|phone|pin|password)\b",
    re.IGNORECASE | re.DOTALL,
)
_FINANCIAL_MARKET_RE = re.compile(
    r"\b("
    r"cryptocurrency|crypto|bitcoin|ethereum|stock alert|stock price|NVIDIA stock|NVDA|"
    r"investing plan|investment decision|financial advisor|portfolio|market timing|"
    r"market volatility|dollar-cost averaging"
    r")\b",
    re.IGNORECASE,
)
_BAD_CODE_ARTIFACT_RE = re.compile(
    r"\bFlask\(name\)|\bif\s+name\s*==\s*['\"]main['\"]|python\s+Copy code|\bblenderpy\b",
    re.IGNORECASE,
)
_SECRET_MARKERS = ("sk-", "api_key", "api-key", "secret", "token", "password", "akia")
_FINANCIAL_MARKET_MARKERS = (
    "cryptocurrency",
    "crypto",
    "bitcoin",
    "ethereum",
    "stock",
    "nvda",
    "invest",
    "financial advisor",
    "portfolio",
    "market",
)
_BAD_CODE_MARKERS = ("flask(", "if name", "copy code", "blenderpy")
_CHILD_PERSONA_MARKERS = ("pretend", "roleplay", "act", "mommy", "baby girl")
_UNSAFE_SECURITY_SENSITIVE_MARKERS = ("passcode", "phone", "pin", "password")
_UNSAFE_SECURITY_ACTION_MARKERS = (
    "guess",
    "brute force",
    "generate combinations",
    "test each combination",
    "try every",
)
_CODE_HINT_MARKERS = (
    "```",
    "traceback",
    "pytest",
    "unittest",
    "def ",
    "class ",
    "function",
    "typeerror",
    "valueerror",
    "compiler",
    "runtime error",
    "stack trace",
    "exception",
)
_TOOL_HINT_MARKERS = ("tool", "function call", "schema", "json schema", "argument", "sandbox", "terminal", "command")
_SECURITY_HINT_MARKERS = (
    "cve",
    "cwe",
    "vulnerability",
    "sanitize",
    "injection",
    "xss",
    "sqli",
    "ssrf",
    "auth bypass",
    "access control",
    "exploitability",
    "security audit",
    "patch diff",
    "secure coding",
)
_REASONING_HINT_MARKERS = ("prove", "derive", "step by step", "why", "because", "therefore")


@dataclass(frozen=True)
class ConversationRecord:
    source: str
    license_id: str
    source_id: str
    messages: list[Message]
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    quality_score: float = 0.0
    category: str = "general"
    quality_tags: tuple[str, ...] = ()
    _normalized_text_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def normalized_text(self) -> str:
        if self._normalized_text_cache is not None:
            return self._normalized_text_cache
        lines: list[str] = []
        for message in self.messages:
            role = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{role}: {message['content'].strip()}")
        text = "\n\n".join(lines).strip()
        object.__setattr__(self, "_normalized_text_cache", text)
        return text

    def fingerprint(self) -> str:
        normalized = _normalize_for_hash(self.normalized_text())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def to_training_record(self) -> dict:
        text = self.normalized_text()
        return {
            "text": text,
            "language": language_group(self, text=text),
            "source": self.source,
            "source_id": self.source_id,
            "license": self.license_id,
            "category": self.category,
            "quality_score": round(self.quality_score, 4),
            "quality_tags": list(self.quality_tags),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CleanPolicy:
    allowed_licenses: frozenset[str] = frozenset({"apache-2.0", "odc-by", "cc-by-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0", "mit"})
    allowed_languages: frozenset[str] = frozenset(
        {
            "en",
            "eng",
            "english",
            "zh",
            "zh-cn",
            "zh-hans",
            "zh-hant",
            "zho",
            "cmn",
            "chinese",
            "simplified chinese",
            "traditional chinese",
        }
    )
    min_turns: int = 2
    max_turns: int = 24
    min_chars: int = 80
    max_chars: int = 12_000
    min_score: float = 0.45
    keep_rejected_samples: int = 200
    target_categories: frozenset[str] | None = None


@dataclass
class CleanReport:
    seen: int = 0
    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    categories: Counter = field(default_factory=Counter)
    sources: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "seen": self.seen,
            "accepted": self.accepted,
            "rejected": dict(sorted(self.rejected.items())),
            "categories": dict(sorted(self.categories.items())),
            "sources": dict(sorted(self.sources.items())),
        }


@dataclass(frozen=True)
class ShardWriteReport:
    records: int
    bytes_written: int
    shards: int
    language_records: dict[str, int] = field(default_factory=dict)
    language_bytes: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)


class StreamingDialogueCleaner:
    def __init__(self, policy: CleanPolicy):
        self.policy = policy
        self.report = CleanReport()
        self._fingerprints: set[bytes] = set()
        self._thread_keys: set[str] = set()

    def remember_training_record(self, record: dict) -> bool:
        text = _string(record.get("text")).strip()
        if not text:
            return False
        normalized = _normalize_for_hash(text)
        self._fingerprints.add(hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest())

        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return True
        tree_id = metadata.get("message_tree_id")
        if tree_id not in {"", None}:
            self._thread_keys.add(f"{_string(record.get('source'))}:{tree_id}")
        return True

    def accept(self, conversation: ConversationRecord | None) -> ConversationRecord | None:
        if conversation is None:
            return None
        self.report.seen += 1
        reason = _pre_text_reject_reason(conversation, self.policy)
        if reason is not None:
            self.report.rejected[reason] += 1
            return None
        text = conversation.normalized_text()
        lower = text.lower()
        reason = _text_reject_reason(text, self.policy, lower=lower)
        if reason is not None:
            self.report.rejected[reason] += 1
            return None
        score, category, tags = score_conversation(conversation, text=text, lower=lower)
        if score < self.policy.min_score:
            self.report.rejected["low_quality"] += 1
            return None
        if self.policy.target_categories is not None and category not in self.policy.target_categories:
            self.report.rejected["category_not_target"] += 1
            return None
        thread_key = _thread_key(conversation)
        if thread_key is not None and thread_key in self._thread_keys:
            self.report.rejected["duplicate_thread"] += 1
            return None
        fingerprint = _fingerprint_digest(conversation, text=text)
        if fingerprint in self._fingerprints:
            self.report.rejected["duplicate"] += 1
            return None
        cleaned_messages = _clean_messages_for_record(conversation.messages)
        cleaned = replace(
            conversation,
            messages=cleaned_messages,
            quality_score=score,
            category=category,
            quality_tags=tuple(tags),
        )
        if cleaned_messages is conversation.messages:
            object.__setattr__(cleaned, "_normalized_text_cache", text)
        self._fingerprints.add(fingerprint)
        if thread_key is not None:
            self._thread_keys.add(thread_key)
        self.report.accepted += 1
        self.report.categories[category] += 1
        self.report.sources[conversation.source] += 1
        return cleaned


def normalize_wildchat_row(
    row: dict,
    *,
    source: str,
    license_id: str,
) -> ConversationRecord | None:
    raw_messages = row.get("conversation") or row.get("messages") or []
    messages = _normalize_messages(raw_messages)
    if not messages:
        return None
    language = _string(row.get("language") or row.get("lang") or "").strip().lower()
    metadata = {
        "model": _string(row.get("model") or row.get("model_name") or ""),
        "language": language,
        "redacted": bool(row.get("redacted", False)),
    }
    source_id = _string(row.get("conversation_hash") or row.get("id") or row.get("conversation_id"))
    if not source_id:
        source_id = hashlib.sha256(_messages_text(messages).encode("utf-8")).hexdigest()[:16]
    return ConversationRecord(
        source=source,
        license_id=license_id.lower(),
        source_id=source_id,
        messages=messages,
        metadata={key: value for key, value in metadata.items() if value not in {"", None}},
    )


def normalize_prompt_target_row(
    row: dict,
    *,
    source: str,
    license_id: str,
    prompt_field: str = "prompt",
    target_field: str = "response",
) -> ConversationRecord | None:
    prompt = _clean_text(_string(row.get(prompt_field)))
    target = _clean_text(_string(row.get(target_field)))
    if not prompt or not target:
        return None
    language = _first_nonempty(row, ("language_code", "language", "lang", "locale")).strip().lower()
    metadata = {
        "language": language,
        "domain": _first_nonempty(row, ("domain", "task_type", "subdomain")),
        "annotation_type": _string(row.get("annotation_type")),
    }
    source_id = _first_nonempty(row, ("id", "example_id", "sample_id", "question_id"))
    if not source_id:
        source_id = hashlib.sha256(f"{prompt}\n{target}".encode("utf-8")).hexdigest()[:16]
    return ConversationRecord(
        source=source,
        license_id=license_id.lower(),
        source_id=source_id,
        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": target}],
        metadata={key: value for key, value in metadata.items() if value not in {"", None}},
    )


def normalize_helpsteer3_preference_row(
    row: dict,
    *,
    source: str,
    license_id: str,
) -> ConversationRecord | None:
    preference = _int_or_none(row.get("overall_preference"))
    if preference is None or preference == 0:
        return None
    chosen_key = "response2" if preference > 0 else "response1"
    chosen_response = _clean_text(_string(row.get(chosen_key)))
    if not chosen_response:
        return None
    messages = _normalize_messages(row.get("context") or [])
    if messages and messages[-1]["role"] == "user":
        messages.append({"role": "assistant", "content": chosen_response})
    elif not messages:
        prompt = _clean_text(_first_nonempty(row, ("prompt", "question", "instruction")))
        if not prompt:
            return None
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen_response}]
    elif messages[-1]["role"] == "assistant":
        messages[-1] = {"role": "assistant", "content": chosen_response}
    if not _is_alternating(messages):
        return None
    metadata = {
        "domain": _string(row.get("domain")),
        "language": _first_nonempty(row, ("language", "language_code", "locale")).strip().lower(),
        "preference": preference,
        "chosen_response": chosen_key,
    }
    source_id = _first_nonempty(row, ("id", "sample_id", "conversation_id"))
    if not source_id:
        source_id = hashlib.sha256(_messages_text(messages).encode("utf-8")).hexdigest()[:16]
    return ConversationRecord(
        source=source,
        license_id=license_id.lower(),
        source_id=source_id,
        messages=messages,
        metadata={key: value for key, value in metadata.items() if value not in {"", None}},
    )


def normalize_hh_rlhf_row(
    row: dict,
    *,
    source: str,
    license_id: str,
) -> ConversationRecord | None:
    transcript = _string(row.get("chosen"))
    messages = _parse_hh_rlhf_transcript(transcript)
    if not messages:
        return None
    source_id = _first_nonempty(row, ("id", "sample_id"))
    if not source_id:
        source_id = hashlib.sha256(_messages_text(messages).encode("utf-8")).hexdigest()[:16]
    return ConversationRecord(
        source=source,
        license_id=license_id.lower(),
        source_id=source_id,
        messages=messages,
        metadata={"language": "en"},
    )


def build_openassistant_conversations(
    rows: Iterable[dict],
    *,
    source: str,
    license_id: str,
) -> list[ConversationRecord]:
    nodes: dict[str, dict] = {}
    children: dict[str | None, list[dict]] = {}
    for row in rows:
        if _is_rejected_openassistant_row(row):
            continue
        message_id = _string(row.get("message_id") or row.get("id"))
        if not message_id:
            continue
        role = _normalize_role(row.get("role"))
        if role not in {"user", "assistant"}:
            continue
        text = _clean_text(_string(row.get("text") or row.get("content")))
        if not text:
            continue
        node = {
            "id": message_id,
            "parent_id": row.get("parent_id"),
            "tree_id": _string(row.get("message_tree_id") or row.get("tree_id") or message_id),
            "role": role,
            "content": text,
            "rank": _rank_value(row.get("rank")),
            "lang": _string(row.get("lang") or row.get("language") or "").lower(),
        }
        nodes[message_id] = node
        children.setdefault(node["parent_id"], []).append(node)

    leaf_nodes = [node for node in nodes.values() if node["id"] not in children and node["role"] == "assistant"]
    candidates: list[ConversationRecord] = []
    used_fingerprints: set[str] = set()
    for leaf in sorted(leaf_nodes, key=lambda node: (node["tree_id"], node["rank"])):
        path = _path_to_root(leaf, nodes)
        messages = [{"role": node["role"], "content": node["content"]} for node in path]
        if not _is_alternating(messages) or len(messages) < 2:
            continue
        tree_id = leaf["tree_id"]
        conversation = ConversationRecord(
            source=source,
            license_id=license_id.lower(),
            source_id=f"{tree_id}:{leaf['id']}",
            messages=messages,
            metadata={"message_tree_id": tree_id, "language": leaf.get("lang") or ""},
        )
        fingerprint = conversation.fingerprint()
        if fingerprint in used_fingerprints:
            continue
        used_fingerprints.add(fingerprint)
        candidates.append(conversation)
    return candidates


def iter_openassistant_conversations_by_tree(
    rows: Iterable[dict],
    *,
    source: str,
    license_id: str,
) -> Iterator[ConversationRecord]:
    """Build OpenAssistant paths while holding only one consecutive message tree.

    Hugging Face OpenAssistant splits are grouped by `message_tree_id` in normal
    streaming order. Processing consecutive trees avoids materializing the whole
    split before cleaning begins.
    """
    current_tree_id: str | None = None
    tree_rows: list[dict] = []
    for row in rows:
        tree_id = _openassistant_tree_id(row)
        if current_tree_id is None:
            current_tree_id = tree_id
        if tree_id != current_tree_id:
            yield from build_openassistant_conversations(
                tree_rows,
                source=source,
                license_id=license_id,
            )
            tree_rows = []
            current_tree_id = tree_id
        tree_rows.append(row)
    if tree_rows:
        yield from build_openassistant_conversations(
            tree_rows,
            source=source,
            license_id=license_id,
        )


def clean_conversations(
    conversations: Iterable[ConversationRecord | None],
    policy: CleanPolicy,
) -> tuple[list[ConversationRecord], CleanReport]:
    cleaner = StreamingDialogueCleaner(policy)
    accepted: list[ConversationRecord] = []
    for conversation in conversations:
        cleaned = cleaner.accept(conversation)
        if cleaned is not None:
            accepted.append(cleaned)
    return accepted, cleaner.report


def score_conversation(
    conversation: ConversationRecord,
    *,
    text: str | None = None,
    lower: str | None = None,
) -> tuple[float, str, list[str]]:
    if text is None:
        text = conversation.normalized_text()
    if lower is None:
        lower = text.lower()
    score = 0.25
    tags: list[str] = []
    category = "general"
    if len(conversation.messages) >= 4:
        score += 0.12
        tags.append("multi_turn")
    if len(text) >= 250:
        score += 0.10
        tags.append("substantive")
    if "?" in text or "？" in text or "how" in lower or "why" in lower or "怎么" in text:
        score += 0.05
        tags.append("question_answer")
    if _has_code_hint(text, lower):
        score += 0.20
        category = "debug" if _has_debug_failure_hint(lower) else "engineering"
        tags.append("code_or_debug")
    if _ZH_DEBUG_HINT_RE.search(text):
        score += 0.22
        category = "debug"
        tags.append("zh_debug")
    if _has_marker(lower, _TOOL_HINT_MARKERS) and _TOOL_HINT_RE.search(text):
        score += 0.12
        category = "tool_calling"
        tags.append("tool_use")
    if _ZH_TOOL_HINT_RE.search(text):
        score += 0.12
        category = "tool_calling"
        tags.append("zh_tool_use")
    if _has_marker(lower, _SECURITY_HINT_MARKERS) and _SECURITY_HINT_RE.search(text):
        score += 0.16
        category = "security_defensive"
        tags.append("security")
    if _ZH_SECURITY_HINT_RE.search(text):
        score += 0.16
        category = "security_defensive"
        tags.append("zh_security")
    if _has_marker(lower, _REASONING_HINT_MARKERS) and _REASONING_HINT_RE.search(text):
        score += 0.08
        tags.append("reasoning")
    if _ZH_REASONING_HINT_RE.search(text):
        score += 0.08
        tags.append("zh_reasoning")
    if _has_correction_pattern(text):
        score += 0.10
        tags.append("repair")
    if _generic_reply_ratio(conversation.messages) > 0.45:
        score -= 0.20
    return max(0.0, min(score, 1.0)), category, tags


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_sharded_jsonl(
    out_dir: str | Path,
    records: Iterable[dict],
    *,
    target_bytes: int,
    shard_bytes: int,
    language_byte_targets: dict[str, int] | None = None,
) -> ShardWriteReport:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    total_records = 0
    shard_index = 0
    shard_current = 0
    language_records: Counter = Counter()
    language_bytes: Counter = Counter()
    skipped: Counter = Counter()
    handle = None
    try:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            encoded = line.encode("utf-8")
            if target_bytes > 0 and total_bytes + len(encoded) > target_bytes:
                break
            lang = _record_language(record)
            if language_byte_targets is not None:
                if lang not in language_byte_targets:
                    skipped["language_not_target"] += 1
                    continue
                if language_bytes[lang] + len(encoded) > language_byte_targets[lang]:
                    skipped["language_quota_full"] += 1
                    continue
            if handle is None or (shard_bytes > 0 and shard_current + len(encoded) > shard_bytes):
                if handle is not None:
                    handle.close()
                handle = (out / f"shard-{shard_index:05d}.jsonl").open("w", encoding="utf-8", newline="\n")
                shard_index += 1
                shard_current = 0
            handle.write(line)
            total_bytes += len(encoded)
            shard_current += len(encoded)
            total_records += 1
            language_records[lang] += 1
            language_bytes[lang] += len(encoded)
    finally:
        if handle is not None:
            handle.close()
    return ShardWriteReport(
        records=total_records,
        bytes_written=total_bytes,
        shards=shard_index,
        language_records=dict(language_records),
        language_bytes=dict(language_bytes),
        skipped=dict(skipped),
    )


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_train_valid(
    records: list[ConversationRecord],
    *,
    valid_ratio: float,
    seed: int,
) -> tuple[list[ConversationRecord], list[ConversationRecord]]:
    if not records:
        return [], []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    valid_count = int(len(shuffled) * valid_ratio)
    if valid_ratio > 0 and len(shuffled) > 1:
        valid_count = max(1, valid_count)
    valid_count = min(valid_count, max(0, len(shuffled) - 1))
    valid = shuffled[:valid_count]
    train = shuffled[valid_count:]
    return train, valid


def render_markdown_report(report: CleanReport, *, train_count: int, valid_count: int) -> str:
    lines = [
        "# Dialogue Corpus Trial Quality Report",
        "",
        f"seen={report.seen} accepted={report.accepted} train={train_count} valid={valid_count}",
        "",
        "## Accepted Categories",
    ]
    if report.categories:
        for category, count in sorted(report.categories.items()):
            lines.append(f"- {category}: {count}")
    else:
        lines.append("- none: 0")
    lines.extend(["", "## Sources"])
    if report.sources:
        for source, count in sorted(report.sources.items()):
            lines.append(f"- {source}: {count}")
    else:
        lines.append("- none: 0")
    lines.extend(["", "## Rejections"])
    if report.rejected:
        for reason, count in sorted(report.rejected.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none: 0")
    lines.extend(
        [
            "",
            "## Quality Gate",
            "- Licenses must be allowlisted.",
            "- Role order must be user/assistant alternating and end with assistant.",
            "- PII and secret-like tokens are rejected instead of redacted into training data.",
            "- Exact normalized duplicates are removed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _hard_reject_reason(
    conversation: ConversationRecord,
    policy: CleanPolicy,
    *,
    text: str | None = None,
) -> str | None:
    reason = _pre_text_reject_reason(conversation, policy)
    if reason is not None:
        return reason
    if text is None:
        text = conversation.normalized_text()
    return _text_reject_reason(text, policy)


def _pre_text_reject_reason(conversation: ConversationRecord, policy: CleanPolicy) -> str | None:
    if conversation.license_id.lower() not in policy.allowed_licenses:
        return "license_not_allowed"
    language = _string(conversation.metadata.get("language", "")).lower()
    if language and language not in policy.allowed_languages:
        return "language_not_allowed"
    turn_count = len(conversation.messages)
    if turn_count < policy.min_turns:
        return "too_few_turns"
    if turn_count > policy.max_turns:
        return "too_many_turns"
    if not _is_alternating(conversation.messages):
        return "role_order"
    return None


def _text_reject_reason(text: str, policy: CleanPolicy, *, lower: str | None = None) -> str | None:
    if lower is None:
        lower = text.lower()
    if ("@" in text and _EMAIL_RE.search(text)) or (
        _has_marker(lower, _SECRET_MARKERS) and _SECRET_RE.search(text)
    ):
        return "pii_or_secret"
    if _has_marker(lower, _FINANCIAL_MARKET_MARKERS) and _FINANCIAL_MARKET_RE.search(text):
        return "financial_market_advice"
    if _has_marker(lower, _BAD_CODE_MARKERS) and _BAD_CODE_ARTIFACT_RE.search(text):
        return "bad_code_artifact"
    if _MOJIBAKE_RE.search(text):
        return "mojibake"
    if _has_marker(lower, _CHILD_PERSONA_MARKERS) and _CHILD_PERSONA_RE.search(text):
        return "child_persona_roleplay"
    if (
        _has_marker(lower, _UNSAFE_SECURITY_SENSITIVE_MARKERS)
        and _has_marker(lower, _UNSAFE_SECURITY_ACTION_MARKERS)
        and _UNSAFE_SECURITY_RE.search(text)
    ):
        return "unsafe_security"
    text_len = len(text)
    if text_len < policy.min_chars:
        return "too_short"
    if text_len > policy.max_chars:
        return "too_long"
    if "http" in lower and _URL_ONLY_RE.match(text):
        return "url_only"
    if _repetition_ratio(text) > 0.35:
        return "repetition"
    if _bad_unicode_ratio(text) > 0.01:
        return "bad_unicode"
    return None


def _clean_messages_for_record(messages: list[Message]) -> list[Message]:
    cleaned: list[Message] = []
    changed = False
    for message in messages:
        content = _clean_text(message["content"])
        cleaned.append({"role": message["role"], "content": content})
        if content != message["content"]:
            changed = True
    return cleaned if changed else messages


def _has_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _has_code_hint(text: str, lower: str) -> bool:
    return _has_marker(lower, _CODE_HINT_MARKERS) and _CODE_HINT_RE.search(text) is not None


def _has_debug_failure_hint(lower: str) -> bool:
    return any(marker in lower for marker in ("fail", "error", "traceback", "pytest", "exception"))


def _normalize_messages(raw_messages: Iterable[dict]) -> list[Message]:
    messages: list[Message] = []
    for raw in raw_messages:
        role = _normalize_role(raw.get("role") or raw.get("from"))
        if role not in {"user", "assistant"}:
            continue
        content = _clean_text(_string(raw.get("content") or raw.get("value") or raw.get("text")))
        if not content:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = f"{messages[-1]['content']}\n{content}"
        else:
            messages.append({"role": role, "content": content})
    return messages


def _normalize_role(role: object) -> str:
    value = _string(role).strip().lower()
    if value in {"user", "human", "prompter"}:
        return "user"
    if value in {"assistant", "gpt", "chatbot", "bot"}:
        return "assistant"
    return value


def _is_rejected_openassistant_row(row: dict) -> bool:
    if row.get("deleted") is True:
        return True
    if row.get("synthetic") is True:
        return True
    review_result = row.get("review_result")
    if review_result is False:
        return True
    if _has_bad_openassistant_labels(row):
        return True
    lang = _string(row.get("lang") or row.get("language") or "").lower()
    return bool(lang and lang not in {"en", "eng", "english", "zh", "zh-cn", "chinese"})


def _openassistant_tree_id(row: dict) -> str:
    return _string(row.get("message_tree_id") or row.get("tree_id") or row.get("message_id") or row.get("id"))


def _path_to_root(leaf: dict, nodes: dict[str, dict]) -> list[dict]:
    path = [leaf]
    parent_id = leaf.get("parent_id")
    seen = {leaf["id"]}
    while parent_id is not None and parent_id in nodes:
        parent = nodes[parent_id]
        if parent["id"] in seen:
            break
        seen.add(parent["id"])
        path.append(parent)
        parent_id = parent.get("parent_id")
    path.reverse()
    return path


def _is_alternating(messages: list[Message]) -> bool:
    if not messages or messages[0]["role"] != "user":
        return False
    for previous, current in zip(messages, messages[1:]):
        if previous["role"] == current["role"]:
            return False
    return messages[-1]["role"] == "assistant"


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line.strip()).strip()


def _normalize_for_hash(text: str) -> str:
    text = _clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _fingerprint_digest(conversation: ConversationRecord, *, text: str | None = None) -> bytes:
    if text is None:
        text = conversation.normalized_text()
    normalized = _normalize_for_hash(text)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()


def _messages_text(messages: list[Message]) -> str:
    return "\n".join(message["content"] for message in messages)


def language_group(conversation: ConversationRecord, *, text: str | None = None) -> str:
    language = _string(conversation.metadata.get("language", "")).strip().lower()
    if language in {
        "zh",
        "zh-cn",
        "zh-hans",
        "zh-hant",
        "zho",
        "cmn",
        "chinese",
        "simplified chinese",
        "traditional chinese",
        "中文",
        "cn",
    }:
        return "zh"
    if language in {"en", "eng", "english"}:
        return "en"
    if text is None:
        text = conversation.normalized_text()
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_letters = sum(1 for char in text if ("a" <= char.lower() <= "z"))
    if cjk >= 8 and cjk >= ascii_letters * 0.2:
        return "zh"
    return "en"


def _record_language(record: dict) -> str:
    language = _string(record.get("language") or record.get("metadata", {}).get("language", "")).strip().lower()
    if language in {
        "zh",
        "zh-cn",
        "zh-hans",
        "zh-hant",
        "zho",
        "cmn",
        "chinese",
        "simplified chinese",
        "traditional chinese",
        "中文",
        "cn",
    }:
        return "zh"
    if language in {"en", "eng", "english"}:
        return "en"
    text = _string(record.get("text", ""))
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_letters = sum(1 for char in text if ("a" <= char.lower() <= "z"))
    if cjk >= 8 and cjk >= ascii_letters * 0.2:
        return "zh"
    return "en"


def _thread_key(conversation: ConversationRecord) -> str | None:
    tree_id = conversation.metadata.get("message_tree_id")
    if tree_id in {"", None}:
        return None
    return f"{conversation.source}:{tree_id}"


def _repetition_ratio(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 20:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count for token, count in counts.items() if count > 4 and len(token) > 2)
    return repeated / max(1, len(tokens))


def _bad_unicode_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    for char in text:
        codepoint = ord(char)
        if (codepoint < 32 or 127 <= codepoint <= 159) and char not in "\n\t":
            bad += 1
    return bad / len(text)


def _generic_reply_ratio(messages: list[Message]) -> float:
    generic = 0
    assistant = 0
    for message in messages:
        if message["role"] != "assistant":
            continue
        assistant += 1
        content = message["content"].strip().lower()
        if content in {"sure", "yes", "ok", "i can help with that"} or len(content) < 25:
            generic += 1
    return generic / max(1, assistant)


def _has_correction_pattern(text: str) -> bool:
    return bool(
        re.search(
            r"\b(actually|correction|fix(?:ed|ing)?\s+(?:bug|issue|error|test)|"
            r"regression test|root cause|I was wrong)\b",
            text,
            re.I,
        )
    )


def _has_bad_openassistant_labels(row: dict) -> bool:
    labels = row.get("labels")
    if not isinstance(labels, dict):
        return False
    names = labels.get("name") or []
    values = labels.get("value") or []
    label_map: dict[str, float] = {}
    for name, value in zip(names, values):
        try:
            label_map[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    quality = label_map.get("quality")
    if quality is not None and quality < 0.35:
        return True
    for bad_label in ("spam", "lang_mismatch", "pii", "not_appropriate", "hate_speech", "sexual_content"):
        if label_map.get(bad_label, 0.0) >= 0.35:
            return True
    if label_map.get("toxicity", 0.0) >= 0.55:
        return True
    return False


def _rank_value(value: object) -> int:
    try:
        if value is None:
            return 999
        return int(value)
    except (TypeError, ValueError):
            return 999


def _parse_hh_rlhf_transcript(transcript: str) -> list[Message]:
    parts = re.split(r"(?m)^\s*(Human|Assistant):\s*", transcript)
    messages: list[Message] = []
    for idx in range(1, len(parts), 2):
        role = _normalize_role(parts[idx])
        content = _clean_text(parts[idx + 1] if idx + 1 < len(parts) else "")
        if role in {"user", "assistant"} and content:
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] = f"{messages[-1]['content']}\n{content}"
            else:
                messages.append({"role": role, "content": content})
    if not _is_alternating(messages):
        return []
    return messages


def _first_nonempty(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _string(row.get(key)).strip()
        if value:
            return value
    return ""


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _string(value: object) -> str:
    if value is None:
        return ""
    return str(value)
