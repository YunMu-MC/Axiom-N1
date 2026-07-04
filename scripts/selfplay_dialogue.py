from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.data.api_distill import (
    OpenAICompatibleDistillClient,
    load_env_file,
    load_teacher_providers,
)
from dopa_coder_n1.data.dialogue_cleaner import ConversationRecord, _MOJIBAKE_RE


CATEGORIES = [
    "language_understanding",
    "rewrite",
    "summarization",
    "translation",
    "tone_control",
    "ambiguity_clarification",
]

ZH_TOPICS = [
    "项目沟通",
    "学习计划",
    "会议结论",
    "客户反馈",
    "服务说明",
    "团队协作",
    "产品上线",
    "问题复盘",
]

EN_TOPICS = [
    "project communication",
    "study planning",
    "meeting decisions",
    "customer feedback",
    "service wording",
    "team collaboration",
    "product release",
    "incident review",
]


MOJIBAKE_MARKERS = (
    "�",
    "Ã",
    "â€",
    "鈥",
    "鎴",
    "鐭",
    "涔",
    "轰",
    "粈",
    "繖",
    "娆",
    "椤圭",
    "洰",
    "寤",
    "舵",
    "湡",
    "鍥",
    "炵",
    "瓟",
    "锛",
    "銆",
    "俓",
)


@dataclass(frozen=True)
class SelfPlayTask:
    prompt_id: str
    lang: str
    category: str
    topic: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate self-play dialogue data with one API call asking and one answering.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "distill_providers.json")
    parser.add_argument("--ask-provider", default="openai_compatible_teacher")
    parser.add_argument("--answer-provider", default="openai_compatible_teacher")
    parser.add_argument("--ask-model", default=None)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--license-id", default=None)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "distill_selfplay_qwen")
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--ask-temperature", type=float, default=0.85)
    parser.add_argument("--answer-temperature", type=float, default=0.65)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--ask-max-tokens", type=int, default=192)
    parser.add_argument("--answer-max-tokens", type=int, default=768)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-key", action="store_true")
    return parser.parse_args()


def iter_selfplay_tasks(*, count: int, start_index: int = 0, seed: int = 20260704) -> Iterator[SelfPlayTask]:
    if count <= 0:
        raise RuntimeError("--count must be positive")
    rng = random.Random(seed)
    for offset in range(count):
        index = start_index + offset
        lang = "zh" if index % 2 == 0 else "en"
        category = CATEGORIES[(index // 2) % len(CATEGORIES)]
        topic_pool = ZH_TOPICS if lang == "zh" else EN_TOPICS
        topic = rng.choice(topic_pool)
        yield SelfPlayTask(
            prompt_id=f"selfplay_{lang}_{index:06d}",
            lang=lang,
            category=category,
            topic=topic,
        )


def build_user_generation_messages(task: SelfPlayTask) -> list[dict[str, str]]:
    if task.lang == "zh":
        system = (
            "你是高质量对话数据的用户侧生成器。只输出用户消息，不要回答，不要解释，"
            "不要加“用户：”标签。消息要真实、自然、具体，适合训练中文语言理解和表达能力。"
        )
        user = (
            f"生成一条中文用户消息。主题：{task.topic}。任务类型：{task.category}。"
            "要求：一句到三句；不要涉及代码、工具调用、色情、违法、攻击、个人隐私或真实账号密钥；"
            "要像真实用户在请求帮助、改写、解释、总结或澄清。"
        )
    else:
        system = (
            "You generate the user side of high-quality dialogue data. Output only the user message. "
            "Do not answer it, do not explain, and do not add a 'User:' label. Make it realistic, natural, and specific."
        )
        user = (
            f"Generate one English user message. Topic: {task.topic}. Task type: {task.category}. "
            "Use one to three sentences. Avoid code, tool calls, sexual content, illegal requests, attacks, "
            "personal data, real accounts, or secrets. It should ask for help with wording, explanation, "
            "summarization, translation, tone, or clarification."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_answer_messages(task: SelfPlayTask, user_message: str) -> list[dict[str, str]]:
    if task.lang == "zh":
        system = (
            "你是语言能力很强的助手。回答要自然、准确、清楚，直接解决用户的文字表达问题。"
            "不要展示隐藏推理，不要编造外部事实，不要写成过度俏皮的闲聊。"
        )
    else:
        system = (
            "You are a strong language assistant. Answer naturally, accurately, and clearly, directly solving "
            "the user's wording or comprehension problem. Do not reveal hidden reasoning or invent external facts."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user_message}]


def contains_mojibake(text: str) -> bool:
    if _MOJIBAKE_RE.search(text):
        return True
    marker_hits = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    return marker_hits >= 2


def clean_generated_user_message(text: str) -> str:
    cleaned = _strip_generated_message(text)
    cleaned = re.sub(r"^\s*(user|用户|question|问题)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    if re.match(r"^\s*(assistant|助手|answer|回答)\s*[:：]", cleaned, flags=re.IGNORECASE):
        return ""
    return _accept_generated_message(_strip_generated_message(cleaned), max_chars=1200)


def clean_generated_assistant_message(text: str) -> str:
    cleaned = _strip_generated_message(text)
    cleaned = re.sub(r"^\s*(assistant|助手|answer|回答)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    if re.match(r"^\s*(user|用户|question|问题)\s*[:：]", cleaned, flags=re.IGNORECASE):
        return ""
    return _accept_generated_message(_strip_generated_message(cleaned), max_chars=6000)


def _strip_generated_message(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.strip().strip('"“”')
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _accept_generated_message(text: str, *, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) > max_chars:
        return ""
    if contains_mojibake(text):
        return ""
    lowered = text.lower()
    banned = ("api key", "password", "sk-", "-----begin", "bomb", "malware")
    if any(item in lowered for item in banned):
        return ""
    return text


def completed_prompt_ids(raw_path: Path, train_path: Path) -> set[str]:
    return _prompt_ids_from_jsonl(raw_path, metadata=False) & _prompt_ids_from_jsonl(train_path, metadata=True)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def build_training_record(
    *,
    task: SelfPlayTask,
    user_message: str,
    assistant_message: str,
    ask_provider: str,
    ask_model: str,
    answer_provider: str,
    answer_model: str,
    license_id: str,
    generation_config: dict,
) -> dict:
    conversation = ConversationRecord(
        source=f"selfplay:{ask_provider}:{ask_model}->{answer_provider}:{answer_model}",
        license_id=license_id.lower(),
        source_id=task.prompt_id,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message.strip()},
        ],
        metadata={
            "language": task.lang,
            "prompt_id": task.prompt_id,
            "category": task.category,
            "topic": task.topic,
            "ask_provider": ask_provider,
            "ask_model": ask_model,
            "answer_provider": answer_provider,
            "answer_model": answer_model,
            "distill_source": "api_selfplay",
            **generation_config,
        },
        category=task.category,
        quality_score=0.0,
        quality_tags=("selfplay", "api", "needs_cleaning"),
    )
    return conversation.to_training_record()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    providers = load_teacher_providers(args.config)
    ask_provider = providers[args.ask_provider]
    answer_provider = providers[args.answer_provider]
    ask_model = args.ask_model or ask_provider.default_model
    answer_model = args.answer_model or answer_provider.default_model
    license_id = args.license_id or answer_provider.license_id
    if not ask_model or not answer_model:
        raise RuntimeError("Both ask and answer models are required.")
    if not license_id or license_id == "set-per-model":
        raise RuntimeError("A concrete --license-id is required.")

    if args.dry_run:
        print(f"ask_provider={ask_provider.name}")
        print(f"answer_provider={answer_provider.name}")
        print(f"ask_model={ask_model}")
        print(f"answer_model={answer_model}")
        print(f"license={license_id}")
        print(f"out_dir={args.out_dir}")
        print(f"count={args.count}")
        if args.check_key:
            ask_provider.resolve_api_key()
            answer_provider.resolve_api_key()
            print("key_present=True")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_selfplay.jsonl"
    train_path = args.out_dir / "distilled_train.jsonl"
    completed = completed_prompt_ids(raw_path, train_path)
    ask_client = OpenAICompatibleDistillClient(ask_provider)
    answer_client = OpenAICompatibleDistillClient(answer_provider)
    generation_config = {
        "ask_temperature": args.ask_temperature,
        "answer_temperature": args.answer_temperature,
        "top_p": args.top_p,
        "ask_max_tokens": args.ask_max_tokens,
        "answer_max_tokens": args.answer_max_tokens,
    }

    generated = 0
    skipped = 0
    rejected = 0
    for task in iter_selfplay_tasks(count=args.count, start_index=args.start_index, seed=args.seed):
        if task.prompt_id in completed:
            skipped += 1
            continue
        user_message = ""
        for _ in range(3):
            raw_user = ask_client.generate(
                model=ask_model,
                messages=build_user_generation_messages(task),
                temperature=args.ask_temperature,
                top_p=args.top_p,
                max_tokens=args.ask_max_tokens,
            )
            user_message = clean_generated_user_message(raw_user)
            if user_message:
                break
        if not user_message:
            rejected += 1
            continue
        raw_assistant = answer_client.generate(
            model=answer_model,
            messages=build_answer_messages(task, user_message),
            temperature=args.answer_temperature,
            top_p=args.top_p,
            max_tokens=args.answer_max_tokens,
        )
        assistant_message = clean_generated_assistant_message(raw_assistant)
        if not assistant_message:
            rejected += 1
            continue
        append_jsonl(
            raw_path,
            {
                "prompt_id": task.prompt_id,
                "lang": task.lang,
                "category": task.category,
                "topic": task.topic,
                "ask_provider": ask_provider.name,
                "ask_model": ask_model,
                "answer_provider": answer_provider.name,
                "answer_model": answer_model,
                "license": license_id,
                "user": user_message,
                "assistant": assistant_message,
                "generation_config": generation_config,
            },
        )
        append_jsonl(
            train_path,
            build_training_record(
                task=task,
                user_message=user_message,
                assistant_message=assistant_message,
                ask_provider=ask_provider.name,
                ask_model=ask_model,
                answer_provider=answer_provider.name,
                answer_model=answer_model,
                license_id=license_id,
                generation_config=generation_config,
            ),
        )
        generated += 1
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    print(f"generated={generated} skipped={skipped} rejected={rejected} out={args.out_dir}")
    return 0


def _prompt_ids_from_jsonl(path: Path, *, metadata: bool) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt_id = record.get("metadata", {}).get("prompt_id") if metadata else record.get("prompt_id")
        if isinstance(prompt_id, str) and prompt_id:
            ids.add(prompt_id)
    return ids


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
