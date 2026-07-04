from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.data.api_distill import (
    DistillPrompt,
    OpenAICompatibleDistillClient,
    build_distilled_conversation,
    load_env_file,
    load_teacher_providers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate distilled dialogue data through approved APIs.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "distill_providers.json")
    parser.add_argument("--provider", required=True, help="Teacher provider name from the config file.")
    parser.add_argument("--model", default=None, help="Teacher model name. Required unless provider has default_model.")
    parser.add_argument("--license-id", default=None, help="Teacher model output license, e.g. apache-2.0 or mit.")
    parser.add_argument("--seed-file", type=Path, default=None, help="JSONL file with prompt_id/lang/category/user.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "distill_api")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env", help="Local ignored KEY=value file.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum prompts to generate. 0 means all.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate config without calling the API.")
    parser.add_argument("--check-key", action="store_true", help="Dry-run also checks whether key env exists.")
    return parser.parse_args()


def iter_prompts(path: Path) -> Iterator[DistillPrompt]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                yield DistillPrompt(
                    prompt_id=str(raw["prompt_id"]),
                    lang=str(raw["lang"]),
                    category=str(raw.get("category") or "daily_dialogue"),
                    user=str(raw["user"]),
                    system=_optional_str(raw.get("system")),
                    metadata=dict(raw.get("metadata") or {}),
                )
            except KeyError as exc:
                raise RuntimeError(f"Invalid seed prompt at line {line_number}: missing {exc.args[0]}") from exc


def completed_prompt_ids(raw_path: Path, train_path: Path) -> set[str]:
    raw_ids = _prompt_ids_from_jsonl(raw_path, metadata=False)
    train_ids = _prompt_ids_from_jsonl(train_path, metadata=True)
    return raw_ids & train_ids


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    providers = load_teacher_providers(args.config)
    if args.provider not in providers:
        known = ", ".join(sorted(providers))
        raise RuntimeError(f"Unknown provider {args.provider!r}. Known providers: {known}")

    provider = providers[args.provider]
    model = args.model or provider.default_model
    license_id = args.license_id or provider.license_id

    if args.dry_run:
        print(f"provider={provider.name}")
        print(f"api_type={provider.api_type}")
        print(f"url={provider.chat_completions_url}")
        print(f"key_env={provider.api_key_env}")
        print(f"key_present={bool(os.environ.get(provider.api_key_env, '').strip())}")
        print(f"model={model or 'pending'}")
        print(f"license={license_id}")
        print(f"terms_checked={provider.terms_checked}")
        if args.check_key:
            provider.resolve_api_key()
        return 0

    if not model:
        raise RuntimeError("Teacher model is required. Pass --model after choosing the model to distill.")
    if not license_id or license_id == "set-per-model":
        raise RuntimeError("Teacher model license is required. Pass --license-id after checking the model license.")
    if args.seed_file is None:
        raise RuntimeError("--seed-file is required for non-dry-run generation.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_generations.jsonl"
    train_path = args.out_dir / "distilled_train.jsonl"
    completed_ids = completed_prompt_ids(raw_path, train_path)
    client = OpenAICompatibleDistillClient(provider)
    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    generated = 0
    skipped = 0
    for index, prompt in enumerate(iter_prompts(args.seed_file), start=1):
        if args.limit > 0 and index > args.limit:
            break
        if prompt.prompt_id in completed_ids:
            skipped += 1
            continue
        assistant = client.generate(
            model=model,
            messages=prompt.to_messages(),
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        conversation = build_distilled_conversation(
            prompt=prompt,
            assistant_text=assistant,
            provider_name=provider.name,
            teacher_model=model,
            license_id=license_id,
            generation_config=generation_config,
        )
        append_jsonl(
            raw_path,
            {
                "prompt_id": prompt.prompt_id,
                "provider": provider.name,
                "teacher_model": model,
                "license": license_id,
                "lang": prompt.lang,
                "category": prompt.category,
                "user": prompt.user,
                "assistant": assistant,
                "generation_config": generation_config,
            },
        )
        append_jsonl(train_path, conversation.to_training_record())
        generated += 1
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print(f"generated={generated} skipped={skipped} out={args.out_dir}")
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


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
