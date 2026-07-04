import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dopa_coder_n1.data.api_distill import (
    DistillPrompt,
    build_distilled_conversation,
    build_openai_chat_payload,
    load_env_file,
    load_teacher_providers,
    parse_openai_chat_content,
)


def test_teacher_provider_config_uses_env_var_without_storing_secret(tmp_path):
    config_path = tmp_path / "providers.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": "openai_compatible_teacher",
                        "api_type": "openai_compatible",
                        "base_url": "https://api.example.invalid",
                        "api_key_env": "DOPA_DISTILL_API_KEY",
                        "default_model": None,
                        "license_id": "set-per-model",
                        "terms_checked": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    providers = load_teacher_providers(config_path)
    provider = providers["openai_compatible_teacher"]

    assert provider.chat_completions_url == "https://api.example.invalid/v1/chat/completions"
    assert provider.resolve_api_key({"DOPA_DISTILL_API_KEY": "secret-value"}) == "secret-value"
    assert "secret-value" not in repr(provider)
    with pytest.raises(RuntimeError, match="DOPA_DISTILL_API_KEY"):
        provider.resolve_api_key({})


def test_load_env_file_reads_local_secret_without_overwriting_existing_value(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# local only",
                "DOPA_DISTILL_API_KEY=local-secret",
                'QUOTED_VALUE="quoted-secret"',
                "EXISTING_VALUE=file-secret",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING_VALUE", "already-set")

    loaded = load_env_file(env_path)

    assert loaded == {"DOPA_DISTILL_API_KEY", "QUOTED_VALUE"}
    assert os.environ["DOPA_DISTILL_API_KEY"] == "local-secret"
    assert os.environ["QUOTED_VALUE"] == "quoted-secret"
    assert os.environ["EXISTING_VALUE"] == "already-set"


def test_openai_compatible_payload_preserves_daily_dialogue_messages():
    payload = build_openai_chat_payload(
        model="qwen-test",
        messages=[
            {"role": "system", "content": "回答自然、简洁。"},
            {"role": "user", "content": "今天有点烦，怎么缓一下？"},
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    assert payload["model"] == "qwen-test"
    assert payload["messages"][1]["content"] == "今天有点烦，怎么缓一下？"
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 512


def test_parse_openai_chat_content_requires_assistant_content():
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "可以先停一下，把最急的事拆成一个小动作。",
                }
            }
        ]
    }

    assert parse_openai_chat_content(body).startswith("可以先停一下")
    with pytest.raises(RuntimeError, match="missing assistant content"):
        parse_openai_chat_content({"choices": [{"message": {"role": "assistant"}}]})


def test_distilled_conversation_records_teacher_metadata_without_key():
    prompt = DistillPrompt(
        prompt_id="zh_daily_0001",
        lang="zh",
        category="daily_dialogue",
        user="今天有点烦，怎么缓一下？",
    )

    conversation = build_distilled_conversation(
        prompt=prompt,
        assistant_text="可以先停一下，把最急的事拆成一个小动作。",
        provider_name="openai_compatible_teacher",
        teacher_model="qwen-test",
        license_id="apache-2.0",
        generation_config={"temperature": 0.7, "max_tokens": 512},
    )
    record = conversation.to_training_record()

    assert record["source"] == "distill:openai_compatible_teacher:qwen-test"
    assert record["language"] == "zh"
    assert record["license"] == "apache-2.0"
    assert record["metadata"]["teacher_model"] == "qwen-test"
    assert "key" not in json.dumps(record, ensure_ascii=False).lower()


def test_distill_dialogue_cli_dry_run_validates_provider_without_key(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "providers.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": "openai_compatible_teacher",
                        "api_type": "openai_compatible",
                        "base_url": "https://api.example.invalid",
                        "api_key_env": "DOPA_DISTILL_API_KEY",
                        "default_model": None,
                        "license_id": "set-per-model",
                        "terms_checked": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "distill_dialogue.py"),
            "--config",
            str(config_path),
            "--provider",
            "openai_compatible_teacher",
            "--dry-run",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "provider=openai_compatible_teacher" in result.stdout
    assert "key_env=DOPA_DISTILL_API_KEY" in result.stdout
    assert "secret" not in result.stdout.lower()


def test_capability_distill_seed_excludes_daily_chat_and_covers_core_skills():
    project_root = Path(__file__).resolve().parents[1]
    seed_path = project_root / "configs" / "distill_seed_capability.jsonl"

    rows = [json.loads(line) for line in seed_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    categories = {row["category"] for row in rows}
    languages = {row["lang"] for row in rows}

    assert "daily_dialogue" not in categories
    assert {"zh", "en"} <= languages
    assert {
        "coding",
        "debug",
        "tool_calling",
        "security_defensive",
        "anti_hallucination",
        "reasoning_self_check",
    } <= categories
    assert all("不要展示隐藏推理" in row["system"] or "Do not reveal hidden reasoning" in row["system"] for row in rows)


def test_language_distill_seed_excludes_code_tool_security_and_covers_language_skills():
    project_root = Path(__file__).resolve().parents[1]
    seed_path = project_root / "configs" / "distill_seed_language.jsonl"

    rows = [json.loads(line) for line in seed_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    categories = {row["category"] for row in rows}
    languages = {row["lang"] for row in rows}

    assert {"zh", "en"} <= languages
    assert not ({"coding", "debug", "tool_calling", "security_defensive"} & categories)
    assert {
        "language_understanding",
        "rewrite",
        "summarization",
        "translation",
        "tone_control",
        "ambiguity_clarification",
    } <= categories
    assert all("代码" not in row["user"] and "code" not in row["user"].lower() for row in rows)


def test_distill_resume_only_skips_prompt_ids_present_in_raw_and_train(tmp_path):
    from scripts.distill_dialogue import completed_prompt_ids

    raw_path = tmp_path / "raw_generations.jsonl"
    train_path = tmp_path / "distilled_train.jsonl"
    raw_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt_id": "done"}),
                json.dumps({"prompt_id": "raw_only"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_path.write_text(
        "\n".join(
            [
                json.dumps({"metadata": {"prompt_id": "done"}}),
                json.dumps({"metadata": {"prompt_id": "train_only"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert completed_prompt_ids(raw_path, train_path) == {"done"}


def test_language_seed_batch_generator_produces_balanced_language_only_prompts(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "language_batch.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "generate_language_distill_seeds.py"),
            "--out",
            str(out_path),
            "--count",
            "24",
            "--seed",
            "7",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    categories = {row["category"] for row in rows}
    languages = [row["lang"] for row in rows]
    text = "\n".join(row["user"] for row in rows).lower()

    assert len(rows) == 24
    assert languages.count("zh") == 12
    assert languages.count("en") == 12
    assert not ({"coding", "debug", "tool_calling", "security_defensive"} & categories)
    assert {"language_understanding", "rewrite", "summarization", "translation", "tone_control", "ambiguity_clarification"} <= categories
    assert "code" not in text
    assert "代码" not in text
