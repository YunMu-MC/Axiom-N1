from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from dopa_coder_n1.data.dialogue_cleaner import ConversationRecord


Message = dict[str, str]


@dataclass(frozen=True)
class TeacherProvider:
    name: str
    base_url: str
    api_key_env: str
    api_type: str = "openai_compatible"
    default_model: str | None = None
    license_id: str = "set-per-model"
    terms_checked: bool = False
    notes: str = ""

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    def resolve_api_key(self, env: Mapping[str, str] | None = None) -> str:
        values = os.environ if env is None else env
        key = values.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"Missing API key environment variable: {self.api_key_env}. "
                "Do not write provider keys into config files."
            )
        return key


@dataclass(frozen=True)
class DistillPrompt:
    prompt_id: str
    lang: str
    category: str
    user: str
    system: str | None = None
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def to_messages(self) -> list[Message]:
        messages: list[Message] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.user})
        return messages


def load_teacher_providers(path: str | Path) -> dict[str, TeacherProvider]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    providers: dict[str, TeacherProvider] = {}
    for raw in config.get("providers", []):
        provider = TeacherProvider(
            name=_required_str(raw, "name"),
            api_type=str(raw.get("api_type") or "openai_compatible"),
            base_url=_required_str(raw, "base_url"),
            api_key_env=_required_str(raw, "api_key_env"),
            default_model=_optional_str(raw.get("default_model")),
            license_id=str(raw.get("license_id") or "set-per-model"),
            terms_checked=bool(raw.get("terms_checked", False)),
            notes=str(raw.get("notes") or ""),
        )
        if provider.api_type != "openai_compatible":
            raise RuntimeError(f"Unsupported teacher provider api_type: {provider.api_type}")
        if provider.name in providers:
            raise RuntimeError(f"Duplicate teacher provider: {provider.name}")
        providers[provider.name] = provider
    if not providers:
        raise RuntimeError(f"No teacher providers configured in {path}")
    return providers


def load_env_file(path: str | Path) -> set[str]:
    env_path = Path(path)
    if not env_path.exists():
        return set()
    loaded: set[str] = set()
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"Invalid env file line {line_number}: expected KEY=value")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise RuntimeError(f"Invalid env file line {line_number}: empty key")
        if key in os.environ:
            continue
        os.environ[key] = _strip_env_value(value.strip())
        loaded.add(key)
    return loaded


def build_openai_chat_payload(
    *,
    model: str,
    messages: Sequence[Message],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 768,
) -> dict:
    if not model.strip():
        raise RuntimeError("Teacher model is required.")
    if not messages:
        raise RuntimeError("At least one chat message is required.")
    return {
        "model": model,
        "messages": [{"role": item["role"], "content": item["content"]} for item in messages],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }


def parse_openai_chat_content(body: Mapping) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenAI-compatible response missing assistant content.") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenAI-compatible response missing assistant content.")
    return content.strip()


class OpenAICompatibleDistillClient:
    def __init__(self, provider: TeacherProvider, *, timeout_seconds: float = 120.0) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 768,
    ) -> str:
        payload = build_openai_chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        request = urllib.request.Request(
            self.provider.chat_completions_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.provider.resolve_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Teacher API request failed: HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Teacher API request failed: {exc.reason}") from exc
        return parse_openai_chat_content(body)


def build_distilled_conversation(
    *,
    prompt: DistillPrompt,
    assistant_text: str,
    provider_name: str,
    teacher_model: str,
    license_id: str,
    generation_config: Mapping[str, str | int | float | bool | None],
) -> ConversationRecord:
    assistant = assistant_text.strip()
    if not assistant:
        raise RuntimeError("Teacher assistant output is empty.")
    messages = [{"role": "user", "content": prompt.user}, {"role": "assistant", "content": assistant}]
    metadata = {
        "language": prompt.lang,
        "prompt_id": prompt.prompt_id,
        "provider": provider_name,
        "teacher_model": teacher_model,
        "distill_source": "api",
        "temperature": generation_config.get("temperature"),
        "top_p": generation_config.get("top_p"),
        "max_tokens": generation_config.get("max_tokens"),
        **prompt.metadata,
    }
    return ConversationRecord(
        source=f"distill:{provider_name}:{teacher_model}",
        license_id=license_id.lower(),
        source_id=prompt.prompt_id,
        messages=messages,
        metadata={key: value for key, value in metadata.items() if value not in {"", None}},
        category=prompt.category,
        quality_score=0.0,
        quality_tags=("distilled", "needs_cleaning"),
    )


def _required_str(raw: Mapping, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Missing required provider field: {key}")
    return value.strip()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
