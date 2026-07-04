from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess

import pytest

from scripts.clean_dialogue_corpus import (
    RustCleaner,
    RustPrefilter,
    apply_prefilter_verdicts,
    build_clean_policy,
    build_cleaner_rows,
    build_prefilter_rows,
    convert_rows,
    iter_source_conversations,
    load_stream_resume_state,
    next_stream_shard_index,
    source_catalog_payload,
    default_source_keys,
    run_rust_prefilter_batch,
    should_emit_progress,
    validate_selected_sources,
)
from dopa_coder_n1.data.dialogue_cleaner import CleanPolicy, ConversationRecord, StreamingDialogueCleaner


def test_default_sources_include_new_unrestricted_sources_and_exclude_restricted_sources():
    defaults = set(default_source_keys())

    assert {"oasst1", "oasst2", "wildchat", "aya_dataset", "helpsteer3_preference"} <= defaults
    assert "lmsys_chat_1m" not in defaults
    assert "ultrachat_200k" not in defaults
    assert "databricks_dolly_15k" not in defaults


def test_restricted_sources_require_explicit_flag():
    with pytest.raises(RuntimeError, match="restricted"):
        validate_selected_sources(["lmsys_chat_1m"], allow_restricted_license=False)

    specs = validate_selected_sources(["lmsys_chat_1m"], allow_restricted_license=True)

    assert specs[0].name == "lmsys/lmsys-chat-1m"


def test_restricted_license_is_only_added_to_policy_after_explicit_flag():
    specs = validate_selected_sources(["lmsys_chat_1m"], allow_restricted_license=True)

    policy = build_clean_policy(
        min_score=0.45,
        target_categories=None,
        selected_specs=specs,
        allow_restricted_license=True,
    )

    assert "lmsys-chat-1m-agreement" in policy.allowed_licenses


def test_convert_rows_supports_added_prompt_target_and_preference_sources():
    aya = validate_selected_sources(["aya_dataset"], allow_restricted_license=False)[0]
    helpsteer = validate_selected_sources(["helpsteer3_preference"], allow_restricted_license=False)[0]

    aya_conversations = convert_rows(
        aya,
        [
            {
                "inputs": "怎么调试 pytest 报错?",
                "targets": "先看堆栈, 再缩小复现, 最后补回归测试。",
                "language_code": "zho",
            }
        ],
    )
    helpsteer_conversations = convert_rows(
        helpsteer,
        [
            {
                "domain": "code",
                "language": "english",
                "context": [{"role": "user", "content": "How do I debug pytest?"}],
                "response1": "Guess randomly.",
                "response2": "Read the assertion and isolate the failing fixture.",
                "overall_preference": 1,
            }
        ],
    )

    assert aya_conversations[0].messages[-1]["role"] == "assistant"
    assert helpsteer_conversations[0].messages[-1]["content"].startswith("Read the assertion")


def test_convert_rows_supports_dolly_instruction_context_source():
    spec = validate_selected_sources(["databricks_dolly_15k"], allow_restricted_license=False)[0]

    conversations = convert_rows(
        spec,
        [
            {
                "instruction": "Explain why the deployment note is ambiguous.",
                "context": "The note says the service will restart later.",
                "response": "The word later is vague; name the date, time, and expected downtime.",
                "category": "brainstorming",
            }
        ],
    )

    assert conversations[0].license_id == "cc-by-sa-3.0"
    assert "Context:" in conversations[0].messages[0]["content"]
    assert conversations[0].messages[-1]["role"] == "assistant"


def test_source_catalog_adds_stack_exchange_and_keeps_synthetic_opt_in():
    catalog = {item["key"]: item for item in source_catalog_payload()}
    defaults = set(default_source_keys())

    assert "stack_exchange_preferences" in defaults
    assert "pmp_stack_exchange" in defaults
    assert "openorca" not in defaults
    assert catalog["stack_exchange_preferences"]["license"] == "cc-by-sa-4.0"
    assert catalog["pmp_stack_exchange"]["restricted"] is False
    assert catalog["openorca"]["default_enabled"] is False
    assert "synthetic" in catalog["openorca"]["notes"].lower()


def test_cc_by_sa_4_stack_exchange_license_is_allowed_by_default_policy():
    specs = validate_selected_sources(["stack_exchange_preferences"], allow_restricted_license=False)

    policy = build_clean_policy(
        min_score=0.45,
        target_categories=None,
        selected_specs=specs,
        allow_restricted_license=False,
    )

    assert "cc-by-sa-4.0" in policy.allowed_licenses


def test_convert_rows_supports_stack_exchange_answer_sources():
    spec = validate_selected_sources(["stack_exchange_preferences"], allow_restricted_license=False)[0]

    conversations = convert_rows(
        spec,
        [
            {
                "qid": 123,
                "title": "How can I debug a flaky pytest fixture?",
                "question": "<p>The test fails with ValueError only on CI. How should I narrow it down?</p>",
                "answers": [
                    {"body": "Just rerun it until it passes.", "score": -2},
                    {
                        "body": "<p>Read the traceback, isolate the fixture state, reproduce the smallest failing test, "
                        "and add a regression test after fixing it.</p>",
                        "score": 12,
                        "is_accepted": True,
                    },
                ],
                "tags": ["python", "pytest"],
                "site": "stackoverflow",
            }
        ],
    )

    assert len(conversations) == 1
    conversation = conversations[0]
    assert conversation.license_id == "cc-by-sa-4.0"
    assert conversation.source_id == "123"
    assert "flaky pytest fixture" in conversation.messages[0]["content"]
    assert conversation.messages[-1]["content"].startswith("Read the traceback")
    assert conversation.metadata["language"] == "en"
    assert conversation.metadata["site"] == "stackoverflow"
    assert conversation.metadata["answer_score"] == 12


def test_convert_rows_supports_openorca_as_manual_synthetic_source():
    spec = validate_selected_sources(["openorca"], allow_restricted_license=False)[0]

    conversations = convert_rows(
        spec,
        [
            {
                "id": "orca-1",
                "system_prompt": "You are a careful assistant.",
                "question": "Why is vague deployment timing risky?",
                "response": "It hides the maintenance window; state the exact time, rollback plan, and user impact.",
            }
        ],
    )

    assert len(conversations) == 1
    assert conversations[0].source_id == "orca-1"
    assert "System:" in conversations[0].messages[0]["content"]
    assert conversations[0].metadata["synthetic"] is True
    assert conversations[0].metadata["language"] == "en"


def test_prefilter_rows_are_tsv_safe_and_verdicts_update_report():
    spec = validate_selected_sources(["aya_dataset"], allow_restricted_license=False)[0]
    conversation = convert_rows(
        spec,
        [
            {
                "inputs": "怎么调试 pytest 报错?\n需要具体步骤。",
                "targets": "先看堆栈, 再缩小复现, 最后补回归测试。",
                "language_code": "zho",
            }
        ],
    )[0]

    rows = build_prefilter_rows([conversation])
    accepted, rejected = apply_prefilter_verdicts(
        [conversation],
        {"0": ("reject", "pii_or_secret")},
    )

    assert len(rows) == 1
    assert rows[0].count("\t") == 4
    assert "\\n" in rows[0]
    assert accepted == []
    assert rejected["pii_or_secret"] == 1


def test_rust_prefilter_batch_uses_utf8_pipe_without_temp_output_file(monkeypatch, tmp_path):
    spec = validate_selected_sources(["aya_dataset"], allow_restricted_license=False)[0]
    conversation = convert_rows(
        spec,
        [
            {
                "inputs": "怎么调试 pytest 报错?",
                "targets": "先看堆栈, 再缩小复现, 最后补回归测试。",
                "language_code": "zho",
            }
        ],
    )[0]
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="0\taccept\taccepted\n", stderr="")

    monkeypatch.setattr("scripts.clean_dialogue_corpus.subprocess.run", fake_run)

    verdicts = run_rust_prefilter_batch(
        Path("fake-dopa-dialogue-filter.exe"),
        [conversation],
        policy=CleanPolicy(),
        cache_dir=tmp_path,
    )

    assert verdicts == {"0": ("accept", "accepted")}
    assert captured["cmd"][1] == "filter-stdin"
    assert "input.tsv" not in captured["cmd"]
    assert "output.tsv" not in captured["cmd"]
    assert "怎么调试 pytest 报错?" in captured["input"]
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_rust_cleaner_reuses_persistent_process_and_returns_jsonl(monkeypatch, tmp_path):
    conversation = ConversationRecord(
        source="unit/source",
        license_id="apache-2.0",
        source_id="rust-clean",
        messages=[
            {"role": "user", "content": "How do I debug pytest fixture failures?"},
            {
                "role": "assistant",
                "content": "Read the traceback, isolate fixture state, reproduce the smallest failing case, then add a regression test.",
            },
        ],
        metadata={"language": "en", "message_tree_id": "tree-1"},
    )
    popen_calls = []

    class FakeStdin:
        def __init__(self):
            self.writes = []
            self.closed = False

        def write(self, value):
            self.writes.append(value)

        def flush(self):
            return None

        def close(self):
            self.closed = True

    class FakeStdout:
        def __init__(self):
            self.lines = iter(
                [
                    '0\taccept\taccepted\ten\tdebug\t0.6000\tquestion_answer,code_or_debug\t225\tunit/source\t{"text":"User: How do I debug pytest fixture failures?","language":"en","source":"unit/source","source_id":"rust-clean","license":"apache-2.0","category":"debug","quality_score":0.6,"quality_tags":["question_answer","code_or_debug"],"metadata":{"language":"en","message_tree_id":"tree-1"}}\n',
                    "0\treject\tduplicate_thread\n",
                ]
            )

        def readline(self):
            return next(self.lines, "")

    class FakeProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = FakeStdout()
            self.stderr = None
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return self.returncode

        def kill(self):
            self.returncode = -9

    fake_process = FakeProcess()

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return fake_process

    monkeypatch.setattr("scripts.clean_dialogue_corpus.subprocess.Popen", fake_popen)
    cleaner = RustCleaner(
        mode="auto",
        binary=Path("fake-dopa-dialogue-filter.exe"),
        batch_size=4096,
        active=True,
    )

    rows = build_cleaner_rows([conversation])
    first = cleaner.clean_batch([conversation], policy=CleanPolicy(min_score=0.6), cache_dir=tmp_path)
    second = cleaner.clean_batch([conversation], policy=CleanPolicy(min_score=0.6), cache_dir=tmp_path)
    cleaner.close()

    assert len(rows) == 1
    assert rows[0].count("\t") == 8
    assert first.accepted[0].language == "en"
    assert first.accepted[0].category == "debug"
    assert first.accepted[0].source == "unit/source"
    assert first.accepted[0].json_line.endswith("\n")
    assert first.rejected == {}
    assert second.accepted == []
    assert second.rejected["duplicate_thread"] == 1
    assert len(popen_calls) == 1
    assert popen_calls[0][0][1] == "clean-batch-stdin"
    assert fake_process.stdin.writes.count("1\n") == 2
    assert fake_process.stdin.closed is True


def test_rust_prefilter_reuses_persistent_batch_process(monkeypatch, tmp_path):
    conversation = ConversationRecord(
        source="unit",
        license_id="apache-2.0",
        source_id="rust-session",
        messages=[
            {"role": "user", "content": "How do I debug pytest fixture failures?"},
            {"role": "assistant", "content": "Read the traceback and add a regression test."},
        ],
    )
    popen_calls = []

    class FakeStdin:
        def __init__(self):
            self.writes = []
            self.closed = False

        def write(self, value):
            self.writes.append(value)

        def flush(self):
            return None

        def close(self):
            self.closed = True

    class FakeStdout:
        def __init__(self):
            self.lines = iter(["0\taccept\taccepted\n", "0\taccept\taccepted\n"])

        def readline(self):
            return next(self.lines, "")

    class FakeProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = FakeStdout()
            self.stderr = None
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return self.returncode

        def kill(self):
            self.returncode = -9

    fake_process = FakeProcess()

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return fake_process

    monkeypatch.setattr("scripts.clean_dialogue_corpus.subprocess.Popen", fake_popen)
    prefilter = RustPrefilter(
        mode="auto",
        binary=Path("fake-dopa-dialogue-filter.exe"),
        batch_size=4096,
        active=True,
    )

    accepted_first, rejected_first = prefilter.filter_batch([conversation], policy=CleanPolicy(), cache_dir=tmp_path)
    accepted_second, rejected_second = prefilter.filter_batch([conversation], policy=CleanPolicy(), cache_dir=tmp_path)
    prefilter.close()

    assert accepted_first == [conversation]
    assert accepted_second == [conversation]
    assert not rejected_first
    assert not rejected_second
    assert len(popen_calls) == 1
    assert popen_calls[0][0][1] == "filter-batch-stdin"
    assert fake_process.stdin.writes.count("1\n") == 2
    assert fake_process.stdin.closed is True


def test_rust_prefilter_auto_falls_back_to_python_batch_when_binary_fails(monkeypatch, tmp_path):
    spec = validate_selected_sources(["aya_dataset"], allow_restricted_license=False)[0]
    conversation = convert_rows(
        spec,
        [
            {
                "inputs": "怎么调试 pytest 报错?",
                "targets": "先看堆栈，再缩小复现，最后补回归测试。",
                "language_code": "zho",
            }
        ],
    )[0]

    def fake_session_filter_batch(*args, **kwargs):
        raise RuntimeError("Rust prefilter failed: 拒绝访问。 (os error 5)")

    monkeypatch.setattr(
        "scripts.clean_dialogue_corpus.RustPrefilterSession.filter_batch",
        fake_session_filter_batch,
    )
    prefilter = RustPrefilter(
        mode="auto",
        binary=Path("fake-dopa-dialogue-filter.exe"),
        batch_size=4096,
        active=True,
    )

    accepted, rejected = prefilter.filter_batch(
        [conversation],
        policy=CleanPolicy(),
        cache_dir=tmp_path,
    )

    assert accepted == [conversation]
    assert not rejected
    assert prefilter.failed_batches == 1
    assert prefilter.to_dict()["fallback_reasons"] == {"runtime_error": 1}


def test_rows_api_streaming_yields_first_page_before_fetching_next_page(monkeypatch, tmp_path):
    spec = validate_selected_sources(["aya_dataset"], allow_restricted_license=False)[0]
    calls = []

    def fake_fetch_json(url, *, timeout):
        calls.append(url)
        if len(calls) > 1:
            raise AssertionError("rows-api streaming fetched a second page before yielding the first record")
        row = {
            "inputs": "How do I debug a pytest fixture that fails with a traceback?",
            "targets": (
                "Read the traceback, isolate the fixture state, reproduce the smallest failing case, "
                "then add a regression test after the fix."
            ),
            "language_code": "eng",
        }
        return {"rows": [{"row": dict(row, id=f"row-{index}")} for index in range(100)]}

    monkeypatch.setattr("scripts.clean_dialogue_corpus._fetch_json", fake_fetch_json)

    iterator = iter_source_conversations(
        spec,
        limit=0,
        backend="rows-api",
        hf_endpoint="https://hf-mirror.com",
        cache_dir=tmp_path,
    )
    conversation = next(iterator)

    assert conversation.messages[0]["content"].startswith("How do I debug")
    assert conversation.messages[-1]["content"].startswith("Read the traceback")
    assert len(calls) == 1


def test_hf_mirror_parquet_backend_streams_downloaded_parquet_rows(monkeypatch, tmp_path):
    spec = validate_selected_sources(["stack_exchange_preferences"], allow_restricted_license=False)[0]
    parquet_path = tmp_path / "sample.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "qid": 7,
                "question": "<p>How can I debug a pytest fixture failure?</p>",
                "answers": [
                    {"answer_id": 1, "pm_score": 1, "selected": False, "text": "<p>Rerun it.</p>"},
                    {
                        "answer_id": 2,
                        "pm_score": 9,
                        "selected": True,
                        "text": (
                            "<p>Read the traceback, isolate fixture state, reproduce the smallest failing case, "
                            "and add a regression test.</p>"
                        ),
                    },
                ],
                "date": "2026/07/04",
                "metadata": ["https://stackoverflow.com/questions/7"],
            }
        ]
    )
    pq.write_table(table, parquet_path)
    downloads = []

    def fake_files(*args, **kwargs):
        return ["data/stackoverflow/train-00000-of-00001.parquet"]

    def fake_download(*args, **kwargs):
        downloads.append(args[1])
        return parquet_path

    monkeypatch.setattr("scripts.clean_dialogue_corpus._hf_mirror_dataset_files", fake_files, raising=False)
    monkeypatch.setattr("scripts.clean_dialogue_corpus._download_hf_mirror_file", fake_download, raising=False)

    iterator = iter_source_conversations(
        spec,
        limit=0,
        backend="hf-mirror-parquet",
        hf_endpoint="https://hf-mirror.com",
        cache_dir=tmp_path,
    )
    conversation = next(iterator)

    assert downloads == ["data/stackoverflow/train-00000-of-00001.parquet"]
    assert conversation.source_id == "7"
    assert conversation.messages[0]["content"] == "How can I debug a pytest fixture failure?"
    assert conversation.messages[-1]["content"].startswith("Read the traceback")
    assert conversation.metadata["answer_score"] == 9
    assert conversation.metadata["answer_accepted"] is True


def test_rust_prefilter_auto_disables_after_first_runtime_failure(monkeypatch, tmp_path):
    spec = validate_selected_sources(["aya_dataset"], allow_restricted_license=False)[0]
    conversation = convert_rows(
        spec,
        [
            {
                "inputs": "How do I debug pytest fixture failures?",
                "targets": "Read the traceback, isolate fixture state, reproduce the smallest failing case, then add a regression test.",
                "language_code": "eng",
            }
        ],
    )[0]
    calls = 0

    def fake_session_filter_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("Rust prefilter failed: 拒绝访问。 (os error 5)")

    monkeypatch.setattr(
        "scripts.clean_dialogue_corpus.RustPrefilterSession.filter_batch",
        fake_session_filter_batch,
    )
    prefilter = RustPrefilter(
        mode="auto",
        binary=Path("fake-dopa-dialogue-filter.exe"),
        batch_size=4096,
        active=True,
    )

    accepted_first, rejected_first = prefilter.filter_batch(
        [conversation],
        policy=CleanPolicy(),
        cache_dir=tmp_path,
    )
    accepted_second, rejected_second = prefilter.filter_batch(
        [conversation],
        policy=CleanPolicy(),
        cache_dir=tmp_path,
    )

    assert accepted_first == [conversation]
    assert accepted_second == [conversation]
    assert not rejected_first
    assert not rejected_second
    assert calls == 1
    assert prefilter.active is False
    assert prefilter.failed_batches == 1
    assert prefilter.to_dict()["disabled_reason"].startswith("runtime_error")


def test_should_emit_progress_respects_record_interval_and_min_seconds():
    assert should_emit_progress(
        written_records=1000,
        progress_every=1000,
        now=20.0,
        last_progress_at=10.0,
        min_seconds=5.0,
    )
    assert not should_emit_progress(
        written_records=1500,
        progress_every=1000,
        now=30.0,
        last_progress_at=10.0,
        min_seconds=5.0,
    )
    assert not should_emit_progress(
        written_records=2000,
        progress_every=1000,
        now=14.0,
        last_progress_at=10.0,
        min_seconds=5.0,
    )
    assert should_emit_progress(
        written_records=2000,
        progress_every=1000,
        now=14.0,
        last_progress_at=10.0,
        min_seconds=0.0,
    )


def test_stream_resume_state_counts_existing_shards_and_seeds_deduplication(tmp_path):
    shard_dir = tmp_path / "clean" / "shards"
    shard_dir.mkdir(parents=True)
    text = (
        "User: How do I debug a pytest failure with a traceback and a flaky fixture?\n\n"
        "Assistant: Read the assertion, isolate the fixture, reproduce the smallest failing case, "
        "then add a regression test after the fix."
    )
    record = {
        "text": text,
        "language": "en",
        "source": "unit/source",
        "source_id": "old-1",
        "license": "mit",
        "category": "debug",
        "quality_score": 0.9,
        "quality_tags": ["debug"],
        "metadata": {"message_tree_id": "tree-1"},
    }
    line = json_line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with (shard_dir / "shard-00000.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(line)

    cleaner = StreamingDialogueCleaner(CleanPolicy(min_score=0.0, min_chars=10))
    state = load_stream_resume_state(shard_dir, cleaner)

    assert state.written_records == 1
    assert state.total_bytes == len(json_line.encode("utf-8"))
    assert state.written_language_records == {"en": 1}
    assert state.categories["debug"] == 1
    assert state.source_stats["unit/source"]["accepted"] == 1
    assert state.next_shard_index == 1

    duplicate = ConversationRecord(
        source="unit/source",
        license_id="mit",
        source_id="new-duplicate-id",
        messages=[
            {"role": "user", "content": "How do I debug a pytest failure with a traceback and a flaky fixture?"},
            {
                "role": "assistant",
                "content": (
                    "Read the assertion, isolate the fixture, reproduce the smallest failing case, "
                    "then add a regression test after the fix."
                ),
            },
        ],
    )

    assert cleaner.accept(duplicate) is None
    assert cleaner.report.rejected["duplicate"] == 1


def test_next_stream_shard_index_ignores_legacy_names(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / "shard-00003.jsonl").write_text("", encoding="utf-8")
    (shard_dir / "legacy-run2-shard-00000.jsonl").write_text("", encoding="utf-8")

    assert next_stream_shard_index(shard_dir) == 4


def test_stream_resume_state_prefers_report_for_fast_restart(tmp_path):
    shard_dir = tmp_path / "clean" / "shards"
    report_dir = tmp_path / "reports"
    shard_dir.mkdir(parents=True)
    report_dir.mkdir()
    (report_dir / "quality_report.json").write_text(
        json.dumps(
            {
                "target": {
                    "written_bytes": 1234,
                    "written_records": 12,
                    "language_written_bytes": {"en": 700, "zh": 534},
                    "language_written_records": {"en": 7, "zh": 5},
                },
                "clean_report": {
                    "categories": {"debug": 12},
                    "sources": {"unit/source": 12},
                },
                "source_stats": {
                    "unit/source": {
                        "converted_conversations": 20,
                        "accepted": 12,
                        "exhausted": True,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cleaner = StreamingDialogueCleaner(CleanPolicy())
    state = load_stream_resume_state(shard_dir, cleaner, report_dir=report_dir)

    assert state.total_bytes == 1234
    assert state.written_records == 12
    assert state.written_language_records == {"en": 7, "zh": 5}
    assert state.source_stats["unit/source"]["exhausted"] is True
