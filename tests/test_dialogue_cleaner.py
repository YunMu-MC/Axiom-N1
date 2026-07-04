import json

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
    score_conversation,
    split_train_valid,
    write_sharded_jsonl,
)


def test_wildchat_normalizer_keeps_messages_and_drops_private_metadata():
    row = {
        "conversation_hash": "abc123",
        "model": "gpt-4-0314",
        "language": "English",
        "redacted": False,
        "hashed_ip": "secret-hash",
        "header": {"user-agent": "browser"},
        "conversation": [
            {"role": "user", "content": "Can you review this Python function?\n```python\nprint(1)\n```"},
            {"role": "assistant", "content": "Yes. It is valid, but wrap it in a main guard for scripts."},
        ],
    }

    conversation = normalize_wildchat_row(row, source="allenai/WildChat-1M", license_id="odc-by")

    assert conversation is not None
    assert conversation.source_id == "abc123"
    assert [message["role"] for message in conversation.messages] == ["user", "assistant"]
    serialized = json.dumps(conversation.to_training_record(), ensure_ascii=False)
    assert "hashed_ip" not in serialized
    assert "user-agent" not in serialized


def test_prompt_target_normalizer_supports_aya_language_codes_and_drops_annotator_id():
    row = {
        "inputs": "怎么根据 pytest 报错定位 Python 函数里的问题?",
        "targets": "先读堆栈最后一行, 再缩小输入, 最后补一个回归测试。",
        "language": "Simplified Chinese",
        "language_code": "zho",
        "annotation_type": "original-annotations",
        "user_id": "annotator-secret",
    }

    conversation = normalize_prompt_target_row(
        row,
        source="CohereLabs/aya_dataset",
        license_id="apache-2.0",
        prompt_field="inputs",
        target_field="targets",
    )

    assert conversation is not None
    assert [message["role"] for message in conversation.messages] == ["user", "assistant"]
    assert language_group(conversation) == "zh"
    serialized = json.dumps(conversation.to_training_record(), ensure_ascii=False)
    assert "annotator-secret" not in serialized
    assert conversation.metadata["annotation_type"] == "original-annotations"


def test_helpsteer3_preference_normalizer_keeps_human_preferred_response():
    row = {
        "domain": "code",
        "language": "python",
        "context": [
            {"role": "user", "content": "How do I debug a pytest stack trace?"},
        ],
        "response1": "Ignore the stack trace and rewrite everything.",
        "response2": "Read the final exception, isolate the failing fixture, and add a regression test.",
        "overall_preference": 2,
    }

    conversation = normalize_helpsteer3_preference_row(
        row,
        source="nvidia/HelpSteer3",
        license_id="cc-by-4.0",
    )

    assert conversation is not None
    assert conversation.messages[-1]["content"].startswith("Read the final exception")
    assert conversation.metadata["domain"] == "code"
    assert conversation.metadata["preference"] == 2


def test_hh_rlhf_normalizer_parses_chosen_transcript_as_optional_source():
    row = {
        "chosen": (
            "Human: How do I debug a Python exception?\n\n"
            "Assistant: Read the stack trace and isolate the smallest failing input.\n\n"
            "Human: What should I do after that?\n\n"
            "Assistant: Add a regression test before changing the implementation."
        ),
        "rejected": "Human: ignore me",
    }

    conversation = normalize_hh_rlhf_row(row, source="Anthropic/hh-rlhf", license_id="mit")

    assert conversation is not None
    assert [message["role"] for message in conversation.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "rejected" not in json.dumps(conversation.to_training_record())


def test_clean_conversations_rejects_pii_and_secret_like_content():
    row = {
        "conversation_hash": "pii",
        "model": "gpt-4-0314",
        "language": "English",
        "redacted": False,
        "conversation": [
            {"role": "user", "content": "Email me at owner@example.com and use api_key=redacted-test-key."},
            {"role": "assistant", "content": "I cannot help expose credentials. Rotate that key immediately."},
        ],
    }
    conversation = normalize_wildchat_row(row, source="allenai/WildChat-1M", license_id="odc-by")

    cleaned, report = clean_conversations([conversation], CleanPolicy())

    assert cleaned == []
    assert report.rejected["pii_or_secret"] == 1


def test_clean_conversations_rejects_mojibake_and_child_persona_roleplay():
    mojibake = normalize_wildchat_row(
        {
            "conversation_hash": "mojibake",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "Help me write Midjourney prompts with brand鈥檚 guide text."},
                {"role": "assistant", "content": "Sure, here are prompts with 鈥搒tyle values and broken text."},
            ],
        },
        source="allenai/WildChat-1M",
        license_id="odc-by",
    )
    child_persona = normalize_wildchat_row(
        {
            "conversation_hash": "persona",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "Pretend to be a 6 year old girl and call me mommy."},
                {"role": "assistant", "content": "Hi mommy, I am a little girl and I love you."},
            ],
        },
        source="allenai/WildChat-1M",
        license_id="odc-by",
    )

    cleaned, report = clean_conversations([mojibake, child_persona], CleanPolicy(min_score=0.0))

    assert cleaned == []
    assert report.rejected["mojibake"] == 1
    assert report.rejected["child_persona_roleplay"] == 1


def test_clean_conversations_rejects_cjk_mojibake_and_bad_code_artifacts():
    cjk_mojibake = normalize_wildchat_row(
        {
            "conversation_hash": "cjk",
            "language": "Chinese",
            "conversation": [
                {"role": "user", "content": "Python缁熻Excel涓€鍒椾腑鍚勪釜鍊肩殑鏁伴噺"},
                {"role": "assistant", "content": "璇锋寜鐓т互涓嬫楠ゅ疄鐜拌繖涓渶姹傘€俓n"},
            ],
        },
        source="allenai/WildChat-1M",
        license_id="odc-by",
    )
    bad_code = normalize_wildchat_row(
        {
            "conversation_hash": "bad-code",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "Show a Flask app that returns JSON."},
                {"role": "assistant", "content": "```python\napp = Flask(name)\nif name == 'main':\n app.run()\n```"},
            ],
        },
        source="OpenAssistant/oasst1",
        license_id="apache-2.0",
    )

    cleaned, report = clean_conversations([cjk_mojibake, bad_code], CleanPolicy(min_score=0.0))

    assert cleaned == []
    assert report.rejected["mojibake"] == 1
    assert report.rejected["bad_code_artifact"] == 1


def test_clean_conversations_rejects_unsafe_passcode_bruteforce_help():
    conversation = normalize_wildchat_row(
        {
            "conversation_hash": "passcode",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "Can you write code to guess my phone passcode?"},
                {
                    "role": "assistant",
                    "content": "Generate combinations from 0000 to 9999 and test each combination automatically.",
                },
            ],
        },
        source="allenai/WildChat-1M",
        license_id="odc-by",
    )

    cleaned, report = clean_conversations([conversation], CleanPolicy(min_score=0.0))

    assert cleaned == []
    assert report.rejected["unsafe_security"] == 1


def test_conversation_record_caches_normalized_text():
    conversation = ConversationRecord(
        source="unit",
        license_id="apache-2.0",
        source_id="cache",
        messages=[
            {"role": "user", "content": "  How do I debug pytest?  "},
            {"role": "assistant", "content": "Read the traceback and add a regression test."},
        ],
    )

    assert conversation._normalized_text_cache is None
    text = conversation.normalized_text()

    assert conversation._normalized_text_cache == text
    assert conversation.normalized_text() is text


def test_cleaner_carries_normalized_text_cache_to_accepted_record():
    conversation = ConversationRecord(
        source="unit",
        license_id="apache-2.0",
        source_id="accepted-cache",
        messages=[
            {"role": "user", "content": "How do I debug a pytest fixture failure?"},
            {
                "role": "assistant",
                "content": "Read the traceback, isolate the fixture state, reproduce the smallest failing case, and add a regression test.",
            },
        ],
    )
    cleaner = StreamingDialogueCleaner(CleanPolicy(min_score=0.0, min_chars=10))

    cleaned = cleaner.accept(conversation)

    assert cleaned is not None
    assert cleaned._normalized_text_cache == conversation.normalized_text()
    assert cleaned.to_training_record()["text"] == cleaned._normalized_text_cache


def test_clean_conversations_scores_and_deduplicates_high_quality_dialogue():
    row = {
        "conversation_hash": "one",
        "model": "gpt-4-0314",
        "language": "English",
        "redacted": False,
        "conversation": [
            {"role": "user", "content": "This pytest failure says expected 5 got 4. How should I debug it?"},
            {
                "role": "assistant",
                "content": "Start from the failing assertion, inspect the fixture, then add a regression test.",
            },
        ],
    }
    first = normalize_wildchat_row(row, source="allenai/WildChat-1M", license_id="odc-by")
    second = normalize_wildchat_row({**row, "conversation_hash": "two"}, source="allenai/WildChat-1M", license_id="odc-by")

    cleaned, report = clean_conversations([first, second], CleanPolicy(min_score=0.55))

    assert len(cleaned) == 1
    assert report.rejected["duplicate"] == 1
    assert cleaned[0].quality_score >= 0.55
    assert cleaned[0].category == "debug"


def test_clean_conversations_scores_chinese_debug_dialogue_as_high_quality():
    conversation = normalize_wildchat_row(
        {
            "conversation_hash": "zh-debug",
            "language": "Chinese",
            "conversation": [
                {"role": "user", "content": "我的 Python 脚本运行时报错, 怎么根据堆栈信息定位问题?"},
                {
                    "role": "assistant",
                    "content": "先阅读最后一行异常类型, 再查看触发异常的函数参数, 最后写一个最小复现和回归测试。",
                },
            ],
        },
        source="allenai/WildChat-1M",
        license_id="odc-by",
    )

    cleaned, report = clean_conversations([conversation], CleanPolicy(min_score=0.60))

    assert report.accepted == 1
    assert cleaned[0].category == "debug"
    assert cleaned[0].to_training_record()["language"] == "zh"


def test_clean_conversations_can_keep_only_target_categories_and_one_tree_path():
    base = {
        "language": "English",
        "conversation": [
            {"role": "user", "content": "How do I debug a pytest failure with a stack trace?"},
            {"role": "assistant", "content": "Inspect the assertion, isolate the fixture, and add a regression test."},
        ],
    }
    debug_one = normalize_wildchat_row(
        {**base, "conversation_hash": "debug-1"},
        source="OpenAssistant/oasst1",
        license_id="apache-2.0",
    )
    debug_one.metadata["message_tree_id"] = "tree"
    debug_two = normalize_wildchat_row(
        {**base, "conversation_hash": "debug-2"},
        source="OpenAssistant/oasst1",
        license_id="apache-2.0",
    )
    debug_two.metadata["message_tree_id"] = "tree"
    general = normalize_wildchat_row(
        {
            "conversation_hash": "general",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "Why is the sky blue in simple terms?"},
                {"role": "assistant", "content": "Air molecules scatter shorter blue wavelengths more strongly."},
            ],
        },
        source="allenai/WildChat-1M",
        license_id="odc-by",
    )

    cleaned, report = clean_conversations(
        [debug_one, debug_two, general],
        CleanPolicy(min_score=0.0, target_categories=frozenset({"debug"})),
    )

    assert len(cleaned) == 1
    assert cleaned[0].category == "debug"
    assert report.rejected["duplicate_thread"] == 1
    assert report.rejected["category_not_target"] == 1


def test_streaming_cleaner_keeps_dedup_state_across_batches():
    row = {
        "conversation_hash": "same",
        "language": "English",
        "conversation": [
            {"role": "user", "content": "How do I debug a pytest failure with a stack trace?"},
            {"role": "assistant", "content": "Inspect the assertion, isolate the fixture, and add a regression test."},
        ],
    }
    first = normalize_wildchat_row(row, source="allenai/WildChat-1M", license_id="odc-by")
    second = normalize_wildchat_row({**row, "conversation_hash": "same-two"}, source="allenai/WildChat-1M", license_id="odc-by")
    cleaner = StreamingDialogueCleaner(CleanPolicy(min_score=0.0))

    assert cleaner.accept(first) is not None
    assert cleaner.accept(second) is None
    assert cleaner.report.rejected["duplicate"] == 1


def test_language_group_uses_metadata_and_text_heuristics():
    english = normalize_wildchat_row(
        {
            "conversation_hash": "en",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "How do I debug Python code?"},
                {"role": "assistant", "content": "Read the stack trace and isolate the failing function."},
            ],
        },
        source="allenai/WildChat-1M",
        license_id="odc-by",
    )
    chinese = normalize_wildchat_row(
        {
            "conversation_hash": "zh",
            "language": "Chinese",
            "conversation": [
                {"role": "user", "content": "怎么调试 Python 报错?"},
                {"role": "assistant", "content": "先看堆栈, 再缩小复现代码。"},
            ],
        },
        source="allenai/WildChat-1M",
        license_id="odc-by",
    )

    assert language_group(english) == "en"
    assert language_group(chinese) == "zh"


def test_write_sharded_jsonl_balances_target_language_bytes(tmp_path):
    records = [
        {"text": "English debugging answer " + "x" * 30, "language": "en"},
        {"text": "Another English answer " + "x" * 30, "language": "en"},
        {"text": "中文调试回答" + "中" * 30, "language": "zh"},
        {"text": "第二条中文回答" + "中" * 30, "language": "zh"},
    ]

    report = write_sharded_jsonl(
        tmp_path,
        records,
        target_bytes=10_000,
        shard_bytes=10_000,
        language_byte_targets={"en": 120, "zh": 220},
    )

    assert report.language_records["en"] == 1
    assert report.language_records["zh"] == 1
    assert report.language_bytes["en"] <= 120
    assert report.language_bytes["zh"] <= 220
    assert report.skipped["language_quota_full"] == 2


def test_write_sharded_jsonl_stops_at_target_bytes(tmp_path):
    records = [{"text": "x" * 20, "source": "test", "license": "apache-2.0"} for _ in range(20)]

    report = write_sharded_jsonl(tmp_path, records, target_bytes=180, shard_bytes=90)

    shards = sorted(tmp_path.glob("shard-*.jsonl"))
    assert report.records > 0
    assert report.bytes_written <= 180
    assert report.shards == len(shards)
    assert all(path.stat().st_size <= 180 for path in shards)


def test_financial_risk_language_is_not_security_category():
    conversation = normalize_wildchat_row(
        {
            "conversation_hash": "finance",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "How should I think about investment risk in cryptocurrency?"},
                {
                    "role": "assistant",
                    "content": "Diversify, size positions carefully, and do not treat volatility as certainty.",
                },
            ],
        },
        source="allenai/WildChat-1M",
        license_id="odc-by",
    )

    score, category, tags = score_conversation(conversation)

    assert score < 0.6
    assert category != "security_defensive"
    assert "security" not in tags


def test_clean_conversations_rejects_financial_market_advice():
    conversation = normalize_wildchat_row(
        {
            "conversation_hash": "stocks",
            "language": "English",
            "conversation": [
                {"role": "user", "content": "Write a script for NVIDIA stock alerts and investing decisions."},
                {
                    "role": "assistant",
                    "content": "Scrape Google every few seconds and use the result for market timing.",
                },
            ],
        },
        source="allenai/WildChat-1M",
        license_id="odc-by",
    )

    cleaned, report = clean_conversations([conversation], CleanPolicy(min_score=0.0))

    assert cleaned == []
    assert report.rejected["financial_market_advice"] == 1


def test_openassistant_low_quality_label_is_rejected():
    rows = [
        {
            "message_id": "root",
            "parent_id": None,
            "message_tree_id": "tree",
            "text": "What regulatory body should handle monopsony abuse?",
            "role": "prompter",
            "lang": "en",
            "review_result": True,
            "deleted": False,
            "synthetic": False,
            "rank": None,
            "labels": {"name": ["quality"], "value": [0.9]},
        },
        {
            "message_id": "bad",
            "parent_id": "root",
            "message_tree_id": "tree",
            "text": "Register the TESR in the same Minecraft mod file.",
            "role": "assistant",
            "lang": "en",
            "review_result": True,
            "deleted": False,
            "synthetic": False,
            "rank": 0,
            "labels": {"name": ["quality"], "value": [0.1]},
        },
    ]

    conversations = build_openassistant_conversations(rows, source="OpenAssistant/oasst1", license_id="apache-2.0")

    assert conversations == []


def test_openassistant_tree_rows_are_rebuilt_into_alternating_paths():
    rows = [
        {
            "message_id": "root",
            "parent_id": None,
            "message_tree_id": "tree",
            "text": "Explain binary search with a small Python example.",
            "role": "prompter",
            "lang": "en",
            "review_result": True,
            "deleted": False,
            "synthetic": False,
            "rank": None,
        },
        {
            "message_id": "reply",
            "parent_id": "root",
            "message_tree_id": "tree",
            "text": "Binary search repeatedly halves a sorted range until the target is found or absent.",
            "role": "assistant",
            "lang": "en",
            "review_result": True,
            "deleted": False,
            "synthetic": False,
            "rank": 0,
        },
    ]

    conversations = build_openassistant_conversations(rows, source="OpenAssistant/oasst1", license_id="apache-2.0")

    assert len(conversations) == 1
    assert [message["role"] for message in conversations[0].messages] == ["user", "assistant"]
    assert conversations[0].metadata["message_tree_id"] == "tree"


def test_openassistant_rows_can_stream_by_consecutive_tree_without_full_split_buffer():
    rows = [
        {
            "message_id": "tree-a-root",
            "parent_id": None,
            "message_tree_id": "tree-a",
            "text": "Explain a pytest stack trace for a Python test failure.",
            "role": "prompter",
            "lang": "en",
            "review_result": True,
            "deleted": False,
            "synthetic": False,
            "rank": None,
        },
        {
            "message_id": "tree-a-reply",
            "parent_id": "tree-a-root",
            "message_tree_id": "tree-a",
            "text": "Read the final exception, inspect the failing fixture, and add a regression test.",
            "role": "assistant",
            "lang": "en",
            "review_result": True,
            "deleted": False,
            "synthetic": False,
            "rank": 0,
        },
        {
            "message_id": "tree-b-root",
            "parent_id": None,
            "message_tree_id": "tree-b",
            "text": "怎么调试 Python 运行时报错?",
            "role": "prompter",
            "lang": "zh",
            "review_result": True,
            "deleted": False,
            "synthetic": False,
            "rank": None,
        },
        {
            "message_id": "tree-b-reply",
            "parent_id": "tree-b-root",
            "message_tree_id": "tree-b",
            "text": "先看堆栈最后一行, 再缩小复现, 最后补一个回归测试。",
            "role": "assistant",
            "lang": "zh",
            "review_result": True,
            "deleted": False,
            "synthetic": False,
            "rank": 0,
        },
    ]

    streamed = list(
        iter_openassistant_conversations_by_tree(
            iter(rows),
            source="OpenAssistant/oasst2",
            license_id="apache-2.0",
        )
    )
    full_buffer = build_openassistant_conversations(
        rows,
        source="OpenAssistant/oasst2",
        license_id="apache-2.0",
    )

    assert [item.source_id for item in streamed] == [item.source_id for item in full_buffer]
    assert [item.metadata["message_tree_id"] for item in streamed] == ["tree-a", "tree-b"]


def test_split_train_valid_is_deterministic_and_keeps_validation_non_empty():
    records = [
        normalize_wildchat_row(
            {
                "conversation_hash": str(index),
                "language": "English",
                "conversation": [
                    {"role": "user", "content": f"How do I debug failure {index} with pytest output and stack trace?"},
                    {
                        "role": "assistant",
                        "content": "Inspect the assertion, reduce the fixture, and write a regression test.",
                    },
                ],
            },
            source="allenai/WildChat-1M",
            license_id="odc-by",
        )
        for index in range(5)
    ]
    cleaned, _ = clean_conversations(records, CleanPolicy(min_score=0.45))

    train, valid = split_train_valid(cleaned, valid_ratio=0.2, seed=7)
    again_train, again_valid = split_train_valid(cleaned, valid_ratio=0.2, seed=7)

    assert [item.source_id for item in train] == [item.source_id for item in again_train]
    assert [item.source_id for item in valid] == [item.source_id for item in again_valid]
    assert len(train) == 4
    assert len(valid) == 1


def test_markdown_report_contains_quality_counts():
    row = {
        "conversation_hash": "report",
        "language": "English",
        "conversation": [
            {"role": "user", "content": "How do I debug a failing pytest stack trace?"},
            {"role": "assistant", "content": "Read the assertion, inspect inputs, and add a regression test."},
        ],
    }
    cleaned, report = clean_conversations(
        [normalize_wildchat_row(row, source="allenai/WildChat-1M", license_id="odc-by")],
        CleanPolicy(min_score=0.45),
    )

    markdown = render_markdown_report(report, train_count=len(cleaned), valid_count=0)

    assert "accepted=1" in markdown
    assert "debug" in markdown
    assert "allenai/WildChat-1M" in markdown
