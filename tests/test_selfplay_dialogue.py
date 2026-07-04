import json

from scripts.selfplay_dialogue import (
    SelfPlayTask,
    build_answer_messages,
    build_user_generation_messages,
    clean_generated_user_message,
    clean_generated_assistant_message,
    contains_mojibake,
    completed_prompt_ids,
    iter_selfplay_tasks,
)


def test_selfplay_tasks_are_balanced_and_deterministic():
    tasks = list(iter_selfplay_tasks(count=6, start_index=10, seed=123))

    assert [task.prompt_id for task in tasks] == [
        "selfplay_zh_000010",
        "selfplay_en_000011",
        "selfplay_zh_000012",
        "selfplay_en_000013",
        "selfplay_zh_000014",
        "selfplay_en_000015",
    ]
    assert {task.lang for task in tasks} == {"zh", "en"}
    assert all(task.category for task in tasks)


def test_user_generation_prompt_asks_only_for_user_message():
    task = SelfPlayTask(
        prompt_id="selfplay_zh_000001",
        lang="zh",
        category="language_understanding",
        topic="项目沟通",
    )

    messages = build_user_generation_messages(task)

    assert messages[0]["role"] == "system"
    assert "只输出用户消息" in messages[0]["content"]
    assert "不要回答" in messages[0]["content"]
    assert "项目沟通" in messages[-1]["content"]


def test_answer_prompt_preserves_generated_user_message():
    task = SelfPlayTask(
        prompt_id="selfplay_en_000002",
        lang="en",
        category="rewrite",
        topic="team planning",
    )
    user_message = "Could you make this update sound calmer without changing the meaning?"

    messages = build_answer_messages(task, user_message)

    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": user_message}
    assert "natural" in messages[0]["content"].lower()


def test_clean_generated_user_message_removes_role_prefix_and_quotes():
    assert clean_generated_user_message('User: "我有点担心这个安排是不是太仓促了。"') == "我有点担心这个安排是不是太仓促了。"
    assert clean_generated_user_message("Assistant: 这是回答") == ""


def test_completed_prompt_ids_requires_raw_and_train_records(tmp_path):
    raw = tmp_path / "raw_selfplay.jsonl"
    train = tmp_path / "distilled_train.jsonl"
    raw.write_text(
        "\n".join([json.dumps({"prompt_id": "done"}), json.dumps({"prompt_id": "raw_only"})]) + "\n",
        encoding="utf-8",
    )
    train.write_text(
        "\n".join(
            [
                json.dumps({"metadata": {"prompt_id": "done"}}),
                json.dumps({"metadata": {"prompt_id": "train_only"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert completed_prompt_ids(raw, train) == {"done"}



def test_zh_selfplay_prompts_are_readable_utf8_not_mojibake():
    task = next(iter_selfplay_tasks(count=1, start_index=0, seed=123))

    messages = build_user_generation_messages(task)
    joined = "\n".join(message["content"] for message in messages)

    assert task.topic in {"项目沟通", "学习计划", "会议结论", "客户反馈", "服务说明", "团队协作", "产品上线", "问题复盘"}
    assert "只输出用户消息" in joined
    assert "不要回答" in joined
    for marker in ("椤", "浣", "鍙", "涓", "闂"):
        assert marker not in joined



def test_selfplay_rejects_mojibake_generated_messages():
    bad = (
        "\u6211\u60f3\u77e5\u9053\u4e3a\u4ec0\u4e48\u8fd9\u6b21\u7684\u9879\u76ee\u5ef6\u671f\u4e86\uff1f"
        .encode("utf-8")
        .decode("gbk", errors="replace")
    )

    assert contains_mojibake(bad)
    assert clean_generated_user_message(bad) == ""
    assert clean_generated_assistant_message("Assistant: " + bad) == ""


def test_selfplay_accepts_readable_chinese_generated_messages():
    good = "我想知道这次项目延期的主要原因，能不能帮我复盘一下沟通环节？"

    assert not contains_mojibake(good)
    assert clean_generated_user_message("用户：" + good) == good
    assert clean_generated_assistant_message(good) == good
