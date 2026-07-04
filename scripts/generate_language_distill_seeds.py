from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


CATEGORIES = [
    "language_understanding",
    "rewrite",
    "summarization",
    "translation",
    "tone_control",
    "ambiguity_clarification",
]

ZH_SYSTEM = (
    "你是一名语言能力教师。回答要清晰、自然、准确，重点展示理解、表达和措辞取舍能力。"
    "不要展示隐藏推理，不要写成俏皮闲聊。"
)
EN_SYSTEM = (
    "You are a language-skill teacher. Answer clearly, naturally, and accurately, "
    "focusing on comprehension, expression, and wording choices. Do not reveal hidden reasoning. "
    "Do not use playful banter."
)

ZH_SENTENCES = [
    "我不是反对这个安排，只是觉得我们好像还没有把真正的问题说清楚。",
    "如果大家都觉得这样可以，那我也不是不能接受，只是心里还有点不踏实。",
    "我知道你很忙，但这件事一直没有回应，我不知道该不该继续等。",
    "这不是一句道歉就能解决的事，关键是以后怎么避免重复发生。",
    "我不想把话说得太重，但现在的沟通方式确实让我很被动。",
    "我们不一定要马上做决定，但至少应该先把标准统一下来。",
    "我担心的不是进度慢，而是每个人理解的目标可能并不一样。",
    "这件事如果只看结果还可以，但中间的沟通成本已经明显偏高。",
]

EN_SENTENCES = [
    "I am not against the proposal; I just do not think we have named the actual problem yet.",
    "If everyone is comfortable with that, I can accept it, but I still feel there is some risk.",
    "I know you are busy, but without a response I do not know whether I should keep waiting.",
    "This is not just about saying sorry; it is about preventing the same issue from happening again.",
    "I do not want to overstate it, but the current communication pattern leaves me stuck.",
    "We do not need to decide immediately, but we should at least agree on the criteria first.",
    "My concern is not that the project is slow, but that we may be aiming at different targets.",
    "The final result may be acceptable, but the communication cost has become too high.",
]

ZH_TOPICS = [
    "活动延期通知",
    "项目进度说明",
    "客户反馈摘要",
    "会议结论整理",
    "合作边界说明",
    "学习计划调整",
    "团队沟通建议",
    "服务说明文案",
    "产品上线提醒",
    "问题复盘记录",
]

EN_TOPICS = [
    "launch delay update",
    "project status note",
    "customer feedback summary",
    "meeting decision recap",
    "collaboration boundary note",
    "study plan adjustment",
    "team communication advice",
    "service description copy",
    "product release reminder",
    "incident review note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate language-only distillation seed prompts.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_prompt(index: int, lang: str, category: str, rng: random.Random) -> dict:
    if lang == "zh":
        return _build_zh_prompt(index, category, rng)
    return _build_en_prompt(index, category, rng)


def generate_prompts(count: int, seed: int) -> list[dict]:
    if count <= 0:
        raise RuntimeError("--count must be positive")
    rng = random.Random(seed)
    prompts: list[dict] = []
    for index in range(count):
        lang = "zh" if index % 2 == 0 else "en"
        category = CATEGORIES[(index // 2) % len(CATEGORIES)]
        prompts.append(build_prompt(index, lang, category, rng))
    return prompts


def main() -> int:
    args = parse_args()
    prompts = generate_prompts(args.count, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote={len(prompts)} out={args.out}")
    return 0


def _build_zh_prompt(index: int, category: str, rng: random.Random) -> dict:
    sentence = rng.choice(ZH_SENTENCES)
    topic = rng.choice(ZH_TOPICS)
    prompt_id = f"zh_lang_batch_{index:06d}"
    user_by_category = {
        "language_understanding": (
            f"请解释这句话真正表达的含义和情绪，不超过三句话：\n“{sentence}”"
        ),
        "rewrite": (
            f"把下面这段关于“{topic}”的话改写得更清楚、更稳妥，保留原意：\n“{sentence}”"
        ),
        "summarization": (
            f"围绕“{topic}”写一段 120 字以内的原始说明，再把它压缩成 4 条要点，并标出最重要的一条。"
        ),
        "translation": (
            f"把这句话翻成自然英文，不要逐字硬译，并说明一个措辞选择：\n“{sentence}”"
        ),
        "tone_control": (
            f"围绕“{topic}”写同一个意思的三种语气：正式、温和、直接。每种一句。"
        ),
        "ambiguity_clarification": (
            f"用户说“这段关于{topic}的文字不太对”。请只从文字表达角度说明歧义，并给 3 个澄清问题。"
        ),
    }
    return {
        "prompt_id": prompt_id,
        "lang": "zh",
        "category": category,
        "system": ZH_SYSTEM,
        "user": user_by_category[category],
        "metadata": {"batch": "language_auto", "template": category},
    }


def _build_en_prompt(index: int, category: str, rng: random.Random) -> dict:
    sentence = rng.choice(EN_SENTENCES)
    topic = rng.choice(EN_TOPICS)
    prompt_id = f"en_lang_batch_{index:06d}"
    user_by_category = {
        "language_understanding": (
            "Explain the implied meaning and emotional concern in this sentence in no more than three sentences:\n"
            f'"{sentence}"'
        ),
        "rewrite": (
            f"Rewrite this note about a {topic} so it is clearer and more measured while preserving the meaning:\n"
            f'"{sentence}"'
        ),
        "summarization": (
            f"Write a short raw update of under 120 words about a {topic}, "
            "then compress it into 4 concise bullets and mark the most important one."
        ),
        "translation": (
            f"Translate this into natural Chinese, not word-for-word, and explain one wording choice:\n"
            f'"{sentence}"'
        ),
        "tone_control": (
            f"Write the same message about a {topic} in three tones: formal, warm, and direct. "
            "One sentence for each tone."
        ),
        "ambiguity_clarification": (
            f'A user says, "This wording about the {topic} feels off." '
            "Identify the language-only ambiguity and ask 3 clarification questions."
        ),
    }
    return {
        "prompt_id": prompt_id,
        "lang": "en",
        "category": category,
        "system": EN_SYSTEM,
        "user": user_by_category[category],
        "metadata": {"batch": "language_auto", "template": category},
    }


if __name__ == "__main__":
    raise SystemExit(main())
