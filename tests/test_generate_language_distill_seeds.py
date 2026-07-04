from scripts.generate_language_distill_seeds import generate_prompts


def test_generated_chinese_prompts_are_not_mojibake():
    prompts = generate_prompts(12, seed=7)
    chinese = [prompt for prompt in prompts if prompt["lang"] == "zh"]

    assert chinese
    assert any("你是一名语言能力教师" in prompt["system"] for prompt in chinese)
    assert all("浣犳槸" not in prompt["system"] for prompt in chinese)
    assert all("鈥" not in prompt["user"] for prompt in chinese)
