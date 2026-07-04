import json

from scripts.generate_skeleton_tasks import function_to_skeleton, iter_python_function_skeleton_records


def test_function_to_skeleton_extracts_name_params_and_steps():
    source = """
def solve(nums, target=0):
    total = 0
    for item in nums:
        if item > target:
            total += item
    return total
"""

    skeleton = function_to_skeleton(source)

    assert skeleton["name"] == "solve"
    assert skeleton["params"] == ["nums", "target"]
    assert {"op": "loop"} in skeleton["steps"]
    assert {"op": "branch"} in skeleton["steps"]
    assert skeleton["steps"][-1]["op"] == "return"


def test_iter_python_function_skeleton_records_builds_skeleton_prompt(tmp_path):
    source_path = tmp_path / "sample.py"
    source_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    records = list(iter_python_function_skeleton_records([source_path], sample_rate=1.0, seed=7))

    assert len(records) == 1
    assert records[0]["text"].startswith("[skeleton]\n")
    assert "Skeleton JSON:" in records[0]["text"]
    assert json.loads(records[0]["text"].split("Skeleton JSON:\n", 1)[1]) == records[0]["skeleton"]
