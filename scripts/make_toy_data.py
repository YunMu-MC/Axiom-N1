from __future__ import annotations

import json
from pathlib import Path


ROWS = [
    {
        "text": "def add(a, b):\n    return a + b\n",
        "skeleton": {"name": "simple_function", "steps": [{"op": "parse_input"}, {"op": "emit_output"}]},
    },
    {
        "text": "from collections import deque\nq = deque([start])\nwhile q:\n    node = q.popleft()\n",
        "skeleton": {"name": "grid_bfs", "steps": [{"op": "graph_search"}, {"op": "loop"}]},
    },
    {
        "text": "intervals.sort(key=lambda x: x[0])\nfor start, end in intervals:\n    pass\n",
        "skeleton": {"name": "interval_sort", "steps": [{"op": "sort"}, {"op": "loop"}]},
    },
]


def main() -> None:
    path = Path("data/toy.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in ROWS * 100:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
