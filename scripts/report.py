from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a DOPA final_report.json as Markdown.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    md = render_report(report)
    out = Path(args.out) if args.out else Path(args.report).with_suffix(".md")
    out.write_text(md, encoding="utf-8")
    print(out)


def render_report(report: dict) -> str:
    lines = [
        "# DOPA Training Report",
        "",
        f"- Config: `{report.get('config')}`",
        f"- Output: `{report.get('out_dir')}`",
        "",
        "## Stages",
        "",
        "| Stage | Step | Loss | LM Loss |",
        "| --- | ---: | ---: | ---: |",
    ]
    for stage in report.get("stages", []):
        lines.append(
            f"| {stage.get('stage')} | {stage.get('step', '')} | "
            f"{_fmt(stage.get('loss'))} | {_fmt(stage.get('lm_loss'))} |"
        )
    eval_metrics = report.get("eval", {})
    lines += [
        "",
        "## Evaluation",
        "",
        f"- Loss: {_fmt(eval_metrics.get('loss'))}",
        f"- Perplexity: {_fmt(eval_metrics.get('perplexity'))}",
        f"- Difficulty mean: {_fmt(eval_metrics.get('difficulty_mean'))}",
        f"- Cold unit count mean: {_fmt(eval_metrics.get('cold_unit_count_mean'))}",
        "",
        "## Samples",
        "",
    ]
    for sample in eval_metrics.get("samples", []):
        lines += [
            f"Prompt: `{sample.get('prompt', '').replace(chr(10), '\\n')}`",
            "",
            "```text",
            sample.get("completion", ""),
            "```",
            "",
        ]
    writeback = report.get("writeback", {})
    lines += [
        "## Cold Unit Writeback",
        "",
        f"- Enabled: {writeback.get('enabled')}",
        f"- Count: {writeback.get('count', 0)}",
        f"- Format: {writeback.get('format', 'none')}",
    ]
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    main()
