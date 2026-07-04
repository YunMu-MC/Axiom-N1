from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CopyStats:
    copied_files: int = 0
    skipped_files: int = 0
    copied_bytes: int = 0
    runs: list[dict] = field(default_factory=list)


def run_sort_key(path: Path) -> tuple[int, str]:
    digits = ""
    for char in reversed(path.name):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    return (int(digits) if digits else 0, path.name)


def next_shard_index(shard_dir: Path) -> int:
    max_index = -1
    if not shard_dir.exists():
        return 0
    for path in shard_dir.glob("shard-*.jsonl"):
        suffix = path.stem[len("shard-") :]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def count_shards(shard_dir: Path) -> int:
    if not shard_dir.exists():
        return 0
    return sum(1 for path in shard_dir.glob("*.jsonl") if path.is_file() and path.stat().st_size > 0)


def discover_run_dirs(parent: Path, run_prefix: str) -> list[Path]:
    if not parent.exists():
        return []
    return sorted(
        [path for path in parent.iterdir() if path.is_dir() and path.name.startswith(run_prefix)],
        key=run_sort_key,
    )


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def add_int_map(target: dict, source: dict) -> dict:
    for key, value in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)
    return target


def merge_source_stats(target: dict, source: dict) -> dict:
    for name, stats in source.items():
        if not isinstance(stats, dict):
            continue
        current = target.setdefault(
            str(name),
            {"converted_conversations": 0, "accepted": 0, "exhausted": False},
        )
        current["converted_conversations"] = int(current.get("converted_conversations", 0)) + int(
            stats.get("converted_conversations") or 0
        )
        current["accepted"] = int(current.get("accepted", 0)) + int(stats.get("accepted") or 0)
        current["exhausted"] = bool(current.get("exhausted", False)) or bool(stats.get("exhausted", False))
    return target


def merged_report_payload(target_root: Path, run_reports: list[dict], copied_stats: CopyStats) -> dict:
    report_dir = target_root / "reports"
    base = read_json(report_dir / "quality_report.json")
    target = dict(base.get("target") or {})
    clean_report = dict(base.get("clean_report") or {})
    source_stats = dict(base.get("source_stats") or {})

    language_bytes = dict(target.get("language_written_bytes") or {})
    language_records = dict(target.get("language_written_records") or {})
    categories = dict(clean_report.get("categories") or {})
    sources = dict(clean_report.get("sources") or {})
    rejected = dict(clean_report.get("rejected") or {})

    written_bytes = int(target.get("written_bytes") or 0)
    written_records = int(target.get("written_records") or 0)
    seen = int(clean_report.get("seen") or 0)
    accepted = int(clean_report.get("accepted") or 0)

    for report in run_reports:
        run_target = report.get("target") if isinstance(report.get("target"), dict) else {}
        run_clean = report.get("clean_report") if isinstance(report.get("clean_report"), dict) else {}
        written_bytes += int(run_target.get("written_bytes") or 0)
        written_records += int(run_target.get("written_records") or 0)
        seen += int(run_clean.get("seen") or 0)
        accepted += int(run_clean.get("accepted") or 0)
        add_int_map(language_bytes, run_target.get("language_written_bytes") or {})
        add_int_map(language_records, run_target.get("language_written_records") or {})
        add_int_map(categories, run_clean.get("categories") or {})
        add_int_map(sources, run_clean.get("sources") or {})
        add_int_map(rejected, run_clean.get("rejected") or {})
        merge_source_stats(source_stats, report.get("source_stats") or {})

    target_bytes = int(target.get("target_bytes") or int(float(target.get("target_gb") or 100.0) * 1024**3))
    target_gb = float(target.get("target_gb") or (target_bytes / 1024**3))
    target.update(
        {
            "target_gb": target_gb,
            "target_bytes": target_bytes,
            "written_bytes": written_bytes,
            "written_gib": written_bytes / 1024**3,
            "written_records": written_records,
            "shards": count_shards(target_root / "clean" / "shards"),
            "stopped_at_target": written_bytes >= target_bytes,
            "language_written_bytes": language_bytes,
            "language_written_gib": {lang: int(value) / 1024**3 for lang, value in sorted(language_bytes.items())},
            "language_written_records": language_records,
        }
    )
    clean_report.update(
        {
            "seen": seen,
            "accepted": accepted,
            "categories": categories,
            "sources": sources,
            "rejected": rejected,
        }
    )
    base.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "clean_report": clean_report,
            "source_stats": source_stats,
            "outputs": {
                "shard_dir": str(target_root / "clean" / "shards"),
                "accepted_sample_jsonl": str(target_root / "reports" / "accepted_sample.jsonl"),
            },
            "legacy_merge": {
                "copied_files": copied_stats.copied_files,
                "skipped_files": copied_stats.skipped_files,
                "copied_bytes": copied_stats.copied_bytes,
                "copied_gib": copied_stats.copied_bytes / 1024**3,
            },
        }
    )
    return base


def existing_copied_sources(manifest_path: Path) -> set[str]:
    manifest = read_json(manifest_path)
    copied: set[str] = set()
    for run in manifest.get("runs", []) if isinstance(manifest.get("runs"), list) else []:
        items = []
        if isinstance(run, dict):
            items.extend(run.get("copied", []) if isinstance(run.get("copied"), list) else [])
            items.extend(run.get("skipped", []) if isinstance(run.get("skipped"), list) else [])
        for item in items:
            source = item.get("source") if isinstance(item, dict) else None
            if source:
                copied.add(str(source))
    return copied


def copy_runs(*, target_root: Path, run_dirs: list[Path], force_lock: bool = False) -> CopyStats:
    shard_dir = target_root / "clean" / "shards"
    report_dir = target_root / "reports"
    shard_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    lock_path = report_dir / "legacy_run_merge.lock"
    if lock_path.exists() and not force_lock:
        raise RuntimeError(f"Merge lock exists: {lock_path}. Use --force-lock only after checking no merge is running.")
    lock_path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    manifest_path = report_dir / "legacy_run_merge_manifest.json"
    already_copied = existing_copied_sources(manifest_path)
    stats = CopyStats()
    run_reports: list[dict] = []
    next_index = next_shard_index(shard_dir)
    try:
        for run_dir in run_dirs:
            source_shard_dir = run_dir / "clean" / "shards"
            run_entry = {"run": str(run_dir), "copied": [], "skipped": []}
            report = read_json(run_dir / "reports" / "quality_report.json")
            if not source_shard_dir.exists():
                stats.runs.append(run_entry)
                print(f"run_missing {run_dir}", flush=True)
                continue
            shards = [path for path in sorted(source_shard_dir.glob("*.jsonl")) if path.is_file() and path.stat().st_size > 0]
            print(f"run_start {run_dir.name} files={len(shards)}", flush=True)
            copied_this_run = False
            for source in shards:
                source_key = str(source)
                if source_key in already_copied:
                    stats.skipped_files += 1
                    run_entry["skipped"].append({"source": source_key, "reason": "already_copied"})
                    continue
                dest = shard_dir / f"shard-{next_index:05d}.jsonl"
                while dest.exists():
                    next_index += 1
                    dest = shard_dir / f"shard-{next_index:05d}.jsonl"
                shutil.copy2(source, dest)
                copied_bytes = dest.stat().st_size
                stats.copied_files += 1
                stats.copied_bytes += copied_bytes
                run_entry["copied"].append(
                    {"source": source_key, "dest": str(dest), "bytes": copied_bytes}
                )
                copied_this_run = True
                already_copied.add(source_key)
                print(f"copied {source} -> {dest} size_mib={copied_bytes / 1024**2:.2f}", flush=True)
                next_index += 1
            if copied_this_run and report:
                run_reports.append(report)
            stats.runs.append(run_entry)

        if stats.copied_files > 0:
            report_payload = merged_report_payload(target_root, run_reports, stats)
            (report_dir / "quality_report.json").write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            print("no_new_shards_copied; quality_report unchanged", flush=True)
        if stats.copied_files > 0 or not manifest_path.exists():
            manifest = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "target_root": str(target_root),
                "copied_files": stats.copied_files,
                "skipped_files": stats.skipped_files,
                "copied_bytes": stats.copied_bytes,
                "runs": stats.runs,
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return stats
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy old dialogue corpus run shards into one resumable root.")
    parser.add_argument("--target-root", default="data/dialogue_corpus_100gb")
    parser.add_argument("--runs-parent", default=".")
    parser.add_argument("--run-prefix", default="dialogue_corpus_run")
    parser.add_argument("--force-lock", action="store_true")
    args = parser.parse_args()

    target_root = Path(args.target_root)
    run_dirs = discover_run_dirs(Path(args.runs_parent), args.run_prefix)
    stats = copy_runs(target_root=target_root, run_dirs=run_dirs, force_lock=args.force_lock)
    print(
        "merged "
        f"copied_files={stats.copied_files} skipped_files={stats.skipped_files} "
        f"copied_gib={stats.copied_bytes / 1024**3:.3f}"
    )


if __name__ == "__main__":
    main()
