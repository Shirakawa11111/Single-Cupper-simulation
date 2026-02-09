#!/usr/bin/env bash
set -euo pipefail

python3 - "$@" <<'PY'
from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cleanup_artifacts.sh",
        description="Cleanup transient/duplicate artifacts.",
    )
    parser.add_argument("--apply", action="store_true", help="Execute deletion (default: dry-run).")
    parser.add_argument(
        "--retention-days",
        type=int,
        default=0,
        help="Delete untracked dated dirs older than N days (default: disabled).",
    )
    parser.add_argument(
        "--keep-date",
        action="append",
        default=[],
        help="Keep dated directory (YYYY-MM-DD) even when older than retention.",
    )
    parser.add_argument(
        "--include-regress-timestamp-dups",
        action="store_true",
        help="Also deduplicate *_HHMMSS dirs under sim/tests/regress_runs.",
    )
    parser.add_argument(
        "--force-referenced",
        action="store_true",
        help="Allow deleting paths referenced in README/HANDOFF/docs.",
    )
    return parser.parse_args(argv)


def git_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return Path(out)
    except subprocess.SubprocessError:
        pass
    return Path.cwd()


def tracked_paths(root: Path) -> list[str]:
    try:
        out = subprocess.check_output(["git", "ls-files"], text=True, cwd=root)
    except subprocess.SubprocessError:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_tracked(rel: str, tracked_set: set[str], tracked_list: list[str]) -> bool:
    if rel in tracked_set:
        return True
    prefix = rel.rstrip("/") + "/"
    for item in tracked_list:
        if item.startswith(prefix):
            return True
    return False


def has_git_in_path(path: Path) -> bool:
    return ".git" in path.parts


def in_virtual_env(path: Path) -> bool:
    for part in path.parts:
        if part == "venv" or part.startswith(".venv"):
            return True
    return False


def is_referenced(rel: str, targets: list[Path], force_referenced: bool) -> bool:
    if force_referenced or not targets:
        return False
    cmd = ["rg", "-F", "-q", "--", rel] + [str(p) for p in targets]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return res.returncode == 0


def queue_delete(
    path: Path,
    reason: str,
    root: Path,
    tracked_set: set[str],
    tracked_list: list[str],
    ref_targets: list[Path],
    force_referenced: bool,
    candidates: dict[str, set[str]],
) -> None:
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        rel = path.as_posix().lstrip("./")
    rel = rel.strip("/")
    if not rel or rel in {".", ".."}:
        return
    abs_path = root / rel
    if not abs_path.exists():
        return
    if is_tracked(rel, tracked_set, tracked_list):
        return
    if is_referenced(rel, ref_targets, force_referenced):
        return
    if rel not in candidates:
        candidates[rel] = set()
    candidates[rel].add(reason)


def collect_cache_candidates(
    root: Path,
    tracked_set: set[str],
    tracked_list: list[str],
    ref_targets: list[Path],
    force_referenced: bool,
    candidates: dict[str, set[str]],
) -> None:
    for p in root.rglob("__pycache__"):
        if p.is_dir() and not has_git_in_path(p) and not in_virtual_env(p):
            queue_delete(
                p, "python_cache", root, tracked_set, tracked_list, ref_targets, force_referenced, candidates
            )
    for pattern, reason in [(".DS_Store", "system_cache"), ("*.pyc", "system_cache"), ("*.pyo", "system_cache")]:
        for p in root.rglob(pattern):
            if p.is_file() and not has_git_in_path(p) and not in_virtual_env(p):
                queue_delete(
                    p, reason, root, tracked_set, tracked_list, ref_targets, force_referenced, candidates
                )
    for p in root.rglob(".pytest_cache"):
        if p.is_dir() and not has_git_in_path(p) and not in_virtual_env(p):
            queue_delete(
                p, "pytest_cache", root, tracked_set, tracked_list, ref_targets, force_referenced, candidates
            )


def collect_transient_files(
    root: Path,
    tracked_set: set[str],
    tracked_list: list[str],
    ref_targets: list[Path],
    force_referenced: bool,
    candidates: dict[str, set[str]],
) -> None:
    patterns = [
        "sim/tests/virtual_cycle*.csv",
        "sim/tests/virtual_cycle*.png",
        "sim/tests/*_stress_strain.csv",
        "sim/tests/*.stdout",
        "sim/tests/*.stderr",
    ]
    for pattern in patterns:
        for p in root.glob(pattern):
            if p.exists():
                queue_delete(
                    p, "transient_file", root, tracked_set, tracked_list, ref_targets, force_referenced, candidates
                )


def collect_superseded_pairs(
    root: Path,
    bucket: str,
    tracked_set: set[str],
    tracked_list: list[str],
    ref_targets: list[Path],
    force_referenced: bool,
    candidates: dict[str, set[str]],
) -> None:
    bucket_root = root / bucket
    if not bucket_root.is_dir():
        return
    for date_dir in bucket_root.glob("????-??-??"):
        if not date_dir.is_dir():
            continue
        for rerun in date_dir.glob("*_rerun_[0-9][0-9][0-9][0-9][0-9][0-9]"):
            if not rerun.is_dir():
                continue
            base = Path(str(rerun).rsplit("_rerun_", 1)[0])
            if base.is_dir():
                queue_delete(
                    base,
                    "superseded_by_rerun",
                    root,
                    tracked_set,
                    tracked_list,
                    ref_targets,
                    force_referenced,
                    candidates,
                )
        for fixed in date_dir.glob("*_verify_fix"):
            if not fixed.is_dir():
                continue
            verify = Path(str(fixed)[: -len("_fix")])
            if verify.is_dir():
                queue_delete(
                    verify,
                    "superseded_by_verify_fix",
                    root,
                    tracked_set,
                    tracked_list,
                    ref_targets,
                    force_referenced,
                    candidates,
                )


def collect_timestamp_duplicates(
    root: Path,
    bucket: str,
    tracked_set: set[str],
    tracked_list: list[str],
    ref_targets: list[Path],
    force_referenced: bool,
    candidates: dict[str, set[str]],
) -> None:
    bucket_root = root / bucket
    if not bucket_root.is_dir():
        return
    pat = re.compile(r"^(.+)_([0-9]{6})$")
    for date_dir in bucket_root.glob("????-??-??"):
        if not date_dir.is_dir():
            continue
        group: dict[str, list[tuple[str, Path]]] = {}
        for child in date_dir.iterdir():
            if not child.is_dir():
                continue
            m = pat.match(child.name)
            if not m:
                continue
            prefix, stamp = m.group(1), m.group(2)
            group.setdefault(prefix, []).append((stamp, child))
        for items in group.values():
            if len(items) <= 1:
                continue
            items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
            for _stamp, old_path in items_sorted[1:]:
                queue_delete(
                    old_path,
                    "older_timestamp_duplicate",
                    root,
                    tracked_set,
                    tracked_list,
                    ref_targets,
                    force_referenced,
                    candidates,
                )


def collect_old_date_dirs(
    root: Path,
    bucket: str,
    retention_days: int,
    keep_dates: set[str],
    tracked_set: set[str],
    tracked_list: list[str],
    ref_targets: list[Path],
    force_referenced: bool,
    candidates: dict[str, set[str]],
) -> None:
    if retention_days <= 0:
        return
    bucket_root = root / bucket
    if not bucket_root.is_dir():
        return
    cutoff = (dt.date.today() - dt.timedelta(days=retention_days)).isoformat()
    for date_dir in bucket_root.glob("????-??-??"):
        if not date_dir.is_dir():
            continue
        name = date_dir.name
        if name in keep_dates:
            continue
        if name < cutoff:
            queue_delete(
                date_dir, "old_date_dir", root, tracked_set, tracked_list, ref_targets, force_referenced, candidates
            )


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=False)
    else:
        path.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = git_root()
    tracked = tracked_paths(root)
    tracked_set = set(tracked)

    ref_targets: list[Path] = []
    for p in [
        root / "README.md",
        root / "HANDOFF.md",
        root / "HANDOFF_RELEASE_PACK_V2.md",
        root / "WEEKLY_CHECKLIST.md",
        root / "WEEK9_CHECKLIST.md",
        root / "docs",
    ]:
        if p.exists():
            ref_targets.append(p)

    candidates: dict[str, set[str]] = {}
    keep_dates = set(args.keep_date or [])

    collect_cache_candidates(
        root, tracked_set, tracked, ref_targets, args.force_referenced, candidates
    )
    collect_transient_files(
        root, tracked_set, tracked, ref_targets, args.force_referenced, candidates
    )
    collect_superseded_pairs(
        root, "sim/tests/runs", tracked_set, tracked, ref_targets, args.force_referenced, candidates
    )
    collect_superseded_pairs(
        root, "sim/tests/regress_runs", tracked_set, tracked, ref_targets, args.force_referenced, candidates
    )
    collect_timestamp_duplicates(
        root, "sim/tests/runs", tracked_set, tracked, ref_targets, args.force_referenced, candidates
    )
    if args.include_regress_timestamp_dups:
        collect_timestamp_duplicates(
            root, "sim/tests/regress_runs", tracked_set, tracked, ref_targets, args.force_referenced, candidates
        )
    collect_old_date_dirs(
        root,
        "sim/tests/runs",
        args.retention_days,
        keep_dates,
        tracked_set,
        tracked,
        ref_targets,
        args.force_referenced,
        candidates,
    )
    collect_old_date_dirs(
        root,
        "sim/tests/regress_runs",
        args.retention_days,
        keep_dates,
        tracked_set,
        tracked,
        ref_targets,
        args.force_referenced,
        candidates,
    )

    if not candidates:
        print("No cleanup candidates.")
        return 0

    ordered = sorted(candidates.keys())
    for rel in ordered:
        reason = ",".join(sorted(candidates[rel]))
        abs_path = root / rel
        if args.apply:
            if abs_path.exists():
                delete_path(abs_path)
            print(f"[deleted] {rel} :: {reason}")
        else:
            print(f"[dry-run] {rel} :: {reason}")
    print(f"Total candidates: {len(ordered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
PY
