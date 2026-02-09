"""
Run seed robustness in batches (default 20 seeds) and build CI summary.

Template goal:
- execute seed-repeat gate in manageable batches
- aggregate all batch outputs
- produce confidence-interval summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any


def _default_out_root() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"seed_robustness_20_{ts}"


def _parse_seed_list(text: str) -> list[int]:
    vals: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals.append(int(chunk))
    if not vals:
        raise ValueError("Expected non-empty --seeds list.")
    return vals


def _chunk(vals: list[int], size: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("--batch-size must be > 0.")
    out: list[list[int]] = []
    for i in range(0, len(vals), size):
        out.append(vals[i : i + size])
    return out


def _run(name: str, cmd: list[str], logs_dir: Path) -> dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = perf_counter() - t0
    stdout_path = logs_dir / f"{name}.stdout"
    stderr_path = logs_dir / f"{name}.stderr"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {
        "name": name,
        "command": cmd,
        "returncode": int(proc.returncode),
        "duration_s": dt,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "passed": proc.returncode == 0,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run batched seed robustness template (default 20 seeds).")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--base-config", type=Path, default=Path("sim/configs/crack_onset_scan.yaml"))
    parser.add_argument("--case-mode", type=str, default="full", choices=("pair", "full"))
    parser.add_argument("--notch-case", type=str, default="control_notch_mild")
    parser.add_argument("--negative-case", type=str, default="no_notch_control")
    parser.add_argument("--seed-start", type=int, default=41)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated explicit seeds; overrides start/count.")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--auto-output", action="store_true")
    parser.add_argument("--max-runtime-warnings", type=int, default=50)
    parser.add_argument("--max-mechanical-not-accepted-steps", type=int, default=160)
    parser.add_argument("--max-crack-cg-nonconverged-steps", type=int, default=20)
    parser.add_argument("--max-nonfinite-count", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    if args.seeds.strip():
        seeds = _parse_seed_list(args.seeds)
    else:
        if args.seed_count <= 0:
            raise ValueError("--seed-count must be > 0.")
        seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    batches = _chunk(seeds, args.batch_size)

    out_root = args.out_root or _default_out_root()
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for idx, seed_batch in enumerate(batches, start=1):
        tag = f"seed_batch_{idx:02d}"
        batch_out = out_root / tag
        cmd = [
            args.python,
            "sim/tests/repeat_crack_onset_seeds.py",
            "--base-config",
            str(args.base_config),
            "--seeds",
            ",".join(str(v) for v in seed_batch),
            "--case-mode",
            args.case_mode,
            "--notch-case",
            args.notch_case,
            "--negative-case",
            args.negative_case,
            "--out-root",
            str(batch_out),
            "--max-runtime-warnings",
            str(args.max_runtime_warnings),
            "--max-mechanical-not-accepted-steps",
            str(args.max_mechanical_not_accepted_steps),
            "--max-crack-cg-nonconverged-steps",
            str(args.max_crack_cg_nonconverged_steps),
            "--max-nonfinite-count",
            str(args.max_nonfinite_count),
        ]
        if args.auto_output:
            cmd.append("--auto-output")
        rec = _run(tag, cmd, logs_dir)
        rec["seed_batch"] = seed_batch
        rec["summary_json"] = str(batch_out / "summary.json")
        rec["summary"] = _read_json(batch_out / "summary.json")
        rec["passed"] = bool(
            rec["passed"] and isinstance(rec["summary"], dict) and rec["summary"].get("all_seed_gate_passed", False)
        )
        records.append(rec)

    ci_json = out_root / "ci_summary.json"
    ci_md = out_root / "ci_summary.md"
    ci_csv = out_root / "ci_aggregate.csv"
    ci_cmd = [
        args.python,
        "sim/tests/summarize_seed_robustness_ci.py",
        "--batch-glob",
        str(out_root / "seed_batch_*"),
        "--out",
        str(ci_json),
        "--markdown-out",
        str(ci_md),
        "--aggregate-csv-out",
        str(ci_csv),
        "--confidence",
        str(args.confidence),
    ]
    ci_rec = _run("ci_summary", ci_cmd, logs_dir)
    ci_payload = _read_json(ci_json)
    ci_rec["summary_json"] = str(ci_json)
    ci_rec["summary"] = ci_payload
    ci_rec["passed"] = bool(ci_rec["passed"] and isinstance(ci_payload, dict))
    records.append(ci_rec)

    bundle = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "base_config": str(args.base_config),
        "case_mode": args.case_mode,
        "seeds": seeds,
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "confidence": args.confidence,
        "ci_summary_json": str(ci_json),
        "ci_summary_markdown": str(ci_md),
        "ci_aggregate_csv": str(ci_csv),
        "tasks": records,
        "passed": all(bool(r.get("passed", False)) for r in records),
    }
    summary_path = out_root / "bundle_summary.json"
    summary_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(summary_path)
    print(json.dumps(bundle, indent=2))
    return 0 if bundle["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
