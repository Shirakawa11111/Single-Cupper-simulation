"""
Summarize seed-robustness batch results with confidence intervals.

Inputs are batch directories produced by `repeat_crack_onset_seeds.py`.
Each batch directory is expected to contain:
- summary.json
- results.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from datetime import date, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any


BOOL_TARGETS: dict[str, bool] = {
    "seed_gate_pass": True,
    "passed": True,
    "checks_passed": True,
    "notch_onset": True,
    "notch_checks_ok": True,
    "negative_onset": False,
    "negative_checks_ok": True,
    "notch_onset_all": True,
    "notch_checks_all": True,
    "negative_onset_all_false": True,
    "negative_checks_all": True,
}


NUMERIC_FIELDS = (
    "onset_cases",
    "cases_total",
    "notch_onset_length",
    "notch_cycles_completed",
    "negative_cycles_completed",
    "notch_cycles_min",
    "negative_cycles_min",
)


def _default_out() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"seed_robustness_ci_{ts}" / "summary.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _parse_dirs(batch_dirs: str, batch_glob: str) -> list[Path]:
    seen: set[Path] = set()
    rows: list[Path] = []
    for chunk in batch_dirs.split(","):
        p = chunk.strip()
        if not p:
            continue
        cand = Path(p)
        if cand not in seen:
            rows.append(cand)
            seen.add(cand)
    if batch_glob.strip():
        for p in sorted(glob.glob(batch_glob.strip())):
            cand = Path(p)
            if cand not in seen:
                rows.append(cand)
                seen.add(cand)
    if not rows:
        raise ValueError("No batch dirs found from --batch-dirs/--batch-glob.")
    return rows


def _parse_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    text = str(val).strip().lower()
    if not text:
        return None
    if text in {"true", "t", "yes", "y"}:
        return True
    if text in {"false", "f", "no", "n"}:
        return False
    return None


def _parse_int(val: Any) -> int | None:
    try:
        return int(str(val).strip())
    except (TypeError, ValueError):
        return None


def _parse_float(val: Any) -> float | None:
    try:
        x = float(str(val).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _z_value(confidence: float) -> float:
    alpha = (1.0 - confidence) / 2.0
    return NormalDist().inv_cdf(1.0 - alpha)


def _wilson_interval(success: int, total: int, confidence: float) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    z = _z_value(confidence)
    n = float(total)
    p = float(success) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    radius = (z / denom) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return (max(0.0, center - radius), min(1.0, center + radius))


def _mean_ci(values: list[float], confidence: float) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean = sum(values) / float(n)
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    var = sum((x - mean) ** 2 for x in values) / float(n - 1)
    std = math.sqrt(max(var, 0.0))
    z = _z_value(confidence)
    half = z * std / math.sqrt(float(n))
    return {"n": n, "mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def _fmt(x: Any, digits: int = 6) -> str:
    if isinstance(x, (int, float)):
        if isinstance(x, float) and not math.isfinite(x):
            return "nan"
        return f"{float(x):.{digits}g}"
    return str(x)


def _build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Seed Robustness CI Summary ({payload['generated_at'][:10]})")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Batch count: `{payload['batch_count']}`")
    lines.append(f"- Unique seeds: `{payload['seed_unique_count']}`")
    lines.append(f"- Rows used: `{payload['row_count_after_dedup']}`")
    lines.append(f"- Confidence: `{payload['confidence']}`")
    lines.append(f"- `all_seed_gate_passed`: `{payload['all_seed_gate_passed']}`")
    dup = payload.get("duplicate_seeds", [])
    lines.append(f"- Duplicate seeds: `{dup if dup else 'none'}`")
    lines.append("")

    lines.append("## Binary Metrics (Wilson CI)")
    lines.append("")
    lines.append("| Metric | Target | Success/Total | Rate | CI Low | CI High |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in sorted(payload.get("binary_metrics", {}).keys()):
        row = payload["binary_metrics"][key]
        lines.append(
            f"| {key} | {row['target']} | {row['success']}/{row['n']} | "
            f"{_fmt(row['rate'])} | {_fmt(row['ci_low'])} | {_fmt(row['ci_high'])} |"
        )

    lines.append("")
    lines.append("## Numeric Metrics (Mean CI)")
    lines.append("")
    lines.append("| Metric | N | Mean | Std | CI Low | CI High |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in sorted(payload.get("numeric_metrics", {}).keys()):
        row = payload["numeric_metrics"][key]
        lines.append(
            f"| {key} | {row['n']} | {_fmt(row['mean'])} | {_fmt(row['std'])} | "
            f"{_fmt(row['ci_low'])} | {_fmt(row['ci_high'])} |"
        )

    lines.append("")
    lines.append("## Batch Inputs")
    for row in payload.get("batches", []):
        lines.append(
            f"- `{row['batch_dir']}`: seed_gate `{row['seed_gate_pass_count']}/{row['seed_gate_total']}`, "
            f"`all_seed_gate_passed={row['all_seed_gate_passed']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize seed robustness batches with CI.")
    parser.add_argument("--batch-dirs", type=str, default="", help="Comma-separated batch directories.")
    parser.add_argument("--batch-glob", type=str, default="", help="Glob pattern for batch directories.")
    parser.add_argument("--out", type=Path, default=None, help="Output summary JSON path.")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Optional markdown report output path.")
    parser.add_argument("--aggregate-csv-out", type=Path, default=None, help="Optional merged per-seed CSV path.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level in (0,1).")
    args = parser.parse_args()

    confidence = float(args.confidence)
    if not (0.0 < confidence < 1.0):
        raise ValueError("--confidence must be in (0, 1).")

    dirs = _parse_dirs(args.batch_dirs, args.batch_glob)
    out_path = args.out or _default_out()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batch_rows: list[dict[str, Any]] = []
    all_rows_raw: list[dict[str, Any]] = []
    for d in dirs:
        summary_path = d / "summary.json"
        csv_path = d / "results.csv"
        s = _read_json(summary_path) or {}
        rows = _read_csv(csv_path)
        for r in rows:
            rr = dict(r)
            rr["batch_dir"] = str(d)
            rr["summary_json"] = str(summary_path)
            all_rows_raw.append(rr)
        batch_rows.append(
            {
                "batch_dir": str(d),
                "summary_json": str(summary_path),
                "results_csv": str(csv_path),
                "seed_gate_pass_count": int(s.get("seed_gate_pass_count", 0)),
                "seed_gate_total": int(s.get("seed_gate_total", 0)),
                "all_seed_gate_passed": bool(s.get("all_seed_gate_passed", False)),
                "case_mode": s.get("case_mode"),
            }
        )

    # Deduplicate by seed (keep first occurrence); rows without seed are kept.
    dedup_rows: list[dict[str, Any]] = []
    seen_seed: set[int] = set()
    dup_seed: set[int] = set()
    for r in all_rows_raw:
        seed = _parse_int(r.get("seed"))
        if seed is None:
            dedup_rows.append(r)
            continue
        if seed in seen_seed:
            dup_seed.add(seed)
            continue
        seen_seed.add(seed)
        dedup_rows.append(r)

    if args.aggregate_csv_out is not None:
        args.aggregate_csv_out.parent.mkdir(parents=True, exist_ok=True)
        fields: list[str] = []
        for row in dedup_rows:
            for key in row.keys():
                if key not in fields:
                    fields.append(key)
        with args.aggregate_csv_out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for row in dedup_rows:
                w.writerow(row)

    # Binary metrics
    binary_metrics: dict[str, Any] = {}
    for field, target in BOOL_TARGETS.items():
        parsed: list[bool] = []
        for row in dedup_rows:
            b = _parse_bool(row.get(field))
            if b is not None:
                parsed.append(b)
        if not parsed:
            continue
        success = sum(1 for b in parsed if b == target)
        n = len(parsed)
        ci_low, ci_high = _wilson_interval(success, n, confidence)
        binary_metrics[field] = {
            "target": target,
            "n": n,
            "success": success,
            "rate": float(success) / float(n),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    # Numeric metrics
    numeric_metrics: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        vals: list[float] = []
        for row in dedup_rows:
            x = _parse_float(row.get(field))
            if x is not None:
                vals.append(x)
        if not vals:
            continue
        numeric_metrics[field] = _mean_ci(vals, confidence)

    seeds_sorted = sorted(seen_seed)
    gate_metric = binary_metrics.get("seed_gate_pass")
    all_seed_gate_passed = bool(gate_metric and gate_metric.get("success") == gate_metric.get("n"))

    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "confidence": confidence,
        "batch_count": len(batch_rows),
        "batches": batch_rows,
        "seed_unique_count": len(seeds_sorted),
        "seeds": seeds_sorted,
        "duplicate_seeds": sorted(dup_seed),
        "row_count_raw": len(all_rows_raw),
        "row_count_after_dedup": len(dedup_rows),
        "all_seed_gate_passed": all_seed_gate_passed,
        "binary_metrics": binary_metrics,
        "numeric_metrics": numeric_metrics,
        "aggregate_csv_out": str(args.aggregate_csv_out) if args.aggregate_csv_out else None,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_build_markdown(payload), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
