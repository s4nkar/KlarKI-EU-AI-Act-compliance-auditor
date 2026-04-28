"""
Evaluation 7 — Regression Runner.

Runs all six evaluation modules, compares each key metric against a saved
baseline, and flags any metric that regressed by more than the allowed
tolerance.

Workflow:
  1. First run:  no baseline exists → all evals run, results saved as baseline.
  2. Subsequent runs: results compared to baseline.
     • Regression = metric dropped by > tolerance (default 2 pp for rates).
     • Improvement = metric improved — baseline updated automatically.

Usage:
    python tests/evaluation/run_regression.py            # compare to baseline
    python tests/evaluation/run_regression.py --reset    # overwrite baseline
    python tests/evaluation/run_regression.py --offline  # skip live-service evals
    python tests/evaluation/run_regression.py --verbose  # full per-module output
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

EVAL_DIR      = Path(__file__).parent
BASELINE_PATH = EVAL_DIR / "baselines" / "baseline_results.json"
RESULTS_DIR   = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
BASELINE_PATH.parent.mkdir(exist_ok=True)

# Metrics to track and their regression tolerances (absolute drop allowed)
TRACKED_METRICS: dict[str, tuple[str, float]] = {
    # (result_key_path,  tolerance)
    "classifier.macro_f1":          ("classifier.macro_f1",          0.02),
    "classifier.accuracy":          ("classifier.accuracy",           0.02),
    "rag.recall@3":                 ("rag_retrieval.recall@3",        0.05),
    "rag.recall@5":                 ("rag_retrieval.recall@5",        0.05),
    "rag.mrr":                      ("rag_retrieval.mrr",             0.05),
    "adversarial.accuracy":         ("adversarial.adversarial_accuracy", 0.05),
    "bert_consistency":             ("consistency.bert.consistency_rate", 0.0),
    "hallucination.citation_rate":  ("hallucination.citation_rate",   0.05),
    "pipeline.checks_passed_ratio": ("pipeline.checks_passed_ratio",  0.10),
    # Phase 3 metrics
    "actor.accuracy":               ("actor.accuracy",                0.05),
    "actor.macro_f1":               ("actor.macro_f1",                0.05),
    "applicability.accuracy":       ("applicability.accuracy",        0.05),
    "applicability.prohibited_recall": ("applicability.by_outcome.prohibited.recall", 0.05),
    "applicability.high_risk_recall":  ("applicability.by_outcome.high_risk.recall",  0.05),
    "evidence_mapper.tpr":          ("evidence_mapper.tpr",           0.05),
    "evidence_mapper.tnr":          ("evidence_mapper.tnr",           0.05),
    # Specialist ML classifier evals (skip gracefully if models not trained)
    "risk.accuracy":                ("risk.accuracy",                 0.05),
    "risk.recall":                  ("risk.recall",                   0.05),
    "risk.tnr":                     ("risk.tnr",                      0.05),
    "prohibited.accuracy":          ("prohibited.accuracy",           0.05),
    "prohibited.recall":            ("prohibited.recall",             0.05),
    "prohibited.tnr":               ("prohibited.tnr",                0.05),
}

# ANSI colour helpers (gracefully degrade on Windows without ANSI support)
def _green(s: str)  -> str: return f"\033[92m{s}\033[0m"
def _red(s: str)    -> str: return f"\033[91m{s}\033[0m"
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m"
def _bold(s: str)   -> str: return f"\033[1m{s}\033[0m"

try:
    import sys as _sys
    if _sys.platform == "win32":
        import os
        os.system("")   # enable ANSI escape codes on Windows terminal
except Exception:
    pass


# ── metric extraction ──────────────────────────────────────────────────────

def _extract(results: dict[str, Any], dotted_key: str) -> float | None:
    """Traverse nested dict using dot-separated key.  Returns None if missing."""
    parts  = dotted_key.split(".")
    cursor: Any = results
    for p in parts:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(p)
    return float(cursor) if cursor is not None else None


def _collect_all_metrics(all_results: dict[str, Any]) -> dict[str, float]:
    """Flatten all tracked metrics from the run results into a single dict."""
    out: dict[str, float] = {}
    for friendly_name, (path, _tol) in TRACKED_METRICS.items():
        val = _extract(all_results, path)
        if val is not None:
            out[friendly_name] = val
    # Compute pipeline check ratio on the fly
    p = all_results.get("pipeline", {})
    if p.get("checks_total", 0) > 0:
        out["pipeline.checks_passed_ratio"] = (
            p["checks_passed"] / p["checks_total"]
        )
    return out


# ── regression comparison ─────────────────────────────────────────────────

def compare_to_baseline(
    current: dict[str, float],
    baseline: dict[str, float],
) -> list[dict]:
    """Return list of regression/improvement events."""
    events: list[dict] = []
    for name, value in current.items():
        if name not in baseline:
            continue
        _, tolerance = TRACKED_METRICS.get(name, ("", 0.02))
        delta = value - baseline[name]
        if delta < -tolerance:
            events.append({
                "metric":   name,
                "baseline": baseline[name],
                "current":  value,
                "delta":    round(delta, 4),
                "type":     "regression",
            })
        elif delta > 0.001:
            events.append({
                "metric":   name,
                "baseline": baseline[name],
                "current":  value,
                "delta":    round(delta, 4),
                "type":     "improvement",
            })
    return events


# ── individual eval runners ────────────────────────────────────────────────

def _run_eval(name: str, fn, verbose: bool) -> dict:
    """Run a single eval module, catch exceptions gracefully."""
    print(f"  Running {name} …", end=" ", flush=True)
    try:
        result = fn()
        status = result.get("status", "unknown")
        icon   = "✓" if status == "pass" else ("~" if status in ("warn", "skip") else "✗")
        print(icon)
        if verbose and status not in ("skip",):
            _print_inline_summary(name, result)
        return result
    except Exception as exc:
        print("✗ (exception)")
        if verbose:
            import traceback
            traceback.print_exc()
        return {"status": "error", "error": str(exc)}


def _print_inline_summary(name: str, r: dict) -> None:
    """Print a one-line metric summary for a completed eval."""
    summaries = {
        "classifier":      lambda r: f"    macro_f1={r.get('macro_f1',0)*100:.1f}%  accuracy={r.get('accuracy',0)*100:.1f}%",
        "rag":             lambda r: f"    recall@1={r.get('recall@1',0)*100:.1f}%  recall@3={r.get('recall@3',0)*100:.1f}%  MRR={r.get('mrr',0):.3f}",
        "adversarial":     lambda r: f"    adversarial_accuracy={r.get('adversarial_accuracy',0)*100:.1f}%",
        "hallucination":   lambda r: f"    citation_rate={r.get('citation_rate',0)*100:.1f}%  violations={r.get('total_violations',0)}",
        "pipeline":        lambda r: f"    overall_score={r.get('overall_score',0):.0f}  checks={r.get('checks_passed',0)}/{r.get('checks_total',0)}",
        "actor":           lambda r: f"    accuracy={r.get('accuracy',0)*100:.1f}%  macro_f1={r.get('macro_f1',0)*100:.1f}%",
        "applicability":   lambda r: f"    accuracy={r.get('accuracy',0)*100:.1f}%  prohibited_recall={r.get('by_outcome',{}).get('prohibited',{}).get('recall',0)*100:.1f}%",
        "evidence_mapper": lambda r: f"    tpr={r.get('tpr',0)*100:.1f}%  tnr={r.get('tnr',0)*100:.1f}%  balanced_acc={r.get('balanced_accuracy',0)*100:.1f}%",
        "risk":            lambda r: f"    accuracy={r.get('accuracy',0)*100:.1f}%  recall={r.get('recall',0)*100:.1f}%  tnr={r.get('tnr',0)*100:.1f}%",
        "prohibited":      lambda r: f"    accuracy={r.get('accuracy',0)*100:.1f}%  recall={r.get('recall',0)*100:.1f}%  tnr={r.get('tnr',0)*100:.1f}%",
    }
    fn = summaries.get(name)
    if fn:
        try:
            print(fn(r))
        except Exception:
            pass


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="KlarKI regression test runner")
    parser.add_argument("--reset",   action="store_true", help="Overwrite saved baseline with current results")
    parser.add_argument("--offline", action="store_true", help="Skip evals that require live services (RAG, pipeline, hallucination, consistency)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Lazy imports so each module's own sys.path setup runs
    from eval_classifier      import run as run_classifier
    from eval_rag             import run as run_rag
    from eval_pipeline        import run as run_pipeline
    from eval_hallucination   import run as run_hallucination
    from eval_adversarial     import run as run_adversarial
    from eval_consistency     import run as run_consistency
    from eval_actor           import run as run_actor
    from eval_applicability   import run as run_applicability
    from eval_evidence_mapper import run as run_evidence_mapper
    from eval_risk            import run as run_risk
    from eval_prohibited      import run as run_prohibited

    print(_bold("\n═══════════════════════════════════════════════════"))
    print(_bold("  KlarKI Evaluation & Regression Suite"))
    print(_bold(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    print(_bold("═══════════════════════════════════════════════════\n"))

    # ── run all evals ──────────────────────────────────────────────────────
    all_results: dict[str, Any] = {}

    # Evals that only need the BERT model (no live services)
    all_results["classifier"] = _run_eval(
        "classifier", lambda: run_classifier(verbose=args.verbose), args.verbose,
    )
    all_results["adversarial"] = _run_eval(
        "adversarial", lambda: run_adversarial(verbose=args.verbose), args.verbose,
    )

    # Phase 3 evals — deterministic, no live services required
    all_results["actor"] = _run_eval(
        "actor", lambda: run_actor(verbose=args.verbose), args.verbose,
    )
    all_results["applicability"] = _run_eval(
        "applicability", lambda: run_applicability(verbose=args.verbose), args.verbose,
    )
    all_results["evidence_mapper"] = _run_eval(
        "evidence_mapper", lambda: run_evidence_mapper(verbose=args.verbose), args.verbose,
    )
    # Specialist ML classifiers — skip gracefully when models not yet trained
    all_results["risk"] = _run_eval(
        "risk", lambda: run_risk(verbose=args.verbose), args.verbose,
    )
    all_results["prohibited"] = _run_eval(
        "prohibited", lambda: run_prohibited(verbose=args.verbose), args.verbose,
    )

    # Evals that need live services (skipped in --offline mode)
    if args.offline:
        print("  [OFFLINE] Skipping RAG, pipeline, hallucination, consistency evals")
        for key in ("rag_retrieval", "pipeline", "hallucination", "consistency"):
            all_results[key] = {"status": "skip", "reason": "--offline flag set"}
    else:
        all_results["rag_retrieval"] = _run_eval(
            "rag", lambda: run_rag(top_k=5, verbose=args.verbose), args.verbose,
        )
        all_results["pipeline"] = _run_eval(
            "pipeline", lambda: run_pipeline(verbose=args.verbose), args.verbose,
        )
        all_results["hallucination"] = _run_eval(
            "hallucination", lambda: run_hallucination(verbose=args.verbose), args.verbose,
        )
        all_results["consistency"] = _run_eval(
            "consistency", lambda: run_consistency(n_runs=5, verbose=args.verbose), args.verbose,
        )

    # ── collect metrics ────────────────────────────────────────────────────
    current_metrics = _collect_all_metrics(all_results)

    # ── regression comparison ──────────────────────────────────────────────
    regressions: list[dict] = []
    improvements: list[dict] = []

    if args.reset or not BASELINE_PATH.exists():
        print(f"\n  {'Saving' if args.reset else 'No baseline found — creating'} baseline …")
        _save_baseline(current_metrics)
    else:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8")).get("metrics", {})
        events   = compare_to_baseline(current_metrics, baseline)
        regressions  = [e for e in events if e["type"] == "regression"]
        improvements = [e for e in events if e["type"] == "improvement"]

        if improvements:
            # Auto-update baseline for improved metrics
            for imp in improvements:
                baseline[imp["metric"]] = imp["current"]
            _save_baseline(baseline)

    # ── print summary ──────────────────────────────────────────────────────
    print(_bold("\n── Metric Summary ─────────────────────────────────"))
    _print_metric_table(current_metrics, regressions)

    if regressions:
        print(_bold(_red(f"\n── {len(regressions)} Regression(s) Detected ───────────────────")))
        for reg in regressions:
            delta_str = f"{reg['delta']*100:+.1f}pp"
            print(_red(
                f"  ✗ {reg['metric']:<40} "
                f"baseline={reg['baseline']*100:.1f}%  "
                f"current={reg['current']*100:.1f}%  "
                f"({delta_str})"
            ))
        print()
        print(_red("  ACTION REQUIRED: Review the pipeline changes that caused these regressions."))
        print(_red("  Run ./run.sh setup or retrain the relevant model component."))
    else:
        print(_green("\n  ✓ No regressions detected."))

    if improvements:
        print(_green(f"\n  ↑ {len(improvements)} improvement(s) — baseline updated automatically."))
        for imp in improvements:
            print(_green(f"    {imp['metric']}: {imp['baseline']*100:.1f}% → {imp['current']*100:.1f}%"))

    # ── save full run results ──────────────────────────────────────────────
    run_record = {
        "timestamp":       datetime.datetime.now().isoformat(),
        "metrics":         current_metrics,
        "regressions":     regressions,
        "improvements":    improvements,
        "eval_statuses":   {k: v.get("status") for k, v in all_results.items()},
    }
    run_log_path = RESULTS_DIR / f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    run_log_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(f"\n  Full results saved to {run_log_path.relative_to(Path.cwd()) if Path.cwd() in run_log_path.parents else run_log_path}")

    return 1 if regressions else 0


def _save_baseline(metrics: dict[str, float]) -> None:
    record = {
        "saved_at": datetime.datetime.now().isoformat(),
        "metrics":  metrics,
    }
    BASELINE_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  Baseline saved → {BASELINE_PATH}")


def _print_metric_table(metrics: dict[str, float], regressions: list[dict]) -> None:
    regressed_names = {r["metric"] for r in regressions}
    for name, value in sorted(metrics.items()):
        pct_str = f"{value*100:.1f}%"
        if name in regressed_names:
            print(_red(f"  ✗ {name:<40} {pct_str}"))
        elif value >= 0.90:
            print(_green(f"  ✓ {name:<40} {pct_str}"))
        elif value >= 0.75:
            print(_yellow(f"  ~ {name:<40} {pct_str}"))
        else:
            print(_red(f"  ✗ {name:<40} {pct_str}"))


# ── pytest integration ─────────────────────────────────────────────────────

def test_no_regressions_offline() -> None:
    """pytest: Run offline-only evals and assert no regressions vs baseline."""
    import subprocess, sys as _sys
    result = subprocess.run(
        [_sys.executable, __file__, "--offline"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent,
    )
    assert result.returncode == 0, (
        "Regression detected in offline evals.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


if __name__ == "__main__":
    sys.exit(main())
