"""Shared gold-set evaluation for the classifier training scripts.

The held-out validation F1 that training prints is computed on the *same*
synthetic distribution the model trained on, so it can read ~99% while the
model fails on realistic policy prose. The hand-curated gold datasets in
tests/evaluation/datasets/ are the honest out-of-distribution signal.

Recording `gold_macro_f1` in each model's metrics.json lets version_manager
gate promotion on the honest number instead of the inflated val metric, and
makes the gap visible at training time instead of only when the eval suite
runs. Never raises — returns None on any problem so it can't break a run.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLD_DIR = _REPO_ROOT / "tests" / "evaluation" / "datasets"

# model_type -> (gold filename, threshold the eval suite asserts)
GOLD_SPECS: dict[str, tuple[str, float]] = {
    "bert":       ("gold_classifier.jsonl", 0.85),
    "actor":      ("gold_actor.jsonl",      0.80),
    "risk":       ("gold_risk.jsonl",       0.80),
    "prohibited": ("gold_prohibited.jsonl", 0.80),
}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_on_gold(
    model_type: str,
    model,
    tokenizer,
    labels: list[str],
    *,
    device: str = "cpu",
    max_length: int = 256,
    batch_size: int = 16,
) -> dict | None:
    """Run `model` on the gold set for `model_type`.

    Returns {accuracy, macro_f1, n, threshold, passed} or None if the gold
    file / required libs are unavailable. Gold rows are filtered to the
    model's label set so a binary model isn't scored on 8-class gold.
    """
    try:
        import torch
        from sklearn.metrics import accuracy_score, f1_score

        spec = GOLD_SPECS.get(model_type)
        if spec is None:
            return None
        fname, threshold = spec
        path = _GOLD_DIR / fname
        if not path.exists():
            print(f"  [gold] {fname} not found — gold gate skipped")
            return None

        label_set = set(labels)
        rows = [r for r in _load_jsonl(path) if r.get("label") in label_set]
        if not rows:
            print(f"  [gold] no rows in {fname} match labels {labels} — skipped")
            return None

        texts = [r["text"] for r in rows]
        y_true = [r["label"] for r in rows]
        id2label = {i: l for i, l in enumerate(labels)}

        model.eval()
        preds: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            preds.extend(id2label[int(p)] for p in logits.argmax(-1).tolist())

        acc = float(accuracy_score(y_true, preds))
        mf1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(mf1, 4),
            "n": len(rows),
            "threshold": threshold,
            "passed": mf1 >= threshold,
        }
    except Exception as exc:  # never break a training run over gold eval
        print(f"  [gold] evaluation skipped: {exc!r}")
        return None
