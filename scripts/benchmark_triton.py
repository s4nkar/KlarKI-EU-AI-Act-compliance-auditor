#!/usr/bin/env python3
"""Benchmark Triton BERT vs Ollama phi3:mini classification latency.

Compares per-chunk classification throughput:
  - Ollama + phi3:mini (sequential, ~2-4 s/chunk)
  - Triton + BERT ONNX (batched, ~0.02-0.05 s/chunk)

Usage:
    # Both services must be running:
    docker compose --profile triton up -d

    python scripts/benchmark_triton.py --n-samples 50 \
        --ollama-host http://localhost:11434 \
        --triton-host localhost --triton-port 8003
"""

import argparse
import asyncio
import json
import statistics
import time

SAMPLE_TEXTS = [
    "The organization maintains a formal risk register updated quarterly.",
    "Training datasets are sourced from internal customer records and public databases.",
    "Technical documentation includes system architecture diagrams and model cards.",
    "AI systems automatically log all predictions with timestamps and confidence scores.",
    "Users are informed at the point of interaction that they are using an AI system.",
    "All high-risk AI decisions are subject to mandatory human review before action.",
    "The system is tested for robustness against adversarial inputs before deployment.",
    "The quarterly board meeting discussed capital allocation for the new product line.",
]


def _expand_samples(texts: list[str], n: int) -> list[str]:
    """Repeat and cycle sample texts to reach n total."""
    result = []
    while len(result) < n:
        result.extend(texts)
    return result[:n]


async def _benchmark_ollama(texts: list[str], host: str, model: str) -> dict:
    """Run sequential Ollama classification and measure latency."""
    import httpx
    from pathlib import Path

    prompt_path = Path(__file__).parent.parent / "api" / "prompts" / "classify_chunk.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    latencies = []
    errors = 0

    async with httpx.AsyncClient(timeout=120) as client:
        for text in texts:
            prompt = prompt_template.replace("{chunk_text}", text)
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    f"{host}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
                latencies.append(time.perf_counter() - t0)
            except Exception:
                errors += 1
                latencies.append(time.perf_counter() - t0)

    return {
        "backend":       "ollama",
        "model":         model,
        "n_samples":     len(texts),
        "errors":        errors,
        "mean_s":        round(statistics.mean(latencies), 3),
        "median_s":      round(statistics.median(latencies), 3),
        "p95_s":         round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        "total_s":       round(sum(latencies), 1),
        "throughput_per_min": round(60 / statistics.mean(latencies), 1),
    }


async def _benchmark_triton(texts: list[str], host: str, grpc_port: int) -> dict:
    """Run batched Triton classification and measure latency."""
    import sys
    sys.path.insert(0, str(__file__).rsplit("/scripts", 1)[0] + "/api")

    from services.triton_client import TritonClient

    client = TritonClient(host=host, grpc_port=grpc_port)

    batch_size = 32
    latencies = []
    errors = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        t0 = time.perf_counter()
        try:
            await client.classify(batch)
            elapsed = time.perf_counter() - t0
            # Record per-sample latency
            latencies.extend([elapsed / len(batch)] * len(batch))
        except Exception as exc:
            errors += len(batch)
            print(f"  Triton error on batch {i//batch_size}: {exc}")

    if not latencies:
        return {"backend": "triton", "error": "all batches failed"}

    return {
        "backend":        "triton",
        "model":          "bert_clause_classifier (ONNX)",
        "n_samples":      len(texts),
        "errors":         errors,
        "mean_s":         round(statistics.mean(latencies), 4),
        "median_s":       round(statistics.median(latencies), 4),
        "p95_s":          round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
        "total_s":        round(sum(latencies), 2),
        "throughput_per_min": round(60 / statistics.mean(latencies), 0),
    }


def _print_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 62)
    print("  KlarKI Classifier Benchmark — Triton BERT vs Ollama phi3:mini")
    print("=" * 62)

    headers = ["Backend", "N", "Mean (s)", "Median (s)", "P95 (s)", "Throughput/min"]
    row_fmt = "{:<22} {:>6} {:>10} {:>12} {:>9} {:>16}"
    print(row_fmt.format(*headers))
    print("-" * 62)

    for r in results:
        if "error" in r:
            print(f"  {r['backend']}: {r['error']}")
            continue
        print(row_fmt.format(
            r["backend"],
            r["n_samples"],
            r["mean_s"],
            r["median_s"],
            r["p95_s"],
            r["throughput_per_min"],
        ))

    print("=" * 62)

    if len(results) == 2 and all("mean_s" in r for r in results):
        speedup = results[0]["mean_s"] / results[1]["mean_s"]
        print(f"\n  Triton is {speedup:.1f}x faster than Ollama per chunk")
        print(f"  (batch size 32, TensorRT FP16, RTX 3050 Ti)")
    print()


async def _main(args: argparse.Namespace) -> None:
    texts = _expand_samples(SAMPLE_TEXTS, args.n_samples)
    results = []

    if not args.triton_only:
        print(f"Benchmarking Ollama ({args.n_samples} samples, sequential)…")
        r = await _benchmark_ollama(texts, args.ollama_host, args.model)
        results.append(r)
        print(f"  Done — mean {r['mean_s']} s/chunk, total {r['total_s']} s")

    if not args.ollama_only:
        print(f"Benchmarking Triton ({args.n_samples} samples, batched)…")
        r = await _benchmark_triton(texts, args.triton_host, args.triton_port)
        results.append(r)
        if "mean_s" in r:
            print(f"  Done — mean {r['mean_s']} s/chunk, total {r['total_s']} s")

    _print_table(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


def run_benchmark(n_samples: int) -> None:
    """Run latency comparison benchmark (programmatic entry point)."""
    args = argparse.Namespace(
        n_samples=n_samples,
        ollama_host="http://localhost:11434",
        model="phi3:mini",
        triton_host="localhost",
        triton_port=8003,
        ollama_only=False,
        triton_only=False,
        output=None,
    )
    asyncio.run(_main(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton BERT vs Ollama classification benchmark")
    parser.add_argument("--n-samples",    type=int,   default=50,                          help="Number of texts to classify")
    parser.add_argument("--ollama-host",  default="http://localhost:11434",                 help="Ollama API base URL")
    parser.add_argument("--model",        default="phi3:mini",                              help="Ollama model name")
    parser.add_argument("--triton-host",  default="localhost",                              help="Triton server hostname")
    parser.add_argument("--triton-port",  type=int,   default=8003,                         help="Triton gRPC port (host-side)")
    parser.add_argument("--ollama-only",  action="store_true",                              help="Only benchmark Ollama")
    parser.add_argument("--triton-only",  action="store_true",                              help="Only benchmark Triton")
    parser.add_argument("--output",       default=None,                                     help="Save JSON results to file")
    asyncio.run(_main(parser.parse_args()))
