#!/usr/bin/env python3
"""Benchmark Triton Inference Server vs Ollama classifier latency. (Phase 5)

Compares per-chunk classification throughput:
  - Ollama + phi3:mini (sequential)
  - Triton + BERT ONNX ensemble (batched)

Usage:
    python scripts/benchmark_triton.py --n-samples 100
"""

# Phase 5 implementation — placeholder


def run_benchmark(n_samples: int) -> None:
    """Run latency comparison benchmark.

    Args:
        n_samples: Number of text samples to classify.
    """
    raise NotImplementedError("benchmark_triton.run_benchmark — implemented in Phase 5")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Triton vs Ollama benchmark (Phase 5)")
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()
    run_benchmark(args.n_samples)
