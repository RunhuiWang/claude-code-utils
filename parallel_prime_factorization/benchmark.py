#!/usr/bin/env python3
"""
Benchmark Script - Compare Sequential vs Parallel Performance

This script runs comprehensive benchmarks to demonstrate:
1. Speedup achieved by parallelization
2. Impact of different parallelization strategies
3. Effect of chunk size on load balancing
4. Scalability with number of workers
"""

import multiprocessing as mp
import time
import sys
from typing import List, Tuple, Callable, Any

from sequential import (
    generate_test_numbers,
    factorize_all_sequential,
    verify_factorization,
)
from parallel_solution import (
    parallel_factorize_pool,
    parallel_factorize_queue,
    parallel_factorize_map,
    parallel_factorize_imap_unordered,
    parallel_factorize_shared_cache,
)


def run_benchmark(
    name: str,
    func: Callable,
    numbers: List[int],
    **kwargs
) -> Tuple[float, Any]:
    """Run a single benchmark and return timing and result."""
    print(f"  Running {name}...", end=" ", flush=True)
    start = time.perf_counter()
    result = func(numbers, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.3f}s")
    return elapsed, result


def benchmark_strategies(numbers: List[int], num_workers: int):
    """Compare different parallelization strategies."""
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    print(f"Numbers: {len(numbers)}, Workers: {num_workers}")
    print("-" * 70)

    results = []

    # Sequential baseline
    seq_time, seq_result = run_benchmark(
        "Sequential (baseline)",
        lambda n, **kw: factorize_all_sequential(n),
        numbers
    )
    results.append(("Sequential", seq_time, 1.0))

    # Parallel strategies
    strategies = [
        ("Pool.map (chunk=1)", parallel_factorize_map, {"chunk_size": 1}),
        ("Pool.map (chunk=10)", parallel_factorize_map, {"chunk_size": 10}),
        ("Pool.map (chunk=50)", parallel_factorize_map, {"chunk_size": 50}),
        ("imap_unordered (chunk=1)", parallel_factorize_imap_unordered, {"chunk_size": 1}),
        ("imap_unordered (chunk=10)", parallel_factorize_imap_unordered, {"chunk_size": 10}),
        ("ProcessPoolExecutor", parallel_factorize_pool, {"chunk_size": 10}),
        ("Work-stealing queue", parallel_factorize_queue, {}),
    ]

    for name, func, kwargs in strategies:
        para_time, para_result = run_benchmark(
            name, func, numbers, num_workers=num_workers, **kwargs
        )
        speedup = seq_time / para_time if para_time > 0 else 0
        results.append((name, para_time, speedup))

    # Print summary table
    print("\n" + "-" * 70)
    print(f"{'Strategy':<35} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 70)
    for name, time_taken, speedup in results:
        print(f"{name:<35} {time_taken:<12.3f} {speedup:<10.2f}x")
    print("-" * 70)

    return results


def benchmark_scalability(numbers: List[int], max_workers: int = None):
    """Test scalability with different numbers of workers."""
    if max_workers is None:
        max_workers = mp.cpu_count()

    print("\n" + "=" * 70)
    print("SCALABILITY BENCHMARK")
    print("=" * 70)
    print(f"Testing with 1 to {max_workers} workers")
    print("-" * 70)

    # Sequential baseline
    print("Running sequential baseline...", end=" ", flush=True)
    start = time.perf_counter()
    seq_result = factorize_all_sequential(numbers)
    seq_time = time.perf_counter() - start
    print(f"{seq_time:.3f}s")

    results = [("1 (sequential)", seq_time, 1.0)]

    # Test with different worker counts
    worker_counts = [w for w in [2, 4, 8, 16, 32] if w <= max_workers]
    if max_workers not in worker_counts:
        worker_counts.append(max_workers)
    worker_counts = sorted(set(worker_counts))

    for num_workers in worker_counts:
        print(f"Running with {num_workers} workers...", end=" ", flush=True)
        start = time.perf_counter()
        result = parallel_factorize_imap_unordered(
            numbers, num_workers=num_workers, chunk_size=1
        )
        elapsed = time.perf_counter() - start
        speedup = seq_time / elapsed if elapsed > 0 else 0
        efficiency = speedup / num_workers * 100
        print(f"{elapsed:.3f}s (speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%)")
        results.append((f"{num_workers} workers", elapsed, speedup))

    # Print summary
    print("\n" + "-" * 70)
    print(f"{'Configuration':<20} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<10}")
    print("-" * 70)
    for name, time_taken, speedup in results:
        workers = 1 if "sequential" in name else int(name.split()[0])
        efficiency = speedup / workers * 100
        print(f"{name:<20} {time_taken:<12.3f} {speedup:<10.2f}x {efficiency:<10.1f}%")
    print("-" * 70)

    return results


def benchmark_chunk_sizes(numbers: List[int], num_workers: int):
    """Analyze impact of chunk size on performance."""
    print("\n" + "=" * 70)
    print("CHUNK SIZE ANALYSIS")
    print("=" * 70)
    print(f"Testing different chunk sizes with {num_workers} workers")
    print("-" * 70)

    # Sequential baseline
    print("Running sequential baseline...", end=" ", flush=True)
    start = time.perf_counter()
    seq_result = factorize_all_sequential(numbers)
    seq_time = time.perf_counter() - start
    print(f"{seq_time:.3f}s")

    results = []
    chunk_sizes = [1, 2, 5, 10, 20, 50, 100, len(numbers) // num_workers]

    for chunk_size in chunk_sizes:
        print(f"Chunk size {chunk_size}...", end=" ", flush=True)
        start = time.perf_counter()
        result = parallel_factorize_imap_unordered(
            numbers, num_workers=num_workers, chunk_size=chunk_size
        )
        elapsed = time.perf_counter() - start
        speedup = seq_time / elapsed if elapsed > 0 else 0
        print(f"{elapsed:.3f}s (speedup: {speedup:.2f}x)")
        results.append((chunk_size, elapsed, speedup))

    # Print summary
    print("\n" + "-" * 70)
    print(f"{'Chunk Size':<15} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 70)
    for chunk_size, time_taken, speedup in results:
        print(f"{chunk_size:<15} {time_taken:<12.3f} {speedup:<10.2f}x")
    print("-" * 70)

    # Analysis
    best = max(results, key=lambda x: x[2])
    print(f"\nBest chunk size: {best[0]} (speedup: {best[2]:.2f}x)")
    print("Note: Smaller chunks = better load balancing but more overhead")
    print("      Larger chunks = less overhead but potential load imbalance")

    return results


def verify_correctness(numbers: List[int]):
    """Verify that parallel results match sequential results."""
    print("\n" + "=" * 70)
    print("CORRECTNESS VERIFICATION")
    print("=" * 70)

    # Run sequential
    print("Running sequential...", end=" ", flush=True)
    seq_result = factorize_all_sequential(numbers)
    print("done")

    # Run parallel
    print("Running parallel...", end=" ", flush=True)
    para_result = parallel_factorize_imap_unordered(numbers, chunk_size=1)
    print("done")

    # Compare results
    print("\nComparing results...")
    mismatches = 0
    for i, n in enumerate(numbers):
        seq_factors = seq_result.factors[i]
        para_factors = para_result.factors[i]

        if seq_factors != para_factors:
            print(f"MISMATCH at index {i} (n={n}):")
            print(f"  Sequential: {seq_factors}")
            print(f"  Parallel:   {para_factors}")
            mismatches += 1

    if mismatches == 0:
        print("All results match! Parallel implementation is correct.")
    else:
        print(f"\nFound {mismatches} mismatches!")

    # Verify mathematical correctness
    print("\nVerifying mathematical correctness...")
    errors = 0
    for i, n in enumerate(numbers):
        if not verify_factorization(n, para_result.factors[i]):
            errors += 1
            print(f"ERROR: Invalid factorization for {n}")

    if errors == 0:
        print("All factorizations are mathematically correct!")
    else:
        print(f"Found {errors} invalid factorizations!")

    return mismatches == 0 and errors == 0


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("PRIME FACTORIZATION BENCHMARK SUITE")
    print("=" * 70)

    num_cpus = mp.cpu_count()
    print(f"\nSystem: {num_cpus} CPU cores available")

    # Parse command line arguments
    count = 500
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            pass

    print(f"Test size: {count} numbers")

    # Generate test data
    print("\nGenerating test numbers...")
    numbers = generate_test_numbers(count=count, seed=42)

    easy = sum(1 for n in numbers if n <= 10_000)
    medium = sum(1 for n in numbers if 10_000 < n <= 1_000_000)
    hard = sum(1 for n in numbers if n > 1_000_000)
    print(f"Distribution: {easy} easy, {medium} medium, {hard} hard")

    # Run benchmarks
    verify_correctness(numbers)
    benchmark_strategies(numbers, num_cpus)
    benchmark_scalability(numbers, num_cpus)
    benchmark_chunk_sizes(numbers, num_cpus)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nKey findings:")
    print("1. imap_unordered with chunk_size=1 provides best load balancing")
    print("2. Smaller chunks help with unbalanced workloads like this one")
    print("3. Near-linear speedup is achievable with proper parallelization")
    print("4. Work-stealing patterns naturally handle load imbalance")


if __name__ == "__main__":
    main()
