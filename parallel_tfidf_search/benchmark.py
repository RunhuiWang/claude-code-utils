#!/usr/bin/env python3
"""
Benchmark Script - Compare Sequential vs Parallel TF-IDF Performance

This script runs comprehensive benchmarks to demonstrate:
1. Speedup achieved by parallel indexing
2. Impact of corpus size on speedup
3. Parallel batch search performance
4. Scalability with number of workers
"""

import time
import multiprocessing as mp
import argparse
from typing import List, Tuple, Dict
import sys

from document_generator import generate_corpus, Document
from sequential import (
    build_tfidf_index_sequential,
    batch_search_sequential,
)
from parallel_solution import (
    build_tfidf_index_parallel,
    build_tfidf_index_parallel_futures,
    batch_search_parallel,
)


def benchmark_indexing(
    documents: List[Document],
    num_workers: int
) -> Dict:
    """Run indexing benchmarks."""
    results = {}

    # Sequential baseline
    print("  Running sequential indexing...", end=" ", flush=True)
    start = time.perf_counter()
    seq_result = build_tfidf_index_sequential(documents)
    seq_time = time.perf_counter() - start
    print(f"{seq_time:.3f}s")
    results["sequential"] = seq_time

    # Parallel with Pool
    print("  Running parallel (Pool)...", end=" ", flush=True)
    start = time.perf_counter()
    para_result = build_tfidf_index_parallel(documents, num_workers=num_workers)
    para_time = time.perf_counter() - start
    print(f"{para_time:.3f}s (speedup: {seq_time/para_time:.2f}x)")
    results["parallel_pool"] = para_time

    # Parallel with Futures
    print("  Running parallel (Futures)...", end=" ", flush=True)
    start = time.perf_counter()
    futures_result = build_tfidf_index_parallel_futures(documents, num_workers=num_workers)
    futures_time = time.perf_counter() - start
    print(f"{futures_time:.3f}s (speedup: {seq_time/futures_time:.2f}x)")
    results["parallel_futures"] = futures_time

    results["vocab_size"] = seq_result.vocabulary_size
    return results


def benchmark_search(
    index,
    queries: List[str],
    documents: List[Document],
    num_workers: int
) -> Dict:
    """Run search benchmarks."""
    results = {}

    # Sequential batch search
    print("  Running sequential batch search...", end=" ", flush=True)
    start = time.perf_counter()
    seq_results = batch_search_sequential(queries, index, top_k=10, documents=documents)
    seq_time = time.perf_counter() - start
    print(f"{seq_time*1000:.2f}ms")
    results["sequential"] = seq_time

    # Parallel batch search
    print("  Running parallel batch search...", end=" ", flush=True)
    start = time.perf_counter()
    para_results, para_time = batch_search_parallel(
        queries, index, top_k=10, num_workers=num_workers, documents=documents
    )
    print(f"{para_time*1000:.2f}ms (speedup: {seq_time/para_time:.2f}x)")
    results["parallel"] = para_time

    return results


def benchmark_corpus_sizes(num_workers: int, sizes: List[int] = None):
    """Benchmark with different corpus sizes."""
    if sizes is None:
        sizes = [1000, 2500, 5000, 10000, 20000]

    print("\n" + "=" * 70)
    print("CORPUS SIZE SCALING BENCHMARK")
    print("=" * 70)
    print(f"Workers: {num_workers}")

    results = []

    for size in sizes:
        print(f"\n--- Corpus size: {size:,} documents ---")
        documents = generate_corpus(size, seed=42)
        total_words = sum(d.word_count for d in documents)
        print(f"Total words: {total_words:,}")

        bench_result = benchmark_indexing(documents, num_workers)
        seq_time = bench_result["sequential"]
        para_time = bench_result["parallel_pool"]
        speedup = seq_time / para_time

        results.append({
            "size": size,
            "words": total_words,
            "sequential": seq_time,
            "parallel": para_time,
            "speedup": speedup,
            "vocab_size": bench_result["vocab_size"]
        })

    # Print summary table
    print("\n" + "-" * 70)
    print(f"{'Docs':<10} {'Words':<12} {'Seq (s)':<10} {'Para (s)':<10} {'Speedup':<10} {'Vocab':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['size']:<10} {r['words']:<12,} {r['sequential']:<10.3f} {r['parallel']:<10.3f} {r['speedup']:<10.2f}x {r['vocab_size']:<10,}")
    print("-" * 70)

    return results


def benchmark_scalability(documents: List[Document], max_workers: int = None):
    """Test scalability with different worker counts."""
    if max_workers is None:
        max_workers = mp.cpu_count()

    print("\n" + "=" * 70)
    print("WORKER SCALABILITY BENCHMARK")
    print("=" * 70)
    print(f"Corpus: {len(documents):,} documents")

    # Sequential baseline
    print("\nRunning sequential baseline...", end=" ", flush=True)
    start = time.perf_counter()
    seq_result = build_tfidf_index_sequential(documents)
    seq_time = time.perf_counter() - start
    print(f"{seq_time:.3f}s")

    results = [("1 (sequential)", seq_time, 1.0, 100.0)]

    # Test different worker counts
    worker_counts = [w for w in [2, 4, 8, 12, 16, 24, 32] if w <= max_workers]
    if max_workers not in worker_counts:
        worker_counts.append(max_workers)
    worker_counts = sorted(set(worker_counts))

    for num_workers in worker_counts:
        print(f"Running with {num_workers} workers...", end=" ", flush=True)
        start = time.perf_counter()
        para_result = build_tfidf_index_parallel(documents, num_workers=num_workers)
        para_time = time.perf_counter() - start
        speedup = seq_time / para_time
        efficiency = speedup / num_workers * 100
        print(f"{para_time:.3f}s (speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%)")
        results.append((f"{num_workers} workers", para_time, speedup, efficiency))

    # Print summary
    print("\n" + "-" * 70)
    print(f"{'Configuration':<20} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<10}")
    print("-" * 70)
    for name, time_taken, speedup, efficiency in results:
        print(f"{name:<20} {time_taken:<12.3f} {speedup:<10.2f}x {efficiency:<10.1f}%")
    print("-" * 70)

    return results


def benchmark_batch_search(documents: List[Document], num_workers: int):
    """Benchmark batch search with varying query counts."""
    print("\n" + "=" * 70)
    print("BATCH SEARCH BENCHMARK")
    print("=" * 70)

    # Build index first
    print("\nBuilding index...", end=" ", flush=True)
    result = build_tfidf_index_parallel(documents, num_workers=num_workers)
    print(f"done ({result.elapsed_time:.3f}s)")

    # Generate sample queries
    base_queries = [
        "machine learning neural network deep",
        "database query optimization performance index",
        "clinical trial patient treatment outcome",
        "market analysis investment portfolio risk",
        "scientific research methodology experiment",
        "software development agile testing deployment",
        "network security encryption authentication protocol",
        "data analytics streaming pipeline processing",
        "cloud infrastructure scalability reliability",
        "algorithm optimization parallel distributed computing"
    ]

    query_counts = [10, 50, 100, 500, 1000]
    results = []

    for count in query_counts:
        # Repeat queries to get desired count
        queries = (base_queries * (count // len(base_queries) + 1))[:count]

        print(f"\n--- {count} queries ---")

        # Sequential
        start = time.perf_counter()
        seq_results = batch_search_sequential(queries, result.index, top_k=10, documents=documents)
        seq_time = time.perf_counter() - start

        # Parallel
        para_results, para_time = batch_search_parallel(
            queries, result.index, top_k=10, num_workers=num_workers, documents=documents
        )

        speedup = seq_time / para_time if para_time > 0 else 0
        print(f"  Sequential: {seq_time*1000:.2f}ms ({seq_time*1000/count:.2f}ms/query)")
        print(f"  Parallel:   {para_time*1000:.2f}ms ({para_time*1000/count:.2f}ms/query)")
        print(f"  Speedup:    {speedup:.2f}x")

        results.append({
            "queries": count,
            "sequential": seq_time,
            "parallel": para_time,
            "speedup": speedup
        })

    # Print summary
    print("\n" + "-" * 70)
    print(f"{'Queries':<10} {'Seq (ms)':<12} {'Para (ms)':<12} {'Speedup':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['queries']:<10} {r['sequential']*1000:<12.2f} {r['parallel']*1000:<12.2f} {r['speedup']:<10.2f}x")
    print("-" * 70)

    return results


def benchmark_chunk_sizes(documents: List[Document], num_workers: int):
    """Benchmark impact of chunk size on parallel indexing."""
    print("\n" + "=" * 70)
    print("CHUNK SIZE ANALYSIS")
    print("=" * 70)
    print(f"Corpus: {len(documents):,} documents, Workers: {num_workers}")

    # Sequential baseline
    print("\nRunning sequential baseline...", end=" ", flush=True)
    start = time.perf_counter()
    seq_result = build_tfidf_index_sequential(documents)
    seq_time = time.perf_counter() - start
    print(f"{seq_time:.3f}s")

    chunk_sizes = [10, 25, 50, 100, 250, 500, 1000]
    results = []

    for chunk_size in chunk_sizes:
        print(f"Chunk size {chunk_size}...", end=" ", flush=True)
        start = time.perf_counter()
        para_result = build_tfidf_index_parallel(
            documents, num_workers=num_workers, chunk_size=chunk_size
        )
        para_time = time.perf_counter() - start
        speedup = seq_time / para_time
        print(f"{para_time:.3f}s (speedup: {speedup:.2f}x)")
        results.append((chunk_size, para_time, speedup))

    # Print summary
    print("\n" + "-" * 70)
    print(f"{'Chunk Size':<15} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 70)
    for chunk_size, time_taken, speedup in results:
        print(f"{chunk_size:<15} {time_taken:<12.3f} {speedup:<10.2f}x")
    print("-" * 70)

    best = max(results, key=lambda x: x[2])
    print(f"\nBest chunk size: {best[0]} (speedup: {best[2]:.2f}x)")

    return results


def main():
    parser = argparse.ArgumentParser(description="TF-IDF Benchmark Suite")
    parser.add_argument(
        "--num-docs", type=int, default=10000,
        help="Number of documents for main benchmarks"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick benchmark with smaller corpus"
    )

    args = parser.parse_args()

    num_cpus = mp.cpu_count()
    print("=" * 70)
    print("TF-IDF PARALLEL BENCHMARK SUITE")
    print("=" * 70)
    print(f"\nSystem: {num_cpus} CPU cores available")

    if args.quick:
        num_docs = 5000
        print("Running QUICK benchmark mode")
    else:
        num_docs = args.num_docs

    print(f"Main corpus size: {num_docs:,} documents")

    # Generate main corpus
    print("\nGenerating main corpus...")
    documents = generate_corpus(num_docs, seed=42)
    total_words = sum(d.word_count for d in documents)
    print(f"Total words: {total_words:,}")

    # Run benchmarks
    benchmark_corpus_sizes(num_cpus, sizes=[1000, 2500, 5000, 10000] if not args.quick else [1000, 2500, 5000])
    benchmark_scalability(documents, num_cpus)
    benchmark_chunk_sizes(documents, num_cpus)
    benchmark_batch_search(documents, num_cpus)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nKey findings:")
    print("1. Parallel indexing provides significant speedup for large corpora")
    print("2. Speedup improves with corpus size (more work to distribute)")
    print("3. Batch search benefits most from parallelization")
    print("4. Optimal chunk size balances overhead vs load distribution")


if __name__ == "__main__":
    main()
