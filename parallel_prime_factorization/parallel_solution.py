#!/usr/bin/env python3
"""
Parallel Prime Factorization - Optimized Parallel Implementation

This module demonstrates proper parallelization techniques for
computationally-bound tasks with:
1. Unbalanced workload
2. Shared state (prime cache)
3. Dynamic load balancing
4. Race condition handling
"""

import os
import time
import multiprocessing as mp
from multiprocessing import Manager, Queue, Value, Lock
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from queue import Empty
import ctypes

from sequential import (
    generate_test_numbers,
    verify_factorization,
    FactorizationResult,
)


# ============================================================================
# Worker Functions (must be at module level for pickling)
# ============================================================================

def factorize_indexed(args: Tuple[int, int]) -> Tuple[int, Dict[int, int], Set[int]]:
    """
    Factorize a number with its index (for use with imap_unordered).

    Args:
        args: Tuple of (index, number)

    Returns:
        Tuple of (index, factors_dict, discovered_primes)
    """
    idx, n = args
    _, factors, primes = factorize_single(n)
    return (idx, factors, primes)


def factorize_single(n: int) -> Tuple[int, Dict[int, int], Set[int]]:
    """
    Factorize a single number and return discovered primes.

    Returns:
        Tuple of (original_number, factors_dict, discovered_primes)
    """
    if n < 2:
        return (n, {}, set())

    factors = {}
    discovered_primes = set()

    # Check for factor of 2
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2

    if 2 in factors:
        discovered_primes.add(2)

    if n == 1:
        return (n, factors, discovered_primes)

    # Check odd factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
            discovered_primes.add(i)
        i += 2

    # If n is still greater than 1, it's a prime factor
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
        discovered_primes.add(n)

    return (n, factors, discovered_primes)


def worker_process_chunk(args: Tuple[List[Tuple[int, int]], int]) -> List[Tuple[int, Dict[int, int], Set[int]]]:
    """
    Process a chunk of (index, number) pairs.

    This worker function is designed to work with ProcessPoolExecutor.
    Each chunk is processed independently to minimize synchronization overhead.

    Args:
        args: Tuple of (chunk of (index, number) pairs, worker_id)

    Returns:
        List of (index, factors, discovered_primes) for each number in chunk
    """
    chunk, worker_id = args
    results = []

    for idx, number in chunk:
        _, factors, primes = factorize_single(number)
        results.append((idx, factors, primes))

    return results


def worker_with_queue(
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    done_flag: mp.Value,
    worker_id: int
):
    """
    Worker process that pulls work from a shared queue.

    This implements dynamic load balancing - workers keep pulling
    work until the queue is empty, naturally balancing the load.

    Args:
        work_queue: Queue of (index, number) pairs to process
        result_queue: Queue to put results
        done_flag: Shared flag indicating all work is distributed
        worker_id: ID of this worker for debugging
    """
    processed = 0
    local_primes = set()

    while True:
        try:
            # Try to get work with a timeout
            idx, number = work_queue.get(timeout=0.01)
            _, factors, primes = factorize_single(number)
            local_primes.update(primes)
            result_queue.put((idx, factors, primes))
            processed += 1
        except Empty:
            # Check if we're done
            if done_flag.value and work_queue.empty():
                break

    # Signal this worker is done
    result_queue.put(('DONE', worker_id, processed, local_primes))


# ============================================================================
# Parallel Factorization Strategies
# ============================================================================

@dataclass
class ParallelResult:
    """Result from parallel factorization."""
    factors: Dict[int, Dict[int, int]]
    prime_cache: Set[int]
    elapsed_time: float
    strategy: str
    num_workers: int


def parallel_factorize_pool(
    numbers: List[int],
    num_workers: Optional[int] = None,
    chunk_size: int = 10
) -> ParallelResult:
    """
    Parallel factorization using ProcessPoolExecutor with chunking.

    Strategy: Divide work into small chunks and submit to a process pool.
    Small chunks enable better load balancing at the cost of more overhead.

    Args:
        numbers: List of numbers to factorize
        num_workers: Number of worker processes (default: CPU count)
        chunk_size: Number of items per work unit (smaller = better balance)

    Returns:
        ParallelResult with all factorizations
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    start_time = time.perf_counter()

    # Prepare indexed work items
    indexed_numbers = list(enumerate(numbers))

    # Create chunks - smaller chunks = better load balancing
    chunks = []
    for i in range(0, len(indexed_numbers), chunk_size):
        chunk = indexed_numbers[i:i + chunk_size]
        chunks.append((chunk, i // chunk_size))

    # Process chunks in parallel
    all_factors: Dict[int, Dict[int, int]] = {}
    all_primes: Set[int] = set()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all chunks
        futures = [executor.submit(worker_process_chunk, chunk) for chunk in chunks]

        # Collect results as they complete (for better responsiveness)
        for future in as_completed(futures):
            results = future.result()
            for idx, factors, primes in results:
                all_factors[idx] = factors
                all_primes.update(primes)

    elapsed = time.perf_counter() - start_time

    return ParallelResult(
        factors=all_factors,
        prime_cache=all_primes,
        elapsed_time=elapsed,
        strategy="ProcessPoolExecutor with chunking",
        num_workers=num_workers
    )


def parallel_factorize_queue(
    numbers: List[int],
    num_workers: Optional[int] = None
) -> ParallelResult:
    """
    Parallel factorization using work-stealing queue pattern.

    Strategy: Use a shared queue that workers pull from. This provides
    natural load balancing - fast workers process more items.

    This approach handles race conditions through:
    1. Queue operations are atomic
    2. Each result is associated with its original index
    3. Workers only write to their own result entries

    Args:
        numbers: List of numbers to factorize
        num_workers: Number of worker processes (default: CPU count)

    Returns:
        ParallelResult with all factorizations
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    start_time = time.perf_counter()

    # Create work and result queues
    work_queue = mp.Queue()
    result_queue = mp.Queue()
    done_flag = mp.Value(ctypes.c_bool, False)

    # Populate work queue
    for idx, number in enumerate(numbers):
        work_queue.put((idx, number))

    # Start worker processes
    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_with_queue,
            args=(work_queue, result_queue, done_flag, i)
        )
        p.start()
        workers.append(p)

    # Signal that all work has been distributed
    done_flag.value = True

    # Collect results
    all_factors: Dict[int, Dict[int, int]] = {}
    all_primes: Set[int] = set()
    workers_done = 0

    while workers_done < num_workers:
        result = result_queue.get()
        if result[0] == 'DONE':
            _, worker_id, processed, local_primes = result
            all_primes.update(local_primes)
            workers_done += 1
        else:
            idx, factors, primes = result
            all_factors[idx] = factors
            all_primes.update(primes)

    # Wait for all workers to finish
    for p in workers:
        p.join()

    elapsed = time.perf_counter() - start_time

    return ParallelResult(
        factors=all_factors,
        prime_cache=all_primes,
        elapsed_time=elapsed,
        strategy="Work-stealing queue",
        num_workers=num_workers
    )


def parallel_factorize_map(
    numbers: List[int],
    num_workers: Optional[int] = None,
    chunk_size: int = 1
) -> ParallelResult:
    """
    Parallel factorization using Pool.map with automatic chunking.

    Strategy: Use multiprocessing Pool with map for simpler code.
    The chunksize parameter controls load balancing.

    Args:
        numbers: List of numbers to factorize
        num_workers: Number of worker processes (default: CPU count)
        chunk_size: Chunk size for map (1 = best balance, higher = less overhead)

    Returns:
        ParallelResult with all factorizations
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    start_time = time.perf_counter()

    # Process all numbers in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(factorize_single, numbers, chunksize=chunk_size)

    # Aggregate results
    all_factors: Dict[int, Dict[int, int]] = {}
    all_primes: Set[int] = set()

    for idx, (_, factors, primes) in enumerate(results):
        all_factors[idx] = factors
        all_primes.update(primes)

    elapsed = time.perf_counter() - start_time

    return ParallelResult(
        factors=all_factors,
        prime_cache=all_primes,
        elapsed_time=elapsed,
        strategy=f"Pool.map (chunksize={chunk_size})",
        num_workers=num_workers
    )


def parallel_factorize_imap_unordered(
    numbers: List[int],
    num_workers: Optional[int] = None,
    chunk_size: int = 1
) -> ParallelResult:
    """
    Parallel factorization using Pool.imap_unordered for best load balancing.

    Strategy: Use imap_unordered which processes results as they complete,
    providing the best load balancing for uneven workloads.

    Args:
        numbers: List of numbers to factorize
        num_workers: Number of worker processes (default: CPU count)
        chunk_size: Chunk size (1 = best balance)

    Returns:
        ParallelResult with all factorizations
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    start_time = time.perf_counter()

    # Create indexed pairs to track original positions
    indexed = list(enumerate(numbers))

    # Process with imap_unordered for dynamic load balancing
    all_factors: Dict[int, Dict[int, int]] = {}
    all_primes: Set[int] = set()

    with mp.Pool(processes=num_workers) as pool:
        for idx, factors, primes in pool.imap_unordered(
            factorize_indexed, indexed, chunksize=chunk_size
        ):
            all_factors[idx] = factors
            all_primes.update(primes)

    elapsed = time.perf_counter() - start_time

    return ParallelResult(
        factors=all_factors,
        prime_cache=all_primes,
        elapsed_time=elapsed,
        strategy=f"Pool.imap_unordered (chunksize={chunk_size})",
        num_workers=num_workers
    )


# ============================================================================
# Shared Prime Cache Implementation (Advanced)
# ============================================================================

def worker_with_shared_cache(args):
    """
    Worker that uses a shared prime cache.

    This demonstrates handling race conditions with shared mutable state.
    The Manager provides synchronization for the shared set.
    """
    idx, number, shared_primes, lock = args

    if number < 2:
        return (idx, {}, set())

    factors = {}
    local_primes = set()
    n = number

    # Check for factor of 2
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2

    if 2 in factors:
        local_primes.add(2)

    if n == 1:
        return (idx, factors, local_primes)

    # Try to use cached primes first (read without lock for performance)
    # This is a benign race - we might miss some primes but still get correct results
    try:
        cached = set(shared_primes)  # Snapshot of cached primes
    except:
        cached = set()

    # Check cached primes first (optimization)
    for p in sorted(cached):
        if p * p > n:
            break
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
            local_primes.add(p)

    # Continue with trial division for uncached factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
            local_primes.add(i)
        i += 2

    if n > 1:
        factors[n] = factors.get(n, 0) + 1
        local_primes.add(n)

    # Update shared cache with new primes (with lock for correctness)
    with lock:
        for p in local_primes:
            shared_primes.add(p)

    return (idx, factors, local_primes)


def parallel_factorize_shared_cache(
    numbers: List[int],
    num_workers: Optional[int] = None
) -> ParallelResult:
    """
    Parallel factorization with shared prime cache.

    This demonstrates proper handling of shared mutable state:
    1. Use Manager for cross-process shared state
    2. Use Lock for synchronized updates
    3. Read-heavy operations can skip locking (benign race)
    4. Write operations use locks for correctness
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    start_time = time.perf_counter()

    # Create shared state using Manager
    with Manager() as manager:
        shared_primes = manager.list()  # Shared list (acts like set for our purpose)
        lock = manager.Lock()

        # Prepare work items
        work_items = [
            (idx, num, shared_primes, lock)
            for idx, num in enumerate(numbers)
        ]

        # Process in parallel
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(worker_with_shared_cache, work_items, chunksize=1)

        # Aggregate results
        all_factors: Dict[int, Dict[int, int]] = {}
        all_primes: Set[int] = set()

        for idx, factors, primes in results:
            all_factors[idx] = factors
            all_primes.update(primes)

    elapsed = time.perf_counter() - start_time

    return ParallelResult(
        factors=all_factors,
        prime_cache=all_primes,
        elapsed_time=elapsed,
        strategy="Shared prime cache with Manager",
        num_workers=num_workers
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Demonstrate parallel factorization with multiple strategies."""
    print("=" * 70)
    print("Parallel Prime Factorization")
    print("=" * 70)

    num_cpus = mp.cpu_count()
    print(f"\nSystem has {num_cpus} CPU cores")

    # Generate test numbers
    print("\nGenerating test numbers...")
    numbers = generate_test_numbers(count=500, seed=42)
    print(f"Generated {len(numbers)} numbers")

    # Show distribution
    easy = sum(1 for n in numbers if n <= 10_000)
    medium = sum(1 for n in numbers if 10_000 < n <= 1_000_000)
    hard = sum(1 for n in numbers if n > 1_000_000)
    print(f"Distribution: {easy} easy, {medium} medium, {hard} hard")

    # Run the recommended strategy (imap_unordered with chunksize=1)
    print("\n" + "-" * 70)
    print("Running parallel factorization (imap_unordered, chunksize=1)...")
    print("-" * 70)

    result = parallel_factorize_imap_unordered(numbers, chunk_size=1)

    print(f"\nStrategy: {result.strategy}")
    print(f"Workers: {result.num_workers}")
    print(f"Completed in {result.elapsed_time:.3f} seconds")
    print(f"Discovered {len(result.prime_cache)} unique primes")

    # Verify results
    print("\nVerifying results...")
    errors = 0
    for i, n in enumerate(numbers):
        if not verify_factorization(n, result.factors[i]):
            print(f"ERROR: Factorization failed for {n}")
            errors += 1

    if errors == 0:
        print("All factorizations verified correctly!")
    else:
        print(f"Found {errors} errors in factorization")

    # Show some example factorizations
    print("\nExample factorizations:")
    for i in range(min(5, len(numbers))):
        n = numbers[i]
        f = result.factors[i]
        factors_str = " x ".join(f"{p}^{e}" if e > 1 else str(p)
                                  for p, e in sorted(f.items()))
        print(f"  {n} = {factors_str}")

    return result


if __name__ == "__main__":
    main()
