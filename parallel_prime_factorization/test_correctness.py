#!/usr/bin/env python3
"""
Correctness Tests for Parallel Prime Factorization

This module provides comprehensive tests to ensure the parallel
implementation produces identical results to the sequential version.
"""

import unittest
import multiprocessing as mp
from typing import Dict, List

from sequential import (
    generate_test_numbers,
    factorize_all_sequential,
    factorize,
    verify_factorization,
    is_prime_simple,
)
from parallel_solution import (
    parallel_factorize_pool,
    parallel_factorize_queue,
    parallel_factorize_map,
    parallel_factorize_imap_unordered,
    factorize_single,
)


class TestFactorization(unittest.TestCase):
    """Test basic factorization correctness."""

    def test_small_numbers(self):
        """Test factorization of small numbers."""
        test_cases = [
            (2, {2: 1}),
            (3, {3: 1}),
            (4, {2: 2}),
            (6, {2: 1, 3: 1}),
            (12, {2: 2, 3: 1}),
            (84, {2: 2, 3: 1, 7: 1}),
            (100, {2: 2, 5: 2}),
            (1, {}),
            (0, {}),
        ]

        for n, expected in test_cases:
            _, factors, _ = factorize_single(n)
            self.assertEqual(factors, expected, f"Failed for {n}")

    def test_prime_numbers(self):
        """Test factorization of prime numbers."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 97, 101, 1009]

        for p in primes:
            _, factors, _ = factorize_single(p)
            self.assertEqual(factors, {p: 1}, f"Failed for prime {p}")

    def test_powers_of_two(self):
        """Test factorization of powers of two."""
        for exp in range(1, 20):
            n = 2 ** exp
            _, factors, _ = factorize_single(n)
            self.assertEqual(factors, {2: exp}, f"Failed for 2^{exp}")

    def test_semi_primes(self):
        """Test factorization of semi-primes (product of two primes)."""
        test_cases = [
            (6, {2: 1, 3: 1}),
            (15, {3: 1, 5: 1}),
            (21, {3: 1, 7: 1}),
            (143, {11: 1, 13: 1}),  # 11 * 13
            (221, {13: 1, 17: 1}),  # 13 * 17
        ]

        for n, expected in test_cases:
            _, factors, _ = factorize_single(n)
            self.assertEqual(factors, expected, f"Failed for semi-prime {n}")

    def test_large_semi_primes(self):
        """Test factorization of larger semi-primes."""
        # These are the challenging cases for parallelization
        primes = [10007, 10009, 10037, 10039]

        for i in range(len(primes) - 1):
            n = primes[i] * primes[i + 1]
            _, factors, _ = factorize_single(n)

            # Verify product equals original
            product = 1
            for p, exp in factors.items():
                product *= p ** exp
            self.assertEqual(product, n, f"Factorization product mismatch for {n}")

            # Verify all factors are prime
            for p in factors.keys():
                self.assertTrue(is_prime_simple(p), f"Non-prime factor {p} for {n}")

    def test_verify_factorization_function(self):
        """Test the verification helper function."""
        self.assertTrue(verify_factorization(84, {2: 2, 3: 1, 7: 1}))
        self.assertTrue(verify_factorization(100, {2: 2, 5: 2}))
        self.assertTrue(verify_factorization(1, {}))
        self.assertFalse(verify_factorization(84, {2: 1, 3: 1, 7: 1}))  # Wrong
        self.assertFalse(verify_factorization(100, {2: 2, 5: 1}))  # Wrong


class TestParallelCorrectness(unittest.TestCase):
    """Test that parallel implementations match sequential."""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests."""
        cls.numbers = generate_test_numbers(count=200, seed=42)
        cls.sequential_result = factorize_all_sequential(cls.numbers)

    def _compare_results(self, parallel_result, name):
        """Helper to compare parallel results with sequential baseline."""
        for i, n in enumerate(self.numbers):
            seq_factors = self.sequential_result.factors[i]
            para_factors = parallel_result.factors[i]

            self.assertEqual(
                seq_factors, para_factors,
                f"{name}: Mismatch at index {i} (n={n})\n"
                f"Sequential: {seq_factors}\n"
                f"Parallel:   {para_factors}"
            )

    def test_pool_map_correctness(self):
        """Test Pool.map strategy produces correct results."""
        result = parallel_factorize_map(self.numbers, chunk_size=1)
        self._compare_results(result, "Pool.map")

    def test_imap_unordered_correctness(self):
        """Test imap_unordered strategy produces correct results."""
        result = parallel_factorize_imap_unordered(self.numbers, chunk_size=1)
        self._compare_results(result, "imap_unordered")

    def test_process_pool_executor_correctness(self):
        """Test ProcessPoolExecutor strategy produces correct results."""
        result = parallel_factorize_pool(self.numbers, chunk_size=5)
        self._compare_results(result, "ProcessPoolExecutor")

    def test_queue_strategy_correctness(self):
        """Test work-stealing queue strategy produces correct results."""
        result = parallel_factorize_queue(self.numbers)
        self._compare_results(result, "Queue")

    def test_all_results_mathematically_correct(self):
        """Verify all parallel results are mathematically valid."""
        strategies = [
            ("Pool.map", parallel_factorize_map(self.numbers, chunk_size=1)),
            ("imap_unordered", parallel_factorize_imap_unordered(self.numbers, chunk_size=1)),
            ("ProcessPoolExecutor", parallel_factorize_pool(self.numbers, chunk_size=5)),
        ]

        for name, result in strategies:
            for i, n in enumerate(self.numbers):
                factors = result.factors[i]
                self.assertTrue(
                    verify_factorization(n, factors),
                    f"{name}: Invalid factorization for {n}: {factors}"
                )


class TestChunkSizeVariations(unittest.TestCase):
    """Test that different chunk sizes produce consistent results."""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests."""
        cls.numbers = generate_test_numbers(count=100, seed=123)
        cls.sequential_result = factorize_all_sequential(cls.numbers)

    def test_various_chunk_sizes(self):
        """Test that different chunk sizes produce identical results."""
        chunk_sizes = [1, 2, 5, 10, 25, 50]
        results = {}

        for chunk_size in chunk_sizes:
            result = parallel_factorize_imap_unordered(
                self.numbers, chunk_size=chunk_size
            )
            results[chunk_size] = result

        # All results should match
        for chunk_size in chunk_sizes:
            for i, n in enumerate(self.numbers):
                self.assertEqual(
                    self.sequential_result.factors[i],
                    results[chunk_size].factors[i],
                    f"Chunk size {chunk_size} produced different result for {n}"
                )


class TestWorkerCountVariations(unittest.TestCase):
    """Test that different worker counts produce consistent results."""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests."""
        cls.numbers = generate_test_numbers(count=100, seed=456)
        cls.sequential_result = factorize_all_sequential(cls.numbers)

    def test_various_worker_counts(self):
        """Test that different worker counts produce identical results."""
        max_workers = mp.cpu_count()
        worker_counts = [w for w in [1, 2, 4, 8] if w <= max_workers]

        results = {}
        for num_workers in worker_counts:
            result = parallel_factorize_imap_unordered(
                self.numbers, num_workers=num_workers, chunk_size=1
            )
            results[num_workers] = result

        # All results should match sequential
        for num_workers in worker_counts:
            for i, n in enumerate(self.numbers):
                self.assertEqual(
                    self.sequential_result.factors[i],
                    results[num_workers].factors[i],
                    f"Worker count {num_workers} produced different result for {n}"
                )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_single_number(self):
        """Test parallelization with single number."""
        numbers = [12345]
        seq_result = factorize_all_sequential(numbers)
        para_result = parallel_factorize_imap_unordered(numbers, chunk_size=1)

        self.assertEqual(seq_result.factors[0], para_result.factors[0])

    def test_all_same_numbers(self):
        """Test parallelization with all identical numbers."""
        numbers = [84] * 50
        seq_result = factorize_all_sequential(numbers)
        para_result = parallel_factorize_imap_unordered(numbers, chunk_size=1)

        for i in range(len(numbers)):
            self.assertEqual(seq_result.factors[i], para_result.factors[i])

    def test_all_primes(self):
        """Test parallelization with all prime numbers."""
        numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        seq_result = factorize_all_sequential(numbers)
        para_result = parallel_factorize_imap_unordered(numbers, chunk_size=1)

        for i in range(len(numbers)):
            self.assertEqual(seq_result.factors[i], para_result.factors[i])

    def test_powers_sequence(self):
        """Test parallelization with powers of 2."""
        numbers = [2**i for i in range(1, 30)]
        seq_result = factorize_all_sequential(numbers)
        para_result = parallel_factorize_imap_unordered(numbers, chunk_size=1)

        for i in range(len(numbers)):
            self.assertEqual(seq_result.factors[i], para_result.factors[i])


class TestRaceConditions(unittest.TestCase):
    """Test for potential race conditions by running multiple times."""

    def test_repeated_execution(self):
        """Run parallelization multiple times to check for race conditions."""
        numbers = generate_test_numbers(count=100, seed=789)
        seq_result = factorize_all_sequential(numbers)

        # Run parallel version multiple times
        for run in range(5):
            para_result = parallel_factorize_imap_unordered(numbers, chunk_size=1)

            for i, n in enumerate(numbers):
                self.assertEqual(
                    seq_result.factors[i],
                    para_result.factors[i],
                    f"Race condition detected on run {run + 1} for number {n}"
                )


def run_quick_test():
    """Run a quick sanity check."""
    print("Running quick correctness test...")

    numbers = generate_test_numbers(count=100, seed=42)

    print("  Running sequential...", end=" ", flush=True)
    seq_result = factorize_all_sequential(numbers)
    print("done")

    print("  Running parallel...", end=" ", flush=True)
    para_result = parallel_factorize_imap_unordered(numbers, chunk_size=1)
    print("done")

    print("  Comparing results...", end=" ", flush=True)
    mismatches = 0
    for i, n in enumerate(numbers):
        if seq_result.factors[i] != para_result.factors[i]:
            mismatches += 1
            print(f"\n  MISMATCH for {n}:")
            print(f"    Sequential: {seq_result.factors[i]}")
            print(f"    Parallel:   {para_result.factors[i]}")

    if mismatches == 0:
        print("PASSED - All results match!")
    else:
        print(f"\nFAILED - {mismatches} mismatches found")

    return mismatches == 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        # Run full test suite
        unittest.main(verbosity=2)
