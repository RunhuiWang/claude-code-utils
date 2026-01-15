#!/usr/bin/env python3
"""
Sequential Prime Factorization - Baseline Implementation

This module provides a sequential implementation of prime factorization
for a list of numbers. It serves as the baseline for correctness testing
and performance comparison with the parallel implementation.
"""

import random
import time
from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class FactorizationResult:
    """Result of factorizing all numbers."""
    factors: Dict[int, Dict[int, int]]  # number -> {prime: power}
    prime_cache: Set[int]  # All primes discovered during factorization
    elapsed_time: float


def is_prime(n: int, prime_cache: Set[int]) -> bool:
    """Check if n is prime, using cache for known primes."""
    if n < 2:
        return False
    if n in prime_cache:
        return True
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def factorize(n: int, prime_cache: Set[int]) -> Dict[int, int]:
    """
    Factorize a number into its prime factors.

    Args:
        n: The number to factorize
        prime_cache: Set of known primes (will be updated with new discoveries)

    Returns:
        Dictionary mapping prime factors to their powers
    """
    if n < 2:
        return {}

    original_n = n
    factors = {}

    # Check for factor of 2
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2

    if n == 1:
        prime_cache.add(2)
        return factors

    # Check odd factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
            # i is a prime factor, add to cache
            prime_cache.add(i)
        i += 2

    # If n is still greater than 1, it's a prime factor
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
        prime_cache.add(n)

    # Add 2 to cache if it was a factor
    if 2 in factors:
        prime_cache.add(2)

    return factors


def generate_test_numbers(
    count: int = 1000,
    seed: int = 42,
    include_hard_cases: bool = True
) -> List[int]:
    """
    Generate test numbers with deliberately unbalanced complexity.

    Creates a mix of:
    - 70% easy numbers (small, quick to factor)
    - 20% medium numbers (moderate factorization time)
    - 10% hard numbers (large semi-primes, very slow)
    """
    random.seed(seed)
    numbers = []

    # Generate easy numbers (70%)
    easy_count = int(count * 0.7)
    for _ in range(easy_count):
        numbers.append(random.randint(2, 10_000))

    # Generate medium numbers (20%)
    medium_count = int(count * 0.2)
    for _ in range(medium_count):
        numbers.append(random.randint(10_000, 1_000_000))

    # Generate hard numbers - semi-primes (10%)
    if include_hard_cases:
        hard_count = count - easy_count - medium_count
        hard_primes = generate_large_primes(hard_count * 2, seed + 1)
        for i in range(0, len(hard_primes) - 1, 2):
            # Semi-prime = product of two large primes
            semi_prime = hard_primes[i] * hard_primes[i + 1]
            numbers.append(semi_prime)

    # Shuffle to distribute hard cases throughout
    random.seed(seed + 100)
    random.shuffle(numbers)

    return numbers


def generate_large_primes(count: int, seed: int) -> List[int]:
    """Generate large prime numbers for creating semi-primes."""
    random.seed(seed)
    primes = []

    # Generate primes in range 10,000,000 to 20,000,000
    # These create semi-primes (10^14 - 10^15 range) that take ~0.4s each to factor
    # This makes parallelization beneficial for load balancing
    start = 10_000_000
    end = 20_000_000

    # Sample random candidates rather than checking all
    candidates = random.sample(range(start, end), min(count * 100, end - start))

    for candidate in candidates:
        if is_prime_simple(candidate):
            primes.append(candidate)
            if len(primes) >= count:
                break

    # If we didn't find enough, search sequentially
    if len(primes) < count:
        for n in range(start, end):
            if is_prime_simple(n) and n not in primes:
                primes.append(n)
                if len(primes) >= count:
                    break

    return primes


def is_prime_simple(n: int) -> bool:
    """Simple primality check without cache."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def factorize_all_sequential(numbers: List[int]) -> FactorizationResult:
    """
    Factorize all numbers sequentially.

    This is the baseline implementation that the parallel version
    must match for correctness.
    """
    start_time = time.perf_counter()

    prime_cache: Set[int] = set()
    factors: Dict[int, Dict[int, int]] = {}

    for i, n in enumerate(numbers):
        factors[i] = factorize(n, prime_cache)

    elapsed = time.perf_counter() - start_time

    return FactorizationResult(
        factors=factors,
        prime_cache=prime_cache,
        elapsed_time=elapsed
    )


def verify_factorization(n: int, factors: Dict[int, int]) -> bool:
    """Verify that the factorization is correct."""
    if n < 2:
        return factors == {}

    product = 1
    for prime, power in factors.items():
        product *= prime ** power

    return product == n


def main():
    """Run sequential factorization and display results."""
    print("=" * 60)
    print("Sequential Prime Factorization")
    print("=" * 60)

    # Generate test numbers
    print("\nGenerating test numbers...")
    numbers = generate_test_numbers(count=500, seed=42)
    print(f"Generated {len(numbers)} numbers")

    # Show distribution
    easy = sum(1 for n in numbers if n <= 10_000)
    medium = sum(1 for n in numbers if 10_000 < n <= 1_000_000)
    hard = sum(1 for n in numbers if n > 1_000_000)
    print(f"Distribution: {easy} easy, {medium} medium, {hard} hard")

    # Run factorization
    print("\nRunning sequential factorization...")
    result = factorize_all_sequential(numbers)

    print(f"\nCompleted in {result.elapsed_time:.3f} seconds")
    print(f"Discovered {len(result.prime_cache)} unique primes")

    # Verify a sample of results
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
