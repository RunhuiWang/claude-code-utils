# Parallel Prime Factorization Challenge

## Problem Description

This task involves parallelizing a prime factorization algorithm with several challenges that make naive parallelization ineffective:

### Why This Task is Challenging

1. **Unbalanced Workload**: The input contains numbers of vastly different sizes:
   - Small numbers (< 1000) factor instantly
   - Medium numbers (10^6 - 10^9) take milliseconds
   - Large semi-primes (product of two large primes ~10^6 each) take seconds

   Simple static partitioning will leave most workers idle while one struggles with a large semi-prime.

2. **Shared Prime Cache**: Workers share a cache of discovered primes to avoid redundant computation:
   - When a worker discovers a prime during factorization, it adds it to the shared cache
   - Other workers can use cached primes to speed up their factorizations
   - This creates race conditions when multiple workers try to update the cache simultaneously

3. **Work Stealing**: Efficient parallelization requires dynamic load balancing:
   - Workers that finish early should steal work from busy workers
   - This requires careful synchronization to avoid duplicate work or missed numbers

4. **Result Aggregation**: All factorization results must be collected into a shared dictionary:
   - Race conditions can occur when multiple workers finish simultaneously
   - Results must be consistent and complete

## Task Structure

- `sequential.py` - Baseline sequential implementation
- `parallel_solution.py` - Parallelized solution using multiprocessing
- `benchmark.py` - Performance comparison between sequential and parallel
- `test_correctness.py` - Verify parallel results match sequential results

## The Algorithm

For each number N, find all prime factors with their multiplicities:
- 84 → {2: 2, 3: 1, 7: 1} (since 84 = 2² × 3 × 7)

The factorization uses trial division with optimization:
1. Check divisibility by 2
2. Check odd divisors up to √N
3. Use cached primes when available to skip non-prime divisors

## Input Data Characteristics

The input is deliberately crafted to create load imbalance:
- 70% small numbers (1-10,000) - very fast to factor
- 20% medium numbers (10,000-1,000,000) - moderate time
- 10% large semi-primes (product of two primes ~1,000-100,000) - very slow

## Success Criteria

1. **Correctness**: Parallel results must exactly match sequential results
2. **Speedup**: Should achieve near-linear speedup with CPU cores (e.g., 4x on 4 cores)
3. **No Race Conditions**: No lost or corrupted results
4. **Efficient Load Balancing**: All cores should stay busy until completion

## Running the Code

```bash
# Run sequential baseline
python sequential.py

# Run parallel solution
python parallel_solution.py

# Run benchmark comparison
python benchmark.py

# Verify correctness
python test_correctness.py
```

## Key Parallelization Techniques Used

1. **Work Queue with Chunking**: Instead of static partitioning, use a shared queue
2. **Manager for Shared State**: Use `multiprocessing.Manager` for the prime cache
3. **Lock-free Result Collection**: Use `Manager().dict()` for thread-safe result storage
4. **Dynamic Chunk Sizing**: Smaller chunks for better load balancing
5. **Process Pool**: Use `ProcessPoolExecutor` for efficient worker management
