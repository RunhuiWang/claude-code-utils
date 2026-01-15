#!/usr/bin/env python3
"""
Tests for Parallel TF-IDF Implementation

Tests correctness and performance of the parallel implementation.
"""

import sys
import time
import random
import pytest

sys.path.insert(0, '/app/workspace')

from document_generator import generate_corpus
from sequential import (
    build_tfidf_index_sequential,
    batch_search_sequential,
)


class TestParallelImplementation:
    """Test that parallel implementation exists and has correct interface."""

    def test_parallel_solution_exists(self):
        """Test that parallel_solution.py exists and can be imported."""
        try:
            from parallel_solution import (
                build_tfidf_index_parallel,
                batch_search_parallel,
            )
        except ImportError as e:
            pytest.fail(f"Could not import parallel_solution: {e}")

    def test_parallel_index_function_signature(self):
        """Test that build_tfidf_index_parallel has correct signature."""
        from parallel_solution import build_tfidf_index_parallel
        import inspect
        sig = inspect.signature(build_tfidf_index_parallel)
        params = list(sig.parameters.keys())
        assert 'documents' in params, "Missing 'documents' parameter"

    def test_parallel_search_function_signature(self):
        """Test that batch_search_parallel has correct signature."""
        from parallel_solution import batch_search_parallel
        import inspect
        sig = inspect.signature(batch_search_parallel)
        params = list(sig.parameters.keys())
        assert 'queries' in params, "Missing 'queries' parameter"
        assert 'index' in params, "Missing 'index' parameter"


class TestCorrectnessSmall:
    """Test correctness with small dataset."""

    @pytest.fixture(scope="class")
    def small_corpus(self):
        """Generate small test corpus."""
        return generate_corpus(500, seed=42)

    @pytest.fixture(scope="class")
    def sequential_index(self, small_corpus):
        """Build sequential index for comparison."""
        return build_tfidf_index_sequential(small_corpus)

    def test_vocabulary_match(self, small_corpus, sequential_index):
        """Test that parallel index has same vocabulary."""
        from parallel_solution import build_tfidf_index_parallel

        para_result = build_tfidf_index_parallel(small_corpus, num_workers=4)

        assert sequential_index.index.vocabulary == para_result.index.vocabulary, \
            "Vocabulary mismatch between sequential and parallel"

    def test_idf_values_match(self, small_corpus, sequential_index):
        """Test that IDF values match."""
        from parallel_solution import build_tfidf_index_parallel

        para_result = build_tfidf_index_parallel(small_corpus, num_workers=4)

        for term in sequential_index.index.vocabulary:
            seq_idf = sequential_index.index.idf.get(term, 0)
            para_idf = para_result.index.idf.get(term, 0)
            assert abs(seq_idf - para_idf) < 1e-6, \
                f"IDF mismatch for term '{term}': {seq_idf} vs {para_idf}"

    def test_search_results_match(self, small_corpus, sequential_index):
        """Test that search results match."""
        from parallel_solution import build_tfidf_index_parallel, batch_search_parallel

        para_result = build_tfidf_index_parallel(small_corpus, num_workers=4)

        queries = [
            "machine learning algorithm",
            "database optimization",
            "clinical trial treatment"
        ]

        seq_results = batch_search_sequential(
            queries, sequential_index.index, top_k=10, documents=small_corpus
        )
        para_results, _ = batch_search_parallel(
            queries, para_result.index, top_k=10, num_workers=4, documents=small_corpus
        )

        for i, (seq_res, para_res) in enumerate(zip(seq_results, para_results)):
            assert len(seq_res) == len(para_res), \
                f"Different result count for query {i}"
            for j, (s, p) in enumerate(zip(seq_res, para_res)):
                assert s.doc_id == p.doc_id, \
                    f"Doc ID mismatch for query {i}, result {j}"
                assert abs(s.score - p.score) < 1e-6, \
                    f"Score mismatch for query {i}, result {j}"


class TestCorrectnessLarge:
    """Test correctness with larger dataset."""

    @pytest.fixture(scope="class")
    def large_corpus(self):
        """Generate larger test corpus."""
        return generate_corpus(5000, seed=123)

    @pytest.fixture(scope="class")
    def sequential_index_large(self, large_corpus):
        """Build sequential index for comparison."""
        return build_tfidf_index_sequential(large_corpus)

    def test_large_vocabulary_match(self, large_corpus, sequential_index_large):
        """Test vocabulary match with larger corpus."""
        from parallel_solution import build_tfidf_index_parallel

        para_result = build_tfidf_index_parallel(large_corpus, num_workers=4)

        assert sequential_index_large.index.vocabulary == para_result.index.vocabulary

    def test_large_search_results_match(self, large_corpus, sequential_index_large):
        """Test search results match with larger corpus."""
        from parallel_solution import build_tfidf_index_parallel, batch_search_parallel

        para_result = build_tfidf_index_parallel(large_corpus, num_workers=4)

        # Generate variable-length queries
        random.seed(456)
        base_terms = ['machine', 'learning', 'algorithm', 'neural', 'network',
                      'database', 'optimization', 'performance', 'clinical', 'trial']

        queries = []
        for _ in range(50):
            length = random.randint(1, 10)
            query = ' '.join(random.choices(base_terms, k=length))
            queries.append(query)

        seq_results = batch_search_sequential(
            queries, sequential_index_large.index, top_k=10, documents=large_corpus
        )
        para_results, _ = batch_search_parallel(
            queries, para_result.index, top_k=10, num_workers=4, documents=large_corpus
        )

        for i, (seq_res, para_res) in enumerate(zip(seq_results, para_results)):
            assert len(seq_res) == len(para_res), f"Result count mismatch for query {i}"


class TestPerformanceIndexing:
    """Test index building performance."""

    @pytest.fixture(scope="class")
    def perf_corpus(self):
        """Generate corpus for performance testing."""
        return generate_corpus(10000, seed=789)

    def test_index_speedup(self, perf_corpus):
        """Test that parallel indexing achieves speedup."""
        from parallel_solution import build_tfidf_index_parallel

        # Sequential baseline
        start = time.perf_counter()
        seq_result = build_tfidf_index_sequential(perf_corpus)
        seq_time = time.perf_counter() - start

        # Parallel with 4 workers
        start = time.perf_counter()
        para_result = build_tfidf_index_parallel(perf_corpus, num_workers=4)
        para_time = time.perf_counter() - start

        speedup = seq_time / para_time

        print(f"\nIndex Building Performance:")
        print(f"  Sequential: {seq_time:.2f}s")
        print(f"  Parallel (4 workers): {para_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Require at least 1.2x speedup
        assert speedup >= 1.2, \
            f"Insufficient speedup: {speedup:.2f}x (required: 1.2x)"


class TestPerformanceSearch:
    """Test batch search performance."""

    @pytest.fixture(scope="class")
    def search_setup(self):
        """Set up corpus and index for search testing."""
        corpus = generate_corpus(10000, seed=101)

        from parallel_solution import build_tfidf_index_parallel
        index_result = build_tfidf_index_parallel(corpus, num_workers=4)

        # Generate variable-length queries
        random.seed(202)
        base_terms = ['machine', 'learning', 'algorithm', 'neural', 'network',
                      'database', 'optimization', 'performance', 'clinical', 'trial',
                      'market', 'analysis', 'investment', 'research', 'methodology']

        queries = []
        for _ in range(200):
            length = random.randint(1, 20)
            query = ' '.join(random.choices(base_terms, k=length))
            queries.append(query)

        return corpus, index_result.index, queries

    def test_search_speedup(self, search_setup):
        """Test that parallel search achieves speedup."""
        from parallel_solution import batch_search_parallel

        corpus, index, queries = search_setup

        # Sequential baseline
        start = time.perf_counter()
        seq_results = batch_search_sequential(queries, index, top_k=10, documents=corpus)
        seq_time = time.perf_counter() - start

        # Parallel with 8 workers
        start = time.perf_counter()
        para_results, _ = batch_search_parallel(
            queries, index, top_k=10, num_workers=8, documents=corpus
        )
        para_time = time.perf_counter() - start

        speedup = seq_time / para_time

        print(f"\nBatch Search Performance (200 queries):")
        print(f"  Sequential: {seq_time:.2f}s")
        print(f"  Parallel (8 workers): {para_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Require at least 3x speedup for batch search
        assert speedup >= 3.0, \
            f"Insufficient speedup: {speedup:.2f}x (required: 3.0x)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
