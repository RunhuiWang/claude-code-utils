#!/usr/bin/env python3
"""
Correctness Tests for Parallel TF-IDF Implementation

This module provides comprehensive tests to ensure the parallel
implementation produces identical results to the sequential version.
"""

import unittest
import math
import sys
from typing import List, Set

from document_generator import generate_corpus, Document
from sequential import (
    tokenize,
    compute_term_frequencies,
    build_tfidf_index_sequential,
    search_sequential,
    batch_search_sequential,
)
from parallel_solution import (
    build_tfidf_index_parallel,
    build_tfidf_index_parallel_futures,
    batch_search_parallel,
)


class TestTokenization(unittest.TestCase):
    """Test tokenization functions."""

    def test_basic_tokenization(self):
        """Test basic text tokenization."""
        text = "The quick brown fox jumps over the lazy dog"
        tokens = tokenize(text)

        # Should remove stop words like "the", "over"
        self.assertNotIn("the", tokens)
        self.assertNotIn("over", tokens)

        # Should keep content words
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)
        self.assertIn("fox", tokens)

    def test_case_insensitivity(self):
        """Test that tokenization is case-insensitive."""
        text = "Machine Learning ALGORITHMS neural Networks"
        tokens = tokenize(text)

        self.assertIn("machine", tokens)
        self.assertIn("learning", tokens)
        self.assertIn("algorithms", tokens)
        self.assertIn("neural", tokens)
        self.assertIn("networks", tokens)

    def test_punctuation_handling(self):
        """Test that punctuation is properly handled."""
        text = "Hello, world! This is a test. How are you?"
        tokens = tokenize(text)

        # Should not include punctuation
        for token in tokens:
            self.assertTrue(token.isalpha())

    def test_empty_input(self):
        """Test tokenization of empty input."""
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   "), [])

    def test_single_letters_filtered(self):
        """Test that single letters are filtered out."""
        text = "I a o u the and or"
        tokens = tokenize(text)
        for token in tokens:
            self.assertGreaterEqual(len(token), 2)


class TestTermFrequency(unittest.TestCase):
    """Test term frequency computation."""

    def test_basic_tf(self):
        """Test basic term frequency calculation."""
        tokens = ["apple", "banana", "apple", "cherry"]
        tf = compute_term_frequencies(tokens)

        self.assertAlmostEqual(tf["apple"], 2/4)
        self.assertAlmostEqual(tf["banana"], 1/4)
        self.assertAlmostEqual(tf["cherry"], 1/4)

    def test_empty_tokens(self):
        """Test TF with empty token list."""
        tf = compute_term_frequencies([])
        self.assertEqual(tf, {})

    def test_single_token(self):
        """Test TF with single token."""
        tf = compute_term_frequencies(["word"])
        self.assertEqual(tf, {"word": 1.0})


class TestIndexBuilding(unittest.TestCase):
    """Test index building correctness."""

    @classmethod
    def setUpClass(cls):
        """Generate test corpus once."""
        cls.documents = generate_corpus(500, seed=42)
        cls.seq_result = build_tfidf_index_sequential(cls.documents)

    def test_vocabulary_match(self):
        """Test that parallel index has same vocabulary."""
        para_result = build_tfidf_index_parallel(self.documents)

        self.assertEqual(
            self.seq_result.index.vocabulary,
            para_result.index.vocabulary
        )

    def test_document_frequencies_match(self):
        """Test that document frequencies match."""
        para_result = build_tfidf_index_parallel(self.documents)

        for term in self.seq_result.index.vocabulary:
            seq_df = self.seq_result.index.document_frequencies.get(term, 0)
            para_df = para_result.index.document_frequencies.get(term, 0)
            self.assertEqual(seq_df, para_df, f"DF mismatch for term '{term}'")

    def test_idf_values_match(self):
        """Test that IDF values match."""
        para_result = build_tfidf_index_parallel(self.documents)

        for term in self.seq_result.index.vocabulary:
            seq_idf = self.seq_result.index.idf.get(term, 0)
            para_idf = para_result.index.idf.get(term, 0)
            self.assertAlmostEqual(
                seq_idf, para_idf, places=6,
                msg=f"IDF mismatch for term '{term}'"
            )

    def test_inverted_index_match(self):
        """Test that inverted index posting lists match."""
        para_result = build_tfidf_index_parallel(self.documents)

        for term in self.seq_result.index.vocabulary:
            seq_postings = dict(self.seq_result.index.inverted_index.get(term, []))
            para_postings = dict(para_result.index.inverted_index.get(term, []))

            self.assertEqual(
                set(seq_postings.keys()),
                set(para_postings.keys()),
                f"Posting list doc_ids mismatch for term '{term}'"
            )

            for doc_id in seq_postings:
                self.assertAlmostEqual(
                    seq_postings[doc_id],
                    para_postings[doc_id],
                    places=6,
                    msg=f"TF-IDF score mismatch for term '{term}', doc {doc_id}"
                )

    def test_document_vectors_match(self):
        """Test that document vectors match."""
        para_result = build_tfidf_index_parallel(self.documents)

        for doc_id in range(len(self.documents)):
            seq_vec = self.seq_result.index.doc_vectors.get(doc_id, {})
            para_vec = para_result.index.doc_vectors.get(doc_id, {})

            self.assertEqual(
                set(seq_vec.keys()),
                set(para_vec.keys()),
                f"Document vector terms mismatch for doc {doc_id}"
            )

            for term in seq_vec:
                self.assertAlmostEqual(
                    seq_vec[term],
                    para_vec[term],
                    places=6,
                    msg=f"Vector value mismatch for doc {doc_id}, term '{term}'"
                )

    def test_document_norms_match(self):
        """Test that document norms match."""
        para_result = build_tfidf_index_parallel(self.documents)

        for doc_id in range(len(self.documents)):
            seq_norm = self.seq_result.index.doc_norms.get(doc_id, 0)
            para_norm = para_result.index.doc_norms.get(doc_id, 0)

            self.assertAlmostEqual(
                seq_norm, para_norm, places=6,
                msg=f"Norm mismatch for doc {doc_id}"
            )

    def test_futures_implementation(self):
        """Test that Futures implementation matches sequential."""
        futures_result = build_tfidf_index_parallel_futures(self.documents)

        # Check vocabulary
        self.assertEqual(
            self.seq_result.index.vocabulary,
            futures_result.index.vocabulary
        )

        # Check IDF values
        for term in self.seq_result.index.vocabulary:
            seq_idf = self.seq_result.index.idf.get(term, 0)
            futures_idf = futures_result.index.idf.get(term, 0)
            self.assertAlmostEqual(seq_idf, futures_idf, places=6)


class TestSearch(unittest.TestCase):
    """Test search functionality correctness."""

    @classmethod
    def setUpClass(cls):
        """Build index once for all search tests."""
        cls.documents = generate_corpus(500, seed=42)
        cls.seq_result = build_tfidf_index_sequential(cls.documents)
        cls.para_result = build_tfidf_index_parallel(cls.documents)

    def test_single_search_match(self):
        """Test that single query search results match."""
        queries = [
            "machine learning algorithm",
            "database optimization",
            "clinical trial treatment",
            "market analysis"
        ]

        for query in queries:
            seq_results = search_sequential(
                query, self.seq_result.index, top_k=10, documents=self.documents
            )
            para_results = search_sequential(
                query, self.para_result.index, top_k=10, documents=self.documents
            )

            # Same number of results
            self.assertEqual(
                len(seq_results), len(para_results),
                f"Different result counts for query '{query}'"
            )

            # Same doc_ids and scores
            for seq_res, para_res in zip(seq_results, para_results):
                self.assertEqual(seq_res.doc_id, para_res.doc_id)
                self.assertAlmostEqual(seq_res.score, para_res.score, places=6)

    def test_batch_search_match(self):
        """Test that batch search results match between implementations."""
        queries = [
            "machine learning neural network",
            "database query performance",
            "scientific research methodology",
            "software development testing",
            "network security protocol"
        ]

        seq_results = batch_search_sequential(
            queries, self.seq_result.index, top_k=10, documents=self.documents
        )
        para_results, _ = batch_search_parallel(
            queries, self.para_result.index, top_k=10, documents=self.documents
        )

        for i, (seq_query_results, para_query_results) in enumerate(zip(seq_results, para_results)):
            self.assertEqual(
                len(seq_query_results), len(para_query_results),
                f"Different result counts for query {i}"
            )

            for seq_res, para_res in zip(seq_query_results, para_query_results):
                self.assertEqual(seq_res.doc_id, para_res.doc_id)
                self.assertAlmostEqual(seq_res.score, para_res.score, places=6)

    def test_empty_query(self):
        """Test search with empty query."""
        results = search_sequential("", self.seq_result.index)
        self.assertEqual(results, [])

    def test_unknown_terms_query(self):
        """Test search with unknown terms."""
        results = search_sequential(
            "xyzabc123 nonexistent terms",
            self.seq_result.index
        )
        self.assertEqual(results, [])


class TestWorkerVariations(unittest.TestCase):
    """Test with different worker counts."""

    @classmethod
    def setUpClass(cls):
        """Generate test corpus."""
        cls.documents = generate_corpus(300, seed=123)
        cls.seq_result = build_tfidf_index_sequential(cls.documents)

    def test_single_worker(self):
        """Test parallel with single worker."""
        para_result = build_tfidf_index_parallel(self.documents, num_workers=1)
        self.assertEqual(
            self.seq_result.index.vocabulary,
            para_result.index.vocabulary
        )

    def test_two_workers(self):
        """Test parallel with two workers."""
        para_result = build_tfidf_index_parallel(self.documents, num_workers=2)
        self.assertEqual(
            self.seq_result.index.vocabulary,
            para_result.index.vocabulary
        )

    def test_four_workers(self):
        """Test parallel with four workers."""
        para_result = build_tfidf_index_parallel(self.documents, num_workers=4)
        self.assertEqual(
            self.seq_result.index.vocabulary,
            para_result.index.vocabulary
        )


class TestChunkSizeVariations(unittest.TestCase):
    """Test with different chunk sizes."""

    @classmethod
    def setUpClass(cls):
        """Generate test corpus."""
        cls.documents = generate_corpus(300, seed=456)
        cls.seq_result = build_tfidf_index_sequential(cls.documents)

    def test_small_chunks(self):
        """Test with small chunk size."""
        para_result = build_tfidf_index_parallel(self.documents, chunk_size=10)
        self._verify_match(para_result)

    def test_medium_chunks(self):
        """Test with medium chunk size."""
        para_result = build_tfidf_index_parallel(self.documents, chunk_size=50)
        self._verify_match(para_result)

    def test_large_chunks(self):
        """Test with large chunk size."""
        para_result = build_tfidf_index_parallel(self.documents, chunk_size=200)
        self._verify_match(para_result)

    def _verify_match(self, para_result):
        """Verify parallel result matches sequential."""
        self.assertEqual(
            self.seq_result.index.vocabulary,
            para_result.index.vocabulary
        )
        for term in self.seq_result.index.vocabulary:
            self.assertAlmostEqual(
                self.seq_result.index.idf[term],
                para_result.index.idf[term],
                places=6
            )


class TestRaceConditions(unittest.TestCase):
    """Test for race conditions by running multiple times."""

    def test_repeated_execution(self):
        """Run parallel indexing multiple times to check for race conditions."""
        documents = generate_corpus(200, seed=789)
        seq_result = build_tfidf_index_sequential(documents)

        for run in range(3):
            para_result = build_tfidf_index_parallel(documents)

            self.assertEqual(
                seq_result.index.vocabulary,
                para_result.index.vocabulary,
                f"Vocabulary mismatch on run {run + 1}"
            )

            for term in seq_result.index.vocabulary:
                self.assertAlmostEqual(
                    seq_result.index.idf[term],
                    para_result.index.idf[term],
                    places=6,
                    msg=f"IDF mismatch on run {run + 1} for term '{term}'"
                )


def run_quick_test():
    """Run a quick sanity check."""
    print("Running quick correctness test...")

    documents = generate_corpus(200, seed=42)

    print("  Building sequential index...", end=" ", flush=True)
    seq_result = build_tfidf_index_sequential(documents)
    print("done")

    print("  Building parallel index...", end=" ", flush=True)
    para_result = build_tfidf_index_parallel(documents)
    print("done")

    print("  Comparing vocabularies...", end=" ", flush=True)
    if seq_result.index.vocabulary != para_result.index.vocabulary:
        print("FAILED - vocabulary mismatch")
        return False
    print("match")

    print("  Comparing IDF values...", end=" ", flush=True)
    for term in seq_result.index.vocabulary:
        seq_idf = seq_result.index.idf[term]
        para_idf = para_result.index.idf[term]
        if abs(seq_idf - para_idf) > 1e-6:
            print(f"FAILED - IDF mismatch for '{term}'")
            return False
    print("match")

    print("  Comparing search results...", end=" ", flush=True)
    queries = ["machine learning", "database optimization", "clinical treatment"]
    for query in queries:
        seq_res = search_sequential(query, seq_result.index, top_k=5, documents=documents)
        para_res = search_sequential(query, para_result.index, top_k=5, documents=documents)

        if len(seq_res) != len(para_res):
            print(f"FAILED - result count mismatch for '{query}'")
            return False

        for s, p in zip(seq_res, para_res):
            if s.doc_id != p.doc_id or abs(s.score - p.score) > 1e-6:
                print(f"FAILED - result mismatch for '{query}'")
                return False
    print("match")

    print("\nAll quick tests PASSED!")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        unittest.main(verbosity=2)
