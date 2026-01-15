# Parallel TF-IDF Similarity Search

## Real-World Application

This task implements a **TF-IDF (Term Frequency-Inverse Document Frequency) based similarity search engine** with an inverted index - the foundation of modern search engines like Elasticsearch, Solr, and Google.

### Use Cases
- Document search engines
- Plagiarism detection
- Content recommendation systems
- Question-answering retrieval
- Duplicate detection

## Why This Task is Challenging for Parallelization

### 1. Mixed Parallel Patterns
- **Document Processing**: Embarrassingly parallel (each doc independent)
- **Vocabulary Building**: Requires reduction/merging across workers
- **IDF Calculation**: Needs global document frequency counts
- **Index Merging**: Race conditions when combining posting lists

### 2. Unbalanced Workload
- Documents vary greatly in length (10 to 10,000+ words)
- Tokenization time scales with document length
- Some terms appear in millions of documents, others in few

### 3. Memory Considerations
- Large vocabularies (100K+ unique terms)
- Posting lists can be huge for common terms
- Need efficient data structures for parallel access

### 4. Synchronization Challenges
- Building global term->document mappings from local results
- Computing document frequencies across all workers
- Merging inverted indices without data loss

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Corpus                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Parallel Document Processing                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker N │        │
│  │Tokenize │  │Tokenize │  │Tokenize │  │Tokenize │        │
│  │Count TF │  │Count TF │  │Count TF │  │Count TF │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
└───────┼────────────┼────────────┼────────────┼──────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Merge & Build Global Index                   │
│  • Combine vocabularies from all workers                     │
│  • Calculate document frequencies (DF)                       │
│  • Compute IDF = log(N / DF)                                │
│  • Build inverted index: term -> [(doc_id, tf-idf), ...]    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Similarity Search                          │
│  Query: "machine learning algorithms"                        │
│  1. Tokenize query                                          │
│  2. Look up posting lists for each term                     │
│  3. Compute cosine similarity scores                        │
│  4. Return top-K results                                    │
└─────────────────────────────────────────────────────────────┘
```

## Files

- `document_generator.py` - Synthetic corpus generator with realistic text
- `sequential.py` - Baseline sequential TF-IDF implementation
- `parallel_solution.py` - Parallelized indexing and search
- `benchmark.py` - Performance comparison suite
- `test_correctness.py` - Correctness verification

## TF-IDF Formula

```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

Where:
- TF(t, d) = frequency of term t in document d / total terms in d
- IDF(t, D) = log(N / DF(t)) + 1
- N = total number of documents
- DF(t) = number of documents containing term t
```

## Cosine Similarity

```
similarity(q, d) = (q · d) / (||q|| × ||d||)

Where q and d are TF-IDF vectors for query and document
```

## Running the Code

```bash
# Generate test corpus (10K documents)
python document_generator.py --num-docs 10000 --output corpus.json

# Run sequential baseline
python sequential.py --corpus corpus.json

# Run parallel solution
python parallel_solution.py --corpus corpus.json

# Run benchmarks
python benchmark.py

# Verify correctness
python test_correctness.py
```

## Performance Insights

This task demonstrates important lessons about parallelization:

### When Parallelization Helps
- **Very large corpora** (50K+ documents): Speedup improves with scale
- **Batch search** (100+ queries): Queries are embarrassingly parallel
- **Compute-intensive operations**: Prime factorization, matrix multiplication

### When Parallelization Hurts
- **Small datasets**: Process startup overhead dominates
- **Memory-bound operations**: Python dict/string operations are already fast
- **High data transfer**: Passing large dicts between processes is expensive

### Benchmark Results (20K documents, 16 cores)

#### Index Building

| Workers | Time | Speedup |
|---------|------|---------|
| Sequential | 9.51s | 1.00x |
| 4 workers | 7.11s | **1.34x** |

#### Batch Search (Variable-length queries 1-20 words)

| Workers | 200 Queries | Speedup | 500 Queries | Speedup |
|---------|-------------|---------|-------------|---------|
| Sequential | 8.84s | 1.00x | 19.38s | 1.00x |
| 4 workers | 2.82s | 3.13x | 5.80s | **3.34x** |
| 8 workers | 2.47s | **3.58x** | 3.69s | **5.25x** |
| 16 workers | 2.91s | 3.03x | 3.11s | **6.24x** |

### Key Optimization: Worker Pool Initializer

The batch search uses a **Pool initializer** pattern to load the index once per worker:

```python
with Pool(
    processes=num_workers,
    initializer=_init_search_worker,
    initargs=(index.inverted_index, index.doc_vectors, ...)
) as pool:
    # Only queries passed - no index transfer per query!
    results = pool.map(_search_query_worker, queries)
```

**Before optimization**: 23.9s (0.38x slower than sequential)
**After optimization**: 2.47s (3.58x faster than sequential)

**Key insight**: Minimizing inter-process data transfer is critical for parallel performance.

## Key Parallelization Techniques

1. **MapReduce Pattern**: Map documents to local indices, reduce to global index
2. **Chunked Processing**: Process documents in batches to reduce overhead
3. **Worker Pool Initializer**: Load shared data once per worker, not per task
4. **Lock-free Merging**: Use process-safe data structures for index merging
5. **Parallel Search**: Distribute queries across workers for batch search
