# REPLUG QA – CS4051 Information Retrieval Project

Open‑domain question answering with dense retrieval and λ‑weighted ensemble (REPLUG Section 3.2).

## Paper

Weijia Shi et al., "REPLUG: Retrieval-Augmented Black-Box Language Models", arXiv:2301.12652 (2023)  
[https://arxiv.org/abs/2301.12652](https://arxiv.org/abs/2301.12652)

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API key**
   ```bash
   echo "COHERE_API_KEY=your_key_here" > .env
   ```
   Get a free API key from [Cohere](https://cohere.com) – the free tier provides `logprobs` access.

3. **Generate large data files (first run only)**  
   The repository does not include the large corpus files (`passages.json`, `embeddings.npy`, `faiss.index`).  
   Run the provided notebook to create them:
   ```bash
   # Open notebooks/replug-kaggle-v2.ipynb in Kaggle (or locally with GPU)
   # It downloads 150k Wikipedia passages, computes Contriever embeddings, and builds the FAISS index.
   # Place the generated files in the `data/` directory.
   ```

4. **Run the pipeline**
   ```bash
   python main.py
   ```

5. **Check outputs**
   - `data/results.json` – final EM and F1 scores
   - `data/k_variance.png` – performance vs. retrieval set size (k)
   - `data/ensemble_results.json` – raw predictions (50 queries, k=1…15)

## Project Structure (Key Files)

```
main.py                 # pipeline entry point
src/
├── data_prep.py        # data preparation and Contriever embeddings
├── ensemble.py         # λ‑weighted logsumexp aggregation
├── evaluate.py         # EM / F1 computation
├── dense_index.py      # FAISS index loading
├── baselines.py        # no‑retrieval, BM25, random
└── utils.py            # normalisation, API retry
notebooks/replug-kaggle-v2.ipynb   # data generation
docs/IR Project Report 23K-0666 23I-0710 23K-0753.pdf   # full report
```

## Requirements

- Python 3.10+
- Cohere API key (free tier)
- Dependencies: see `requirements.txt`

## Results (50 NQ queries)

| Condition       | EM    | F1    |
|----------------|-------|-------|
| no_retrieval    | 0.300 | 0.537 |
| bm25            | 0.220 | 0.446 |
| random          | 0.320 | 0.498 |
| REPLUG k=1      | 0.260 | 0.471 |
| REPLUG k=2      | 0.260 | 0.497 |
| REPLUG k=5      | 0.300 | 0.526 |
| **REPLUG k=10** | **0.340** | **0.576** |
| REPLUG k=15     | 0.340 | 0.562 |

REPLUG k=10 improves over no‑retrieval by **4 percentage points** (absolute).  
Full analysis, methodology, and discussion are in the project report.

## Resources

- DPR corpus: [facebook/wiki_dpr](https://huggingface.co/datasets/facebook/wiki_dpr)
- Contriever model: [facebook/contriever](https://huggingface.co/facebook/contriever)
