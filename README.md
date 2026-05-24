# REPLUG QA

Open-domain question answering with dense retrieval and λ‑weighted ensemble (REPLUG Section 3.2).

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Set `COHERE_API_KEY` in `.env` (copy from `.env.example`)
3. Run: `python main.py`
4. Output: `data/results.json` (EM/F1 scores) and `data/k_variance.png`

## Requirements

- Python 3.10+
- Cohere API key (free tier)

## Running Tests

```bash
pytest tests/ -v
```
