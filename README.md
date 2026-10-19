# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking (GDPR)

Two retrieval pipelines over the same chunked corpus of the EU General Data Protection Regulation (`GDPR.pdf`, Regulation (EU) 2016/679):

| | Pipeline 1 – Baseline | Pipeline 2 – Re-rank |
|---|---|---|
| Retrieve | `all-MiniLM-L6-v2` bi-encoder, FAISS top 10 (cosine) | same top 10 |
| Re-rank | – | `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all 10 pairs |
| Context | top 3 by cosine | top 3 by cross-encoder score |
| Generate | `gemini-3.5-flash`, identical context-only prompt | same |

Everything is in **`pipeline.ipynb`** (executed, outputs included). Results are also saved to `results/`:

- `rank_shift_summary.csv` – gold-chunk rank before/after re-ranking per failure query
- `answers.md` – full top-10 tables and Answer 1 vs. Answer 2 for each query
- `latency.csv` – per-stage timings (20 repeats × 8 queries)
- `scaling.csv` – cross-encoder cost on top-10 vs. the whole corpus
- `llm_cache.json` – every Gemini response with its original latency; re-running the notebook reuses these, so it works without API quota

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install pdfminer.six sentence-transformers faiss-cpu google-genai pandas python-dotenv tabulate jupyter
echo "GEMINI_API_KEY=your_key" > .env        # free key: https://aistudio.google.com/apikey
jupyter notebook pipeline.ipynb
```

`.env` is git-ignored; never commit your API key.
