# Dual-Pass Vectorized Candidate Ranking Engine

A high-throughput, private data engineering pipeline designed to ingest, filter, and score a pool of 100,000 candidates entirely within a local CPU system boundary. This solution completely bypasses cloud API latencies and resource-heavy network calls while guaranteeing absolute candidate data privacy.

## 🚀 Key Architectural Pillars

* **Local Semantic Embeddings (60% Weight):** Uses the `all-MiniLM-L6-v2` Sentence-Transformer model cached locally via `embedding.py` to capture conceptual context rather than shallow keyword matching.
* **Multi-Track Behavioral Scorer (40% Weight):** Processes 23 separate platform interaction signals via `signals_scorer.py` to filter out passive profiles and prioritize highly available talent.
* **Heuristic Domain Guard:** Features an algorithmic interceptor that instantly applies a 95% score reduction multiplier (`0.05`) to non-technical trap profiles (e.g., Marketing, HR, Content Writers).
* **Deterministic Tie-Breaking:** Ensures absolute reproducibility by sorting equal scores alphabetically in ascending order by Candidate ID.

## 📁 System Core Layout

* `run_submission.py` — Orchestrates data ingestion, vector math, semantic guards, and file generation.
* `signals_scorer.py` — Evaluates behavioral activity matrix tracks.
* `embedding.py` — Manages local transformer model caching and execution.
* `team.csv` — Spec-compliant production output containing the final top-100 technical shortlist.
* `validate_submission.py` — Automated verification harness checking data constraints and structural compliance.

## 🛠️ Local Reproducibility Steps

Ensure your terminal environment is activated (`venv`), then run the following setup commands:

```bash
# 1. Install necessary core machine learning dependencies
pip install -r requirements.txt

# 2. Execute the processing pipeline (Ingests data, runs models, and outputs team.csv)
python run_submission.py

# 3. Verify compliance and layout structure using the official local test harness
python validate_submission.py team.csv