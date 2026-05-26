# FE-RUPP RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for the **Faculty of Engineering (FE) at the Royal University of Phnom Penh (RUPP)**, Cambodia. The bot answers questions about academic programs, curriculum, admission, careers, and faculty information via Telegram.

---

## Evaluation Results

Evaluated on **100 questions** across 4 categories using a dual-judge system with threshold-based scoring.

| Judge | Pass Rate | Relevance | Faithfulness | Correctness |
|-------|-----------|-----------|--------------|-------------|
| Judge A — Groq (Qwen3-32B) | **94.0%** | 0.975 | 0.985 | 0.970 |
| Judge B — Cerebras (Qwen3-235B) | **86.0%** | 0.950 | 0.980 | 0.915 |

**Pass threshold:** relevance ≥ 0.6 · faithfulness ≥ 0.7 · correctness ≥ 0.5 · overall ≥ 0.7 (all required)  
**Zero questions failed both judges.**

---

## System Architecture

```
User (Telegram)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                   QUERY PROCESSING                   │
│  1. Normalize query (expand abbreviations & years)   │
│  2. Detect year filter for curriculum questions      │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                 HYBRID RETRIEVAL                     │
│                                                      │
│  Vector Search          BM25 Keyword Search          │
│  (ChromaDB)             (rank-bm25)                  │
│  BAAI/bge-base-en-v1.5  20 candidates               │
│  20 candidates                │                      │
│        └──────────────────────┘                      │
│                    ▼                                 │
│          Reciprocal Rank Fusion (RRF)                │
│          score = Σ 1 / (60 + rank)                  │
│                    ▼                                 │
│    Cross-Encoder Reranking (top-5)                   │
│    cross-encoder/ms-marco-MiniLM-L-6-v2             │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                  GENERATION                          │
│  LLM: llama-3.1-8b-instant (Groq)                  │
│  temp = 0.0  (deterministic)                        │
│  Context-only answers — no hallucination            │
│  Per-user 5-turn conversation memory               │
└─────────────────────────────────────────────────────┘
      │
      ▼
Answer + Sources → Telegram
```

---

## Project Structure

```
Chatbot_v2/
├── fe_index/                        # RAG system & Telegram bot
│   ├── bot.py                       # Main bot: retrieval + generation + memory
│   ├── build_index.py               # Builds ChromaDB from scraped data
│   ├── chroma_db/                   # Vector database (auto-generated, not in git)
│   └── .env                         # API keys (not in git)
│
├── scraper/                         # Data collection pipeline
│   ├── scraper.py                   # Web scraper for FE-RUPP website
│   ├── generate_chunks.py           # Converts JSON → retrieval chunks
│   ├── major.txt                    # Raw curriculum text data
│   └── outputs/
│       ├── all_programs_comprehensive.json   # Master dataset (9 programs)
│       └── chunks/all_chunks.json            # Generated chunks (auto-created)
│
├── evaluation/                      # RAG evaluation framework
│   ├── evaluate.py                  # Dual-judge evaluation pipeline
│   ├── report.py                    # Generates tables and charts
│   ├── ground_truth.json            # 100 Q&A pairs (8 categories)
│   └── results/
│       ├── raw_results.json         # Per-question scores
│       └── plots/                   # 5 evaluation charts
│
├── requirements.txt                 # All Python dependencies
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A [Telegram bot token](https://core.telegram.org/bots#botfather)
- A [Groq API key](https://console.groq.com/) (free tier)
- A [Cerebras API key](https://cloud.cerebras.ai/) (for evaluation only)

### 1 — Install dependencies

```bash
git clone <repo-url>
cd Chatbot_v2

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2 — Set API keys

Create `fe_index/.env`:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
GROQ_API_KEY=your_groq_api_key
```

### 3 — Build the index

```bash
cd fe_index
python build_index.py
```

> This embeds 186 documents into ChromaDB using `BAAI/bge-base-en-v1.5`.  
> Takes ~1–2 minutes on first run (downloads the embedding model).

### 4 — Start the bot

```bash
python bot.py
```

Open Telegram, find your bot, and start chatting.

---

## Sample Questions

| Category | Example |
|----------|---------|
| Programs | *"What programs does FE-RUPP offer?"* |
| Curriculum | *"What courses are in Year 2 of Data Science?"* |
| Curriculum | *"What are the Foundation Year subjects for ITE?"* |
| Career | *"What jobs can DSE graduates get?"* |
| General | *"Where is the Faculty of Engineering located?"* |
| Out-of-scope | *"What is machine learning?"* → Bot politely declines |

**Bot commands:**  
`/start` — Welcome message  
`/help` — Show help  
`/clear` — Reset conversation history

---

## Retrieval Pipeline — Technical Details

| Stage | Method | Detail |
|-------|--------|--------|
| Embedding | `BAAI/bge-base-en-v1.5` | 768-dim, normalize=True |
| Vector search | ChromaDB cosine similarity | 20 candidates, distance ≤ 1.2 |
| Keyword search | BM25 Okapi | 20 candidates |
| Fusion | Reciprocal Rank Fusion | RRF_K = 60 |
| Reranking | Cross-encoder | `ms-marco-MiniLM-L-6-v2`, top-5 |
| Generation | `llama-3.1-8b-instant` | Groq API, temp = 0.0 |

**Year-metadata filtering:** Curriculum queries that mention a specific year (e.g., "Year 3") use a ChromaDB metadata filter `{"year": {"$eq": "year_3"}}` before reranking — this eliminates cross-year confusion entirely.

---

## Running the Evaluation

```bash
# Add Cerebras key to fe_index/.env
CEREBRAS_API_KEY=your_cerebras_api_key

# Run all 100 questions (takes ~30 min)
python evaluation/evaluate.py --reset

# Print tables and generate charts
python evaluation/report.py
```

**Evaluation models:**

| Role | Model | Provider |
|------|-------|----------|
| Generator | `llama-3.1-8b-instant` | Groq |
| Judge A | `qwen/qwen3-32b` | Groq |
| Judge B | `qwen-3-235b-a22b-instruct-2507` | Cerebras |

---

## Data — 9 Academic Programs

| Program | Type | Duration |
|---------|------|----------|
| Data Science and Engineering | Honor Bachelor | 4 years |
| Information Technology Engineering | Honor Bachelor | 4 years |
| Telecommunication and Electronics Engineering | Honor Bachelor | 4 years |
| Bio Engineering Biotechnology | Honor Bachelor | 4 years |
| Food Technology and Engineering | Bachelor | 4 years |
| Environmental Engineering | Bachelor | 4 years |
| Automation and Supply Chain System Engineering | Bachelor | 4 years |
| Master of Science in Information Technology Engineering (MITE) | Master | 2 years |
| Biotechnology and Food Technology Master (MBFT) | Master | 2 years |

---

## Project Info

- **Institution:** Faculty of Engineering, Royal University of Phnom Penh (RUPP)
- **Course:** Year 3, Semester 2 — Practicum Project
- **Stack:** Python · LangChain · ChromaDB · Groq · Cerebras · Telegram
