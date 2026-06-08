# AI Evaluation (Dual-Judge System)

Automated evaluation of RAG system using two LLM judges.

## 📋 Files

| File | Purpose |
|------|---------|
| `evaluate.py` | Run 100 questions through dual judges (Groq + Cerebras) |
| `report.py` | Generate tables and visualization charts |
| `ground_truth.json` | 100 reference Q&A pairs (8 categories) |

## 🚀 Quick Start

```bash
# 1. Set API keys in ../fe_index/.env
# GROQ_API_KEY=...
# CEREBRAS_API_KEY=...

# 2. Run evaluation (takes ~30 min)
python evaluate.py --reset

# 3. Generate reports & charts
python report.py
```

## 📊 Output

Results saved to `../results/`:
- `raw_results.json` — 100 questions with AI scores
- `plots/` — Evaluation visualizations

## 🔍 Next Steps

After automated evaluation:
1. Review results: `python report.py`
2. Use human review: `cd ../human_review && python start_hitl.py`
3. Compare AI vs human verdicts

---

**Evaluation Models:**
- Generator: `llama-3.1-8b-instant` (Groq)
- Judge A: `qwen/qwen3-32b` (Groq)
- Judge B: `qwen-3-235b-a22b-instruct-2507` (Cerebras)
