# Human-in-the-Loop Evaluation

Web-based interface for reviewing and validating AI judge verdicts.

## 📋 Files

| File | Purpose |
|------|---------|
| `human_review_app.py` | Flask web server |
| `hitl_analysis.py` | Analysis & statistics generation |
| `start_hitl.py` | Easy startup script |
| `templates/` | Web UI (dashboard, review, statistics) |

## 🚀 Quick Start

```bash
# From evaluation/human_review/
python start_hitl.py
```

Opens browser at `http://localhost:5000`

## 📖 Documentation

- `HITL_SUMMARY.md` — Overview (5 min read)
- `HITL_QUICKSTART.md` — Setup guide (3 min)
- `HITL_README.md` — Full reference (10 min)
- `BUG_FIXES.md` — Recent fixes

## 🎯 Workflow

```
1. Start web server (start_hitl.py)
2. Review 100 questions (review.html)
3. View real-time stats (/statistics)
4. Generate report (hitl_analysis.py)
5. Analyze patterns & improve model
```

## 📊 Output

Stored in `../results/`:
- `human_feedback.jsonl` — Your verdicts
- `hitl_analysis.json` — Summary statistics

---

**Purpose:** Validate AI judge decisions and identify model weaknesses
