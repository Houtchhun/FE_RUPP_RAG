# Human-in-the-Loop Implementation Guide

## ✅ What's Been Implemented

I've created a complete human-in-the-loop evaluation system with:

### 📁 Files Created

```
evaluation/
├── human_review_app.py           # Flask web server
├── start_hitl.py                 # Easy startup script
├── hitl_analysis.py              # Analysis & comparison tool
├── HITL_README.md                # Full documentation
└── templates/
    ├── dashboard.html            # Progress overview
    ├── review.html               # Main review interface
    └── statistics.html           # Real-time statistics
```

---

## 🚀 Getting Started (3 Steps)

### Step 1: Install Flask

```bash
pip install flask
```

### Step 2: Start the Server

```bash
cd evaluation
python start_hitl.py
```

**What happens:**
- Checks Flask is installed ✓
- Verifies evaluation results exist ✓
- Starts web server on `http://localhost:5000`
- Opens browser automatically

### Step 3: Start Reviewing

- **Dashboard** shows your progress (0/100 reviewed)
- **Review** interface lets you validate each question
- **Statistics** shows real-time AI-vs-Human agreement

---

## 🎯 Core Features

### 1. Review Interface (`/review`)
For each of 100 questions:
- Question + Ground Truth (correct answer)
- Retrieved Context (what RAG found)
- Generated Answer (what chatbot said)
- AI Judge Scores:
  - Judge A: Groq Qwen3-32B (relevance, faithfulness, correctness, overall)
  - Judge B: Cerebras Qwen3-235B (same metrics)
- **Your verdict**: ✓ Correct / ≈ Partial / ✗ Incorrect
- Confidence level: 1–5 slider
- Optional notes for disagreements

**Navigation:**
- Previous/Next buttons
- Progress counter
- Skip button

### 2. Statistics Dashboard (`/statistics`)
Real-time metrics:
- Judge A agreement with humans (%)
- Judge B agreement with humans (%)
- AI both agree (cases)
- AI both disagree (cases)  
- Human overrides (cases where you caught both judges being wrong)
- **Disagreement table**: All conflicts listed

### 3. Analysis Script (`hitl_analysis.py`)
```bash
python hitl_analysis.py
```

Generates:
- Coverage % (e.g., 45/100 reviewed)
- Judge agreement rates
- AI agreement patterns
- Disagreement breakdown by category
- Top 5 failure patterns

Example output:
```
Judge A Agreement: 92.0%
Judge B Agreement: 84.0%
Human Overrides: 2 cases (both judges wrong)

Disagreements by Category:
  program_overview: 1
  curriculum_advanced: 3
  career_paths: 2
```

---

## 💾 Data Flow

```
raw_results.json (100 AI-scored questions)
        ↓
  Your Reviews (web UI)
        ↓
  human_feedback.jsonl (stored after each submit)
        ↓
  hitl_analysis.py
        ↓
  hitl_analysis.json (summary statistics)
```

### Storage Format

**human_feedback.jsonl** (one JSON per line):
```json
{
  "question_id": 5,
  "human_verdict": "pass",
  "confidence": 4,
  "notes": "Answer matches ground truth perfectly",
  "timestamp": "2026-06-08T14:23:45.123456"
}
```

---

## 🎯 Why This Matters

### Find Model Weaknesses
Example: If curriculum questions have low agreement
→ Model needs better year-filter detection

### Catch AI Bias
Where do both judges fail?
→ Common hallucination patterns

### Build Ground Truth v2
Your verdicts become validated reference data
→ Train better models

### Calculate Real Accuracy
```
Effective Pass Rate = (Correct + Partial) / Total Reviewed
```

Example: 45/50 = **90% effective accuracy** (better than raw AI scores)

---

## 📊 Review Strategy

### Priority Questions
1. **Both judges disagree** (hardest cases, your verdict is tie-breaker)
2. **Low confidence AI scores** (edge cases)
3. **Year-filter curriculum questions** (test metadata filtering)
4. **Out-of-scope questions** (test politeness)

### Tips
- ✓ Be consistent with previous verdicts
- ✓ Compare against ground truth (source of truth)
- ✓ Write notes for disagreements (identify patterns)
- ✓ Trust your judgment (you catch what AI misses)

---

## 🔧 Advanced Usage

### Detailed Disagreement Report
Edit `hitl_analysis.py`:
```python
if __name__ == '__main__':
    analyze()
    print_detailed_disagreements()  # ← Uncomment
```

Then run:
```bash
python hitl_analysis.py > disagreement_report.txt
```

### Filter by Category
```python
print_detailed_disagreements(category_filter='curriculum_advanced')
```

### Use Different Port
```bash
python -c "from human_review_app import app; app.run(port=5001)"
```

### Export Analysis
Analysis is automatically saved to `results/hitl_analysis.json`:
```json
{
  "timestamp": "2026-06-08T14:35:22.123456",
  "total_reviewed": 45,
  "total_questions": 100,
  "coverage_percent": 45.0,
  "agreement": {
    "groq_percent": 92.0,
    "cerebras_percent": 84.0,
    "both_agree": 38,
    "both_disagree": 7,
    "human_overrides": 2
  },
  "disagreements_by_category": {...},
  "top_issues": [...]
}
```

---

## 📈 Next Steps After Reviewing

1. **Complete 10–20 reviews** (aim for 50%+ coverage)
2. **Run analysis.py** to identify patterns
3. **Focus on disagreements** — Why did both judges fail?
4. **Plan improvements**:
   - Better retrieval (fix year-filter?)
   - More/better training data
   - Adjust model parameters
   - Improve prompt engineering
5. **Retrain/redeploy** and re-evaluate

---

## ❓ Troubleshooting

### Issue: "Port 5000 already in use"
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Issue: "No feedback file created"
- Check that `evaluation/templates/` exists (should be auto-created)
- Ensure you have write permissions in `results/`
- Submit your first verdict to create the file

### Issue: "0% agreement in statistics"
- You haven't submitted any verdicts yet
- Submit at least one verdict in the review interface
- Then run `python hitl_analysis.py`

### Issue: "Browser doesn't load styles"
- Clear browser cache (Ctrl+Shift+Delete)
- Hard refresh (Ctrl+F5)
- Try incognito mode

---

## 📚 Files Reference

| File | Purpose |
|------|---------|
| `human_review_app.py` | Flask server (routes, API endpoints) |
| `start_hitl.py` | Easy startup script |
| `hitl_analysis.py` | Analysis & reporting tool |
| `templates/dashboard.html` | Overview page |
| `templates/review.html` | Main review interface |
| `templates/statistics.html` | Statistics dashboard |
| `HITL_README.md` | Full documentation |
| `results/raw_results.json` | Input: 100 AI-scored questions |
| `results/human_feedback.jsonl` | Output: Your verdicts |
| `results/hitl_analysis.json` | Output: Summary statistics |

---

## 🎓 Example Workflow

```
1. Open http://localhost:5000
   ↓
2. See dashboard: "0 / 100 reviewed"
   ↓
3. Click "Start Review"
   ↓
4. For each question:
   - Read ground truth
   - Compare with generated answer
   - Select verdict (✓/≈/✗)
   - Set confidence (1-5)
   - Add notes if disagreeing with AI
   - Click "Submit Verdict"
   ↓
5. View statistics in real-time
   ↓
6. After 50+ reviews, run:
   python hitl_analysis.py
   ↓
7. Use insights to improve model
```

---

## 🎯 Expected Outcomes

After reviewing all 100 questions:

✅ **Metrics**
- Judge A agreement: 85–95% (should be high)
- Judge B agreement: 80–90% (more conservative)
- Human overrides: 1–5 cases (where both judges wrong)

✅ **Insights**
- "Model struggles with [category]"
- "Hallucination pattern: [type]"
- "Year-filter not working for [queries]"

✅ **Improvements**
- Effective pass rate = (Correct + Partial) / 100
- Better than raw AI scores due to human judgment
- Validated ground truth for training next version

---

## 📞 Questions?

Refer to:
- `HITL_README.md` — Full detailed guide
- `evaluation/` — Browse source code
- `results/hitl_analysis.json` — See generated statistics
- `results/human_feedback.jsonl` — View your submitted verdicts


