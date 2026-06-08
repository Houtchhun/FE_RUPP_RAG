# 📊 Evaluation System

Comprehensive RAG evaluation with automated AI judges and human-in-the-loop validation.

## 📁 Folder Structure

```
evaluation/
├── judges/                          # Automated AI evaluation
│   ├── evaluate.py                  # Run dual judges
│   ├── report.py                    # Generate charts & tables
│   ├── ground_truth.json            # 100 reference Q&A pairs
│   └── README.md                    # Quick reference
│
├── human_review/                    # Human validation interface
│   ├── human_review_app.py          # Flask web server
│   ├── hitl_analysis.py             # Statistics & analysis
│   ├── start_hitl.py                # Startup script
│   ├── templates/                   # Web UI templates
│   ├── README.md                    # Quick reference
│   ├── HITL_SUMMARY.md              # Overview
│   ├── HITL_QUICKSTART.md           # Setup guide
│   ├── HITL_README.md               # Full documentation
│   ├── HITL_FILES.md                # File reference
│   └── BUG_FIXES.md                 # Recent fixes
│
├── results/                         # Evaluation outputs
│   ├── raw_results.json             # 100 AI-scored questions
│   ├── human_feedback.jsonl         # Human verdicts
│   ├── hitl_analysis.json           # Summary statistics
│   └── plots/                       # Visualization charts
│
└── [This README]
```

## 🚀 Quick Start

### Step 1: Run Automated Evaluation
```bash
cd judges
python evaluate.py --reset
python report.py
```
Creates `../results/raw_results.json` (100 AI-scored questions)

### Step 2: Human Review
```bash
cd ../human_review
python start_hitl.py
```
Opens web interface at `http://localhost:5000`

### Step 3: Generate Analysis
```bash
python hitl_analysis.py
```
Creates `../results/hitl_analysis.json` (statistics)

---

## 📖 Documentation

### For AI Evaluation
→ See `judges/README.md`

### For Human Review
→ Start with `human_review/HITL_SUMMARY.md` (5 min)

---

## 🔄 Evaluation Pipeline

```
┌─────────────────────────────────────────┐
│  1. AUTOMATED AI EVALUATION             │
│     (judges/evaluate.py)                │
├─────────────────────────────────────────┤
│  Input:  100 ground truth Q&A pairs     │
│  Output: raw_results.json               │
│          (AI-scored for pass/fail)      │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  2. HUMAN VALIDATION                    │
│     (human_review/start_hitl.py)        │
├─────────────────────────────────────────┤
│  Input:  raw_results.json               │
│  Output: human_feedback.jsonl           │
│          (Your verdicts)                │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  3. ANALYSIS                            │
│     (human_review/hitl_analysis.py)     │
├─────────────────────────────────────────┤
│  Input:  raw_results.json +             │
│          human_feedback.jsonl           │
│  Output: hitl_analysis.json             │
│          (Agreement stats)              │
└─────────────────────────────────────────┘
              ↓
         INSIGHTS
     (Identify model weaknesses)
           ↓
        IMPROVE
     (Refine retrieval/generation)
```

---

## 📊 Key Metrics

**Automated Evaluation (judges/):**
- Pass rate (%) by each judge
- Relevance, faithfulness, correctness scores
- Judge A: Groq (Qwen3-32B)
- Judge B: Cerebras (Qwen3-235B)

**Human Validation (human_review/):**
- Judge A agreement with humans (%)
- Judge B agreement with humans (%)
- Human overrides (cases where both judges wrong)
- Disagreements by category

**Combined Analysis:**
- Effective pass rate (human-validated)
- Model weakness identification
- Failure pattern analysis

---

## 🎯 Typical Workflow

1. **Day 1: Setup**
   - Run `judges/evaluate.py` (30 min, background)
   - Review results: `judges/report.py`

2. **Day 2-5: Review**
   - Start: `human_review/python start_hitl.py`
   - Review 20 questions per session
   - View stats: Go to `/statistics`

3. **Day 6: Analysis**
   - Run: `human_review/hitl_analysis.py`
   - Identify patterns (e.g., "Model struggles with curriculum")
   - Plan improvements

4. **Day 7+: Iterate**
   - Improve retrieval/generation
   - Re-run evaluation
   - Compare before/after

---

## 🔧 Troubleshooting

**Issue:** "raw_results.json not found"
→ Run `cd judges && python evaluate.py` first

**Issue:** "Flask not installed"
→ `pip install flask` (or `pip install -r ../requirements.txt`)

**Issue:** "Port 5000 in use"
→ Use different port: Edit `start_hitl.py` line 54

**Issue:** Relative paths fail
→ Use Python not Bash (`python script.py` not `./script.py`)

---

## 📞 Getting Help

1. **Setup help** → `human_review/HITL_QUICKSTART.md`
2. **Detailed guide** → `human_review/HITL_README.md`
3. **File reference** → `human_review/HITL_FILES.md`
4. **Recent fixes** → `human_review/BUG_FIXES.md`

---

## 📈 Expected Results

After completing full evaluation:

**AI Judge Performance:**
- Judge A: 88–94% agreement with humans
- Judge B: 80–88% agreement with humans

**Model Quality:**
- Effective pass rate: 85–95%
- Hallucination rate: <5%
- Category-specific accuracy: Identify weak areas

**Actionable Insights:**
- Top 3 failure patterns
- Category-specific issues
- Retrieval vs generation problems

---

**For more details, start with the relevant README:**
- 🤖 AI Evaluation? → `judges/README.md`
- 👤 Human Review? → `human_review/HITL_SUMMARY.md`
