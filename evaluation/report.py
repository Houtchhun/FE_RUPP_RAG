#!/usr/bin/env python3
"""
FE-RUPP RAG — Evaluation Report
Prints Tables 4.1 / 4.2 / 4.3 and saves 5 PNG charts.

Run after evaluate.py:
    python evaluation/report.py
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_JSON  = Path(__file__).parent / "results" / "raw_results.json"
PLOTS_DIR = Path(__file__).parent / "results" / "plots"

# ── Category config ────────────────────────────────────────────────────────────
CAT_ORDER = ["program_overview", "curriculum", "career", "general"]
CAT_LABELS = {
    "program_overview": "Program Overview",
    "curriculum":       "Curriculum",
    "career":           "Career",
    "general":          "General",
}
CAT_COLORS = {
    "program_overview": "#4C72B0",
    "curriculum":       "#DD8452",
    "career":           "#C44E52",
    "general":          "#8C8C8C",
}

# ── Judge config ───────────────────────────────────────────────────────────────
JUDGES = ["groq", "cerebras"]
JUDGE_LABELS = {
    "groq":     "Judge A: Groq (Qwen3 32B)",
    "cerebras": "Judge B: Cerebras (Qwen 3 235B)",
}
JUDGE_COLORS = {"groq": "#2196F3", "cerebras": "#FF5722"}

# Threshold-based scoring floors (must match evaluate.py)
RELEVANCE_FLOOR    = 0.60
FAITHFULNESS_FLOOR = 0.70
CORRECTNESS_FLOOR  = 0.50
OVERALL_PASS       = 0.70
PARTIAL_T          = 0.40
PASS_T             = OVERALL_PASS   # alias for chart lines


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load() -> list:
    with open(RAW_JSON, encoding="utf-8") as f:
        return json.load(f)


def _avg(items: list, judge: str, metric: str) -> float:
    key  = f"{judge}_scores"
    vals = [r[key][metric] for r in items if key in r and metric in r[key]]
    return sum(vals) / len(vals) if vals else 0.0


def _pass_rate(items: list, judge: str) -> float:
    key = f"{judge}_verdict"
    return sum(1 for r in items if r.get(key) == "pass") / len(items) * 100 if items else 0.0


def _count(items: list, judge: str, verdict: str) -> int:
    key = f"{judge}_verdict"
    return sum(1 for r in items if r.get(key) == verdict)


def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart -> {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# TEXT SUMMARY (Tables 4.1 / 4.2 / 4.3)
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(results: list) -> None:
    sep  = "=" * 92
    sep2 = "-" * 92
    N    = len(results)

    print(f"\n{sep}")
    print("  FE-RUPP RAG — DUAL-JUDGE EVALUATION SUMMARY")
    print(f"  Total questions  : {N}")
    print(f"  Generator        : Groq llama-3.1-8b-instant  (temp=0.0)")
    print(f"  Judge A (Groq)   : qwen/qwen3-32b")
    print(f"  Judge B (Cerebras): qwen-3-235b-a22b-instruct-2507")
    print(f"  Retrieval        : BM25 + Vector + RRF -> Cross-encoder rerank  (top-5)")
    print(f"  Threshold logic  : rel≥{RELEVANCE_FLOOR}  faith≥{FAITHFULNESS_FLOOR}  corr≥{CORRECTNESS_FLOOR}  overall≥{OVERALL_PASS}  ALL required for PASS")
    print(sep)

    # ── Table 4.1 ─────────────────────────────────────────────────────────────
    print("\nTable 4.1 — Evaluation Results by Query Category\n")
    print("{:<22} {:>5}  {:>10} {:>14}  {:>7} {:>7}  {:>8} {:>8}  {:>8}".format(
        "Category", "N-QA",
        "Groq Pass%", "Cerebras Pass%",
        "G.Rel", "C.Rel",
        "G.Faith", "C.Faith",
        "G.Corr",
    ))
    print(sep2[:88])
    for cat in CAT_ORDER:
        cat_r = [r for r in results if r["category"] == cat]
        if not cat_r:
            continue
        label = CAT_LABELS.get(cat, cat)
        print("{:<22} {:>5}  {:>9.1f}% {:>13.1f}%  {:>7.3f} {:>7.3f}  {:>8.3f} {:>8.3f}  {:>8.3f}".format(
            label, len(cat_r),
            _pass_rate(cat_r, "groq"), _pass_rate(cat_r, "cerebras"),
            _avg(cat_r, "groq", "relevance"),    _avg(cat_r, "cerebras", "relevance"),
            _avg(cat_r, "groq", "faithfulness"), _avg(cat_r, "cerebras", "faithfulness"),
            _avg(cat_r, "groq", "correctness"),
        ))
    print(sep2[:88])
    print("{:<22} {:>5}  {:>9.1f}% {:>13.1f}%  {:>7.3f} {:>7.3f}  {:>8.3f} {:>8.3f}  {:>8.3f}".format(
        "TOTAL", N,
        _pass_rate(results, "groq"), _pass_rate(results, "cerebras"),
        _avg(results, "groq", "relevance"),    _avg(results, "cerebras", "relevance"),
        _avg(results, "groq", "faithfulness"), _avg(results, "cerebras", "faithfulness"),
        _avg(results, "groq", "correctness"),
    ))

    # ── Table 4.2 ─────────────────────────────────────────────────────────────
    print("\nTable 4.2 — Model Performance Comparison (Threshold-Based Scoring)\n")
    print("{:<34} {:>10} {:>12} {:>14} {:>12} {:>9}".format(
        "Judge Model", "Pass Rate", "Relevance", "Faithfulness", "Correctness", "Overall"
    ))
    print("-" * 94)
    for j in JUDGES:
        print("{:<34} {:>9.1f}% {:>12.3f} {:>14.3f} {:>12.3f} {:>9.3f}".format(
            JUDGE_LABELS[j],
            _pass_rate(results, j),
            _avg(results, j, "relevance"),
            _avg(results, j, "faithfulness"),
            _avg(results, j, "correctness"),
            _avg(results, j, "overall"),
        ))
    print("-" * 94)
    print(f"  Thresholds (PASS floor):  rel≥{RELEVANCE_FLOOR}  faith≥{FAITHFULNESS_FLOOR}  corr≥{CORRECTNESS_FLOOR}  overall≥{OVERALL_PASS}")

    # ── Per-metric floor pass rate ─────────────────────────────────────────────
    print("\nTable 4.2b — Per-Metric Threshold Pass Rate (% of questions above individual floor)\n")
    thresholds = [("relevance", RELEVANCE_FLOOR), ("faithfulness", FAITHFULNESS_FLOOR), ("correctness", CORRECTNESS_FLOOR)]
    print("{:<34} {:>14} {:>16} {:>14}".format("Judge Model", "Rel≥{:.0%}".format(RELEVANCE_FLOOR),
          "Faith≥{:.0%}".format(FAITHFULNESS_FLOOR), "Corr≥{:.0%}".format(CORRECTNESS_FLOOR)))
    print("-" * 80)
    for j in JUDGES:
        key = f"{j}_scores"
        rates = []
        for metric, floor in thresholds:
            n_above = sum(1 for r in results if key in r and r[key].get(metric, 0) >= floor)
            rates.append(n_above / len(results) * 100)
        print("{:<34} {:>13.1f}% {:>15.1f}% {:>13.1f}%".format(JUDGE_LABELS[j], *rates))
    print("-" * 80)

    # ── Table 4.3 ─────────────────────────────────────────────────────────────
    print("\nTable 4.3 — Detailed Verdict Counts per Judge\n")
    print("{:<30} {:>8} {:>9} {:>8}".format("Judge", "Passed", "Partial", "Failed"))
    print("-" * 58)
    for j in JUDGES:
        print("{:<30} {:>8} {:>9} {:>8}".format(
            JUDGE_LABELS[j],
            _count(results, j, "pass"),
            _count(results, j, "partial"),
            _count(results, j, "fail"),
        ))
    print("-" * 58)

    # ── Agreement ─────────────────────────────────────────────────────────────
    agree = sum(1 for r in results if r.get("groq_verdict") == r.get("cerebras_verdict"))
    print(f"\n  Judge agreement : {agree}/{N} ({agree/N*100:.1f}%)")

    # ── Both failed ───────────────────────────────────────────────────────────
    both_fail = [r for r in results if r.get("groq_verdict") == "fail" and r.get("cerebras_verdict") == "fail"]
    print(f"\n  Failed by both judges ({len(both_fail)}):")
    if both_fail:
        for r in both_fail:
            g = r["groq_scores"]["overall"]
            c = r["cerebras_scores"]["overall"]
            print(f"  Q{r['id']:>3} [{r['category']:<16}] G={g:.2f} C={c:.2f} | {r['question'][:60]}")
    else:
        print("  None.")

    print(f"\n{sep}")


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_overall_donut(results: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, judge in zip(axes, JUDGES):
        N       = len(results)
        passed  = _count(results, judge, "pass")
        partial = _count(results, judge, "partial")
        failed  = N - passed - partial
        wedges, _ = ax.pie(
            [passed, partial, failed],
            colors=["#2ecc71", "#f39c12", "#e74c3c"],
            startangle=90,
            wedgeprops=dict(width=0.45, edgecolor="white"),
        )
        ax.text(0,  0.10, f"{passed/N*100:.1f}%", ha="center", va="center",
                fontsize=20, fontweight="bold", color="#2c3e50")
        ax.text(0, -0.15, "Pass Rate", ha="center", va="center", fontsize=10, color="#555")
        ax.text(0, -0.40, f"{passed}/{N}", ha="center", va="center", fontsize=9, color="#888")
        ax.set_title(JUDGE_LABELS[judge], fontsize=11, fontweight="bold", pad=10)

    fig.legend(
        handles=[mpatches.Patch(color=c, label=l) for c, l in
                 [("#2ecc71","Pass"), ("#f39c12","Partial"), ("#e74c3c","Fail")]],
        loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Overall Evaluation — Dual Judge", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "00_overall_donut.png")


def plot_pass_rate_by_category(results: list) -> None:
    cats   = [c for c in CAT_ORDER if any(r["category"] == c for r in results)]
    labels = [CAT_LABELS.get(c, c) for c in cats]
    x, w   = np.arange(len(cats)), 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, judge in enumerate(JUDGES):
        rates = [_pass_rate([r for r in results if r["category"] == c], judge) for c in cats]
        bars  = ax.bar(x + (i - 0.5) * w, rates, w,
                       label=JUDGE_LABELS[judge], color=JUDGE_COLORS[judge],
                       alpha=0.85, edgecolor="white")
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Pass Rate by Category — Groq vs Cerebras", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "01_pass_rate_by_category.png")


def plot_score_per_question(results: list) -> None:
    sorted_r = sorted(results, key=lambda r: r["id"])
    ids      = [r["id"] for r in sorted_r]

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    for ax, judge in zip(axes, JUDGES):
        scores   = [r[f"{judge}_scores"]["overall"] for r in sorted_r]
        verdicts = [r.get(f"{judge}_verdict", "fail") for r in sorted_r]
        colors   = ["#2ecc71" if v == "pass" else ("#f39c12" if v == "partial" else "#e74c3c")
                    for v in verdicts]
        ax.bar(ids, scores, color=colors, width=0.8, edgecolor="none")
        ax.axhline(PASS_T,    color="#333", lw=1.2, ls="--", label=f"Pass ({PASS_T})")
        ax.axhline(PARTIAL_T, color="#999", lw=0.8, ls=":",  label=f"Partial ({PARTIAL_T})")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Overall Score")
        ax.set_title(f"{JUDGE_LABELS[judge]}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Question ID")
    fig.suptitle("Overall Score per Question — Dual Judge", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_score_per_question.png")


def plot_metric_comparison(results: list) -> None:
    cats    = [c for c in CAT_ORDER if any(r["category"] == c for r in results)]
    labels  = [CAT_LABELS.get(c, c) for c in cats]
    metrics = [("relevance", "Relevance"), ("faithfulness", "Faithfulness"), ("correctness", "Correctness")]

    metric_floors = {
        "relevance":    RELEVANCE_FLOOR,
        "faithfulness": FAITHFULNESS_FLOOR,
        "correctness":  CORRECTNESS_FLOOR,
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (metric, mlabel) in zip(axes, metrics):
        floor = metric_floors[metric]
        x, w  = np.arange(len(cats)), 0.35
        for i, judge in enumerate(JUDGES):
            vals = [_avg([r for r in results if r["category"] == c], judge, metric) for c in cats]
            ax.bar(x + (i - 0.5) * w, vals, w,
                   label=JUDGE_LABELS[judge], color=JUDGE_COLORS[judge],
                   alpha=0.85, edgecolor="white")
        ax.axhline(floor, color="#e74c3c", lw=1.4, ls="--",
                   label=f"Pass floor ({floor:.0%})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Metric Breakdown by Category — Dual Judge (red line = pass floor)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_metric_by_category.png")


def plot_judge_agreement(results: list) -> None:
    verdicts = ["pass", "partial", "fail"]
    matrix   = np.zeros((3, 3), dtype=int)
    for r in results:
        gi = verdicts.index(r.get("groq_verdict", "fail"))
        ci = verdicts.index(r.get("cerebras_verdict", "fail"))
        matrix[gi][ci] += 1

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["Pass", "Partial", "Fail"])
    ax.set_yticklabels(["Pass", "Partial", "Fail"])
    ax.set_xlabel("Cerebras (Qwen 3 235B) verdict")
    ax.set_ylabel("Groq (Llama 3.1 8B) verdict")
    ax.set_title("Judge Agreement Matrix", fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if matrix[i, j] > matrix.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _save(fig, "04_judge_agreement.png")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not RAW_JSON.exists():
        print(f"No results at {RAW_JSON}\nRun:  python evaluation/evaluate.py")
        raise SystemExit(1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results = _load()

    print_summary(results)

    print(f"\nGenerating charts ({len(results)} questions)...")
    plot_overall_donut(results)
    plot_pass_rate_by_category(results)
    plot_score_per_question(results)
    plot_metric_comparison(results)
    plot_judge_agreement(results)

    print(f"\nDone. Plots in {PLOTS_DIR}")
