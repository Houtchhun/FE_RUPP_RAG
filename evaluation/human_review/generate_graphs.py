"""
Generate comparison and importance graphs for slides/reports
Creates: Judge comparison, Category breakdown, Failure patterns
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(EVAL_DIR, '..', 'results', 'raw_results.json')
FEEDBACK_FILE = os.path.join(EVAL_DIR, '..', 'results', 'human_feedback.jsonl')
OUTPUT_DIR = os.path.join(EVAL_DIR, '..', 'results', 'graphs')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_results():
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)

def load_feedback():
    feedback = {}
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            for line in f:
                entry = json.loads(line)
                feedback[entry['question_id']] = entry
    return feedback

def plot_judge_comparison():
    """Compare Judge A vs Judge B vs Human"""
    results = load_results()
    feedback = load_feedback()

    if not feedback:
        print("No human feedback yet. Skipping judge comparison.")
        return

    judge_a_pass = 0
    judge_b_pass = 0
    human_pass = 0

    for qid in feedback:
        result = results[qid]
        if result['groq_verdict'] == 'pass':
            judge_a_pass += 1
        if result['cerebras_verdict'] == 'pass':
            judge_b_pass += 1
        if feedback[qid]['human_verdict'] == 'pass':
            human_pass += 1

    total = len(feedback)

    fig, ax = plt.subplots(figsize=(10, 6))

    judges = ['Judge A\n(Groq Qwen3-32B)', 'Judge B\n(Cerebras Qwen3-235B)', 'Human Reviewer']
    pass_rates = [judge_a_pass/total*100, judge_b_pass/total*100, human_pass/total*100]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars = ax.bar(judges, pass_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, rate in zip(bars, pass_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%\n({int(rate/100*total)}/{total})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Judge Comparison: Pass Rates\n(Based on {} Human Reviews)'.format(total),
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_judge_comparison.png'), dpi=300, bbox_inches='tight')
    print("[OK] Saved: 1_judge_comparison.png")
    plt.close()

def plot_verdict_distribution():
    """Distribution of verdicts: Pass, Partial, Fail"""
    results = load_results()
    feedback = load_feedback()

    if not feedback:
        print("No human feedback yet. Skipping verdict distribution.")
        return

    verdicts = {'pass': 0, 'partial': 0, 'fail': 0}
    for qid in feedback:
        verdict = feedback[qid]['human_verdict']
        if verdict in verdicts:
            verdicts[verdict] += 1

    total = len(feedback)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    labels = ['Correct', 'Partial', 'Incorrect']
    values = [verdicts['pass'], verdicts['partial'], verdicts['fail']]
    colors_bar = ['#2ca02c', '#ff7f0e', '#d62728']

    bars = ax1.bar(labels, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Verdict Distribution (Human Reviews)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Pie chart
    ax2.pie(values, labels=labels, colors=colors_bar, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Verdict Distribution (Percentage)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_verdict_distribution.png'), dpi=300, bbox_inches='tight')
    print("[OK] Saved: 2_verdict_distribution.png")
    plt.close()

def plot_category_breakdown():
    """Pass rate by category"""
    results = load_results()
    feedback = load_feedback()

    category_stats = defaultdict(lambda: {'pass': 0, 'total': 0})

    for qid in range(len(results)):
        result = results[qid]
        category = result['category']
        category_stats[category]['total'] += 1

        if qid in feedback:
            if feedback[qid]['human_verdict'] == 'pass':
                category_stats[category]['pass'] += 1

    # Calculate pass rates
    categories = sorted(category_stats.keys())
    pass_rates = []
    counts = []

    for cat in categories:
        total = category_stats[cat]['total']
        passed = category_stats[cat]['pass']
        if total > 0:
            pass_rates.append(passed / total * 100)
            counts.append(f"{passed}/{total}")
        else:
            pass_rates.append(0)
            counts.append("0/0")

    fig, ax = plt.subplots(figsize=(12, 6))

    colors_cat = ['#1f77b4' if rate >= 80 else '#ff7f0e' if rate >= 60 else '#d62728'
                  for rate in pass_rates]

    bars = ax.barh(categories, pass_rates, color=colors_cat, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, rate, count) in enumerate(zip(bars, pass_rates, counts)):
        ax.text(rate + 2, bar.get_y() + bar.get_height()/2.,
               f'{rate:.1f}% ({count})',
               ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Category', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 110)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_category_breakdown.png'), dpi=300, bbox_inches='tight')
    print("[OK] Saved: 3_category_breakdown.png")
    plt.close()

def plot_ai_vs_human_agreement():
    """AI judges vs Human agreement"""
    results = load_results()
    feedback = load_feedback()

    if not feedback:
        print("No human feedback yet. Skipping AI vs human agreement.")
        return

    judge_a_agree = 0
    judge_b_agree = 0
    both_agree = 0
    human_catches = 0

    for qid in feedback:
        result = results[qid]
        hv = feedback[qid]['human_verdict']

        if result['groq_verdict'] == hv:
            judge_a_agree += 1
        if result['cerebras_verdict'] == hv:
            judge_b_agree += 1

        if result['groq_verdict'] == result['cerebras_verdict']:
            both_agree += 1
            if result['groq_verdict'] != hv:
                human_catches += 1

    total = len(feedback)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Agreement rates
    ax1.bar(['Judge A', 'Judge B'],
           [judge_a_agree/total*100, judge_b_agree/total*100],
           color=['#1f77b4', '#ff7f0e'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Agreement (%)', fontweight='bold')
    ax1.set_title('Judge-Human Agreement Rate', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)
    for i, val in enumerate([judge_a_agree/total*100, judge_b_agree/total*100]):
        ax1.text(i, val+2, f'{val:.1f}%', ha='center', fontweight='bold')

    # Disagreement types
    categories_dis = ['Agree', 'Disagree']
    values_dis = [both_agree, total - both_agree]
    colors_dis = ['#2ca02c', '#d62728']
    ax2.pie(values_dis, labels=categories_dis, colors=colors_dis, autopct='%1.1f%%',
           textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Cases Where Both Judges Agree/Disagree', fontweight='bold', fontsize=12)

    # Human catches errors
    ax3.bar(['Both Agree\nWith Human', 'Both Agree\nHuman Catches Error'],
           [both_agree - human_catches, human_catches],
           color=['#2ca02c', '#ff7f0e'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Human Accuracy: Catching AI Errors', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_axisbelow(True)

    # Summary stats
    ax4.axis('off')
    summary_text = f"""
AGREEMENT SUMMARY

Judge A Agreement:     {judge_a_agree}/{total} ({judge_a_agree/total*100:.1f}%)
Judge B Agreement:     {judge_b_agree}/{total} ({judge_b_agree/total*100:.1f}%)

Both Judges Agree:     {both_agree}/{total} ({both_agree/total*100:.1f}%)
Both Judges Disagree:  {total-both_agree}/{total} ({(total-both_agree)/total*100:.1f}%)

Human Catches Errors:  {human_catches}/{both_agree} cases
    (Times both judges wrong)
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_ai_vs_human_agreement.png'), dpi=300, bbox_inches='tight')
    print("[OK] Saved: 4_ai_vs_human_agreement.png")
    plt.close()

def plot_metric_scores():
    """Average scores for each metric"""
    results = load_results()

    metrics = {'relevance': [], 'faithfulness': [], 'correctness': [], 'overall': []}

    for result in results:
        for metric in metrics.keys():
            metrics[metric].append(result['groq_scores'][metric])

    avg_scores = {m: sum(scores)/len(scores) for m, scores in metrics.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    metric_names = ['Relevance', 'Faithfulness', 'Correctness', 'Overall']
    scores = [avg_scores['relevance'], avg_scores['faithfulness'],
              avg_scores['correctness'], avg_scores['overall']]
    colors_met = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    bars = ax1.bar(metric_names, scores, color=colors_met, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Pass Threshold (0.7)')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Average AI Judge Scores (Judge A)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height+0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Threshold compliance
    meets_threshold = [1 if s >= 0.7 else 0 for s in scores]
    colors_th = ['#2ca02c' if m else '#d62728' for m in meets_threshold]
    ax2.bar(metric_names, meets_threshold, color=colors_th, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Meets Threshold', fontsize=12, fontweight='bold')
    ax2.set_title('Metric Threshold Compliance (≥0.7)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.2)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_metric_scores.png'), dpi=300, bbox_inches='tight')
    print("[OK] Saved: 5_metric_scores.png")
    plt.close()

def main():
    print("\n" + "="*70)
    print("GENERATING COMPARISON & IMPORTANCE GRAPHS")
    print("="*70 + "\n")

    plot_judge_comparison()
    plot_verdict_distribution()
    plot_category_breakdown()
    plot_ai_vs_human_agreement()
    plot_metric_scores()

    print("\n" + "="*70)
    print("All graphs saved to:", OUTPUT_DIR)
    print("="*70)
    print("\nGraphs created:")
    print("[1] Judge Comparison - Compare Judge A, B, and Human pass rates")
    print("[2] Verdict Distribution - Pass/Partial/Fail breakdown")
    print("[3] Category Breakdown - Performance by question category")
    print("[4] AI vs Human Agreement - Judge consistency analysis")
    print("[5] Metric Scores - Relevance/Faithfulness/Correctness analysis")
    print("\nReady for slides and reports!\n")

if __name__ == '__main__':
    main()
