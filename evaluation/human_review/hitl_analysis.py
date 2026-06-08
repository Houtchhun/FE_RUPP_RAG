"""
Human-in-the-Loop Analysis
Compare AI judge verdicts with human reviewer judgments
"""
import json
import os
from collections import defaultdict
from datetime import datetime

# Use absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_FILE = os.path.join(EVAL_DIR, 'results', 'raw_results.json')
FEEDBACK_FILE = os.path.join(EVAL_DIR, 'results', 'human_feedback.jsonl')
ANALYSIS_FILE = os.path.join(EVAL_DIR, 'results', 'hitl_analysis.json')

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

def analyze():
    results = load_results()
    feedback = load_feedback()

    if not feedback:
        print("No human feedback yet. Start reviewing to generate analysis.")
        return

    # Metrics
    agreement_groq = 0
    agreement_cerebras = 0
    ai_both_agree = 0
    ai_both_disagree = 0
    human_overrides = 0

    disagreements_by_category = defaultdict(list)
    disagreements_by_type = defaultdict(list)

    total_reviewed = len(feedback)

    for qid in feedback:
        result = results[qid]
        hv = feedback[qid]['human_verdict']

        groq_agree = (result['groq_verdict'] == hv)
        cerebras_agree = (result['cerebras_verdict'] == hv)

        if groq_agree:
            agreement_groq += 1
        if cerebras_agree:
            agreement_cerebras += 1

        # Disagreement analysis
        if not groq_agree or not cerebras_agree:
            category = result['category']
            disagreements_by_category[category].append({
                'question_id': qid,
                'question': result['question'],
                'groq_verdict': result['groq_verdict'],
                'cerebras_verdict': result['cerebras_verdict'],
                'human_verdict': hv,
                'notes': feedback[qid].get('notes', ''),
                'confidence': feedback[qid].get('confidence', 0)
            })

        # Pattern: both AI agree but human disagrees
        if result['groq_verdict'] == result['cerebras_verdict']:
            ai_both_agree += 1
            if result['groq_verdict'] != hv:
                human_overrides += 1
                disagreement_type = f"Both AI say {result['groq_verdict'].upper()}, Human says {hv.upper()}"
                disagreements_by_type[disagreement_type].append(qid)
        else:
            ai_both_disagree += 1

    # Calculate statistics
    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_reviewed': total_reviewed,
        'total_questions': len(results),
        'coverage_percent': round(total_reviewed / len(results) * 100, 1),
        'agreement': {
            'groq_percent': round(agreement_groq / total_reviewed * 100, 1) if total_reviewed > 0 else 0,
            'cerebras_percent': round(agreement_cerebras / total_reviewed * 100, 1) if total_reviewed > 0 else 0,
            'both_agree': ai_both_agree,
            'both_disagree': ai_both_disagree,
            'human_overrides': human_overrides
        },
        'disagreements_by_category': {k: len(v) for k, v in disagreements_by_category.items()},
        'disagreements_by_type': {k: len(v) for k, v in disagreements_by_type.items()},
        'top_issues': sorted([(k, len(v)) for k, v in disagreements_by_type.items()], key=lambda x: x[1], reverse=True)[:5]
    }

    # Save analysis
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*70)
    print("HUMAN-IN-THE-LOOP EVALUATION ANALYSIS")
    print("="*70)
    print(f"\nReview Progress: {total_reviewed}/{len(results)} ({stats['coverage_percent']}%)")
    print(f"\nJudge Agreement with Humans:")
    print(f"  Judge A (Groq Qwen3-32B):      {stats['agreement']['groq_percent']}%")
    print(f"  Judge B (Cerebras Qwen3-235B): {stats['agreement']['cerebras_percent']}%")
    print(f"\nAI Agreement Patterns:")
    print(f"  Both judges agree: {ai_both_agree} cases")
    print(f"  Both judges disagree: {ai_both_disagree} cases")
    print(f"  Human overrides (both AI wrong): {human_overrides} cases")

    if disagreements_by_category:
        print(f"\nDisagreements by Category:")
        for category, items in sorted(disagreements_by_category.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {category}: {len(items)}")

    if stats['top_issues']:
        print(f"\nTop 5 Failure Patterns:")
        for i, (pattern, count) in enumerate(stats['top_issues'][:5], 1):
            print(f"  {i}. {pattern}: {count} cases")

    print(f"\nAnalysis saved to: {ANALYSIS_FILE}")
    print("="*70 + "\n")

    return stats

def print_detailed_disagreements(category_filter=None):
    """Print detailed disagreement analysis"""
    results = load_results()
    feedback = load_feedback()

    print("\nDETAILED DISAGREEMENT REPORT")
    print("="*70)

    for qid in feedback:
        result = results[qid]
        hv = feedback[qid]['human_verdict']

        if category_filter and result['category'] != category_filter:
            continue

        if result['groq_verdict'] != hv or result['cerebras_verdict'] != hv:
            print(f"\n[Q{qid+1}] {result['category'].upper()}")
            print(f"Question: {result['question']}")
            print(f"Judge A says: {result['groq_verdict'].upper()}")
            print(f"Judge B says: {result['cerebras_verdict'].upper()}")
            print(f"You say:     {hv.upper()}")
            if feedback[qid].get('notes'):
                print(f"Notes: {feedback[qid]['notes']}")
            print("-" * 70)

if __name__ == '__main__':
    analyze()
    # Uncomment to see detailed disagreements:
    # print_detailed_disagreements()
