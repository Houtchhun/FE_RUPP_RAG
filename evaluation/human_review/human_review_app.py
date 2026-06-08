"""
Human-in-the-Loop Evaluation Interface
Allows human reviewers to validate/correct AI judge verdicts
"""
import json
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import sys

app = Flask(__name__, template_folder='templates')

# Use absolute paths
HUMAN_REVIEW_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(HUMAN_REVIEW_DIR)
RESULTS_FILE = os.path.join(EVAL_DIR, 'results', 'raw_results.json')
FEEDBACK_FILE = os.path.join(EVAL_DIR, 'results', 'human_feedback.jsonl')

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

def save_feedback(data):
    """Save or update feedback entry"""
    feedback = load_feedback()
    feedback[data['question_id']] = data
    with open(FEEDBACK_FILE, 'w') as f:
        for qid in sorted(feedback.keys()):
            f.write(json.dumps(feedback[qid]) + '\n')

@app.route('/')
def dashboard():
    results = load_results()
    feedback = load_feedback()

    total = len(results)
    reviewed = len(feedback)
    pending = total - reviewed

    return render_template('dashboard.html',
                          total=total,
                          reviewed=reviewed,
                          pending=pending)

@app.route('/review')
def review():
    return render_template('review.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/api/question/<int:question_id>')
def get_question(question_id):
    results = load_results()
    feedback = load_feedback()

    if question_id < 0 or question_id >= len(results):
        return jsonify({'error': 'Invalid question ID'}), 404

    result = results[question_id]
    human_verdict = feedback.get(question_id)

    return jsonify({
        'id': question_id,
        'category': result['category'],
        'question': result['question'],
        'ground_truth': result['ground_truth'],
        'context': result['context_preview'],
        'generated_answer': result['answer'],
        'groq_scores': result['groq_scores'],
        'cerebras_scores': result['cerebras_scores'],
        'groq_verdict': result['groq_verdict'],
        'cerebras_verdict': result['cerebras_verdict'],
        'human_verdict': human_verdict
    })

@app.route('/api/question/next-pending')
def get_next_pending():
    results = load_results()
    feedback = load_feedback()

    for i in range(len(results)):
        if i not in feedback:
            return jsonify({'id': i})

    return jsonify({'id': None})

@app.route('/api/submit-feedback', methods=['POST'])
def submit_feedback():
    data = request.json

    feedback_entry = {
        'question_id': data['question_id'],
        'human_verdict': data['verdict'],
        'confidence': data['confidence'],
        'notes': data['notes'],
        'timestamp': datetime.now().isoformat()
    }

    save_feedback(feedback_entry)
    return jsonify({'status': 'saved', 'id': data['question_id']})

@app.route('/api/statistics')
def get_statistics():
    results = load_results()
    feedback = load_feedback()

    agreement_groq = 0
    agreement_cerebras = 0
    ai_both_agree = 0
    ai_both_disagree = 0
    human_overrides = 0

    for qid in feedback:
        result = results[qid]
        hv = feedback[qid]['human_verdict']

        groq_agree = (result['groq_verdict'] == hv)
        cerebras_agree = (result['cerebras_verdict'] == hv)

        if groq_agree:
            agreement_groq += 1
        if cerebras_agree:
            agreement_cerebras += 1

        if result['groq_verdict'] == result['cerebras_verdict']:
            ai_both_agree += 1
            if result['groq_verdict'] != hv:
                human_overrides += 1
        else:
            ai_both_disagree += 1

    total_reviewed = len(feedback)
    total_questions = len(results)

    return jsonify({
        'total_reviewed': total_reviewed,
        'total_questions': total_questions,
        'agreement_groq': f"{agreement_groq/total_reviewed*100:.1f}%" if total_reviewed > 0 else "0%",
        'agreement_cerebras': f"{agreement_cerebras/total_reviewed*100:.1f}%" if total_reviewed > 0 else "0%",
        'ai_both_agree': ai_both_agree,
        'ai_both_disagree': ai_both_disagree,
        'human_overrides': human_overrides
    })

@app.route('/api/disagreements')
def get_disagreements():
    results = load_results()
    feedback = load_feedback()

    disagreements = []
    for qid in feedback:
        result = results[qid]
        hv = feedback[qid]['human_verdict']

        if result['groq_verdict'] != hv or result['cerebras_verdict'] != hv:
            disagreements.append({
                'id': qid,
                'question': result['question'],
                'category': result['category'],
                'groq_verdict': result['groq_verdict'],
                'cerebras_verdict': result['cerebras_verdict'],
                'human_verdict': hv,
                'notes': feedback[qid].get('notes', '')
            })

    return jsonify(disagreements)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
