#!/usr/bin/env python
"""
Start Human-in-the-Loop Evaluation Web Interface
"""
import os
import sys
import webbrowser
from time import sleep

def main():
    print("\n" + "="*70)
    print("🔍 RAG Human-in-the-Loop Evaluation System")
    print("="*70)

    # Check if Flask is installed
    try:
        import flask
        print("✓ Flask is installed")
    except ImportError:
        print("✗ Flask not found. Installing...")
        os.system(f"{sys.executable} -m pip install flask")

    # Check if raw_results.json exists (in parent/results/)
    eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(eval_dir, 'results', 'raw_results.json')

    if not os.path.exists(results_file):
        print("\n✗ Error: results/raw_results.json not found")
        print("  Run evaluation first: python judges/evaluate.py")
        sys.exit(1)

    print(f"✓ Evaluation results found ({results_file})")

    # Create templates directory if needed
    os.makedirs('templates', exist_ok=True)

    # Start Flask server
    print("\n" + "="*70)
    print("Starting web server on http://localhost:5000")
    print("="*70)
    print("\nPress CTRL+C to stop the server\n")

    # Give user time to read message before opening browser
    sleep(2)

    # Open browser
    try:
        webbrowser.open('http://localhost:5000')
        print("✓ Browser opened automatically")
    except:
        print("⚠ Could not open browser automatically")
        print("  Open http://localhost:5000 manually")

    # Import and run app
    try:
        from human_review_app import app
        app.run(debug=True, port=5000, use_reloader=False)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
