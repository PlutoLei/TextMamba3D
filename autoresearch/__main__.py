"""AutoResearch CLI.

Usage:
    python -m autoresearch status
    python -m autoresearch next
    python -m autoresearch generate-l0
    python -m autoresearch record ID '{"dice_mean": 0.85}'
    python -m autoresearch hypothesize
"""
import json
import sys

from autoresearch.orchestrator import Orchestrator

EXPERIMENTS = 'autoresearch/experiments.json'
RESULTS = 'autoresearch/results.json'


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    orch = Orchestrator(EXPERIMENTS, RESULTS)

    if cmd == 'status':
        print(orch.status())

    elif cmd == 'next':
        exp = orch.next_experiment()
        if exp:
            print(json.dumps(exp, indent=2))
        else:
            print("Queue empty.")

    elif cmd == 'generate-l0':
        from autoresearch.generate_l0 import generate_l0_notebook
        path = generate_l0_notebook('TextMamba3D_AutoResearch_L0.ipynb')
        print(f'Generated: {path}')

    elif cmd == 'record' and len(sys.argv) >= 4:
        exp_id = sys.argv[2]
        metrics = json.loads(sys.argv[3])
        orch.record_result(exp_id, metrics)
        improved = orch.is_improvement(metrics)
        print(f"Recorded {exp_id}: {'IMPROVED!' if improved else 'no improvement'}")
        if orch.should_train_from_scratch():
            print("WARNING: 3 consecutive fine-tune failures. Consider training from scratch.")

    elif cmd == 'hypothesize':
        from autoresearch.hypothesis_engine import generate_hypothesis
        hypothesis = generate_hypothesis(orch.results)
        if hypothesis:
            print(json.dumps(hypothesis, indent=2))
            if hypothesis.get('needs_approval'):
                print("*** Layer 2 experiment. Needs your approval. ***")
        else:
            print("Set ANTHROPIC_API_KEY to enable hypothesis generation.")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == '__main__':
    main()
