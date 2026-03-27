"""Claude API integration for hypothesis generation."""
import json
import os
import re


def format_results_for_analysis(results: dict) -> str:
    lines = [
        "# TextMamba3D AutoResearch Results",
        "",
        f"## Baseline: V5.0 Mean Dice = {results['baseline']['mean_dice']:.4f}",
        f"  ET={results['baseline'].get('dice_ET', 'N/A')}, "
        f"TC={results['baseline'].get('dice_TC', 'N/A')}, "
        f"WT={results['baseline'].get('dice_WT', 'N/A')}",
        "",
        "## Experiments (chronological):",
    ]
    for exp in results.get('experiments', []):
        m = exp['metrics']
        status = 'IMPROVED' if exp.get('improved') else 'no improvement'
        lines.append(
            f"- {exp['id']}: Mean={m.get('dice_mean', 0):.4f} "
            f"ET={m.get('dice_ET', 0):.4f} [{status}]"
        )
    return '\n'.join(lines)


SYSTEM_PROMPT = """You are an ML research assistant for brain tumor segmentation.
Given experiment results, propose the next experiment to try.

Rules:
- Layer 0 (inference-only): post-processing params, TTA config, probability calibration
- Layer 1 (training): learning rate, loss weights, augmentation settings
- Layer 2 (architecture): new modules, loss functions — mark as NEEDS_APPROVAL

Output format (JSON):
{
  "id": "L1-xxx",
  "layer": 1,
  "type": "train",
  "name": "short description",
  "rationale": "why this might work based on results so far",
  "params": { ... config overrides ... },
  "needs_approval": false
}"""


def generate_hypothesis(results: dict, api_key: str | None = None) -> dict | None:
    if api_key is None:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = format_results_for_analysis(results)
        response = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': prompt}],
        )
        text = response.content[0].text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Hypothesis generation failed: {e}")
    return None
