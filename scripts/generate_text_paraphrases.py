# scripts/generate_text_paraphrases.py
"""Generate text paraphrases for TextBraTS dataset using Claude API.

Reads expert text descriptions and generates N paraphrases per case,
preserving clinical meaning while varying wording, sentence order,
and detail level.

Usage:
    python scripts/generate_text_paraphrases.py <data_dir> [--n 10] [--dry-run]

Output:
    <case_dir>/<case_name>_paraphrase_01.txt
    <case_dir>/<case_name>_paraphrase_02.txt
    ...
"""

import os
import sys
import argparse
import time


PARAPHRASE_PROMPT = """You are a neuroradiologist writing clinical MRI reports.
Rewrite the following brain tumor MRI description in a different style while
preserving ALL clinical information (location, signal characteristics, tumor
components, mass effect). Vary the sentence structure, word choice, and level
of detail. Do NOT add or remove any clinical findings.

Original description:
{text}

Write exactly one rewritten version. Output only the rewritten text, nothing else."""


def generate_paraphrases(text: str, n: int = 10, model: str = "claude-sonnet-4-20250514") -> list:
    """Generate N paraphrases of a clinical text using Claude API."""
    try:
        import anthropic
    except ImportError:
        print("Error: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()
    paraphrases = []

    for i in range(n):
        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0.9,
            messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        )
        paraphrases.append(response.content[0].text.strip())
        time.sleep(0.5)

    return paraphrases


def main():
    parser = argparse.ArgumentParser(description="Generate text paraphrases")
    parser.add_argument("data_dir", help="BraTS data directory")
    parser.add_argument("--n", type=int, default=10, help="Paraphrases per case")
    parser.add_argument("--dry-run", action="store_true", help="Print first case only")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    cases = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    )

    for i, case_name in enumerate(cases):
        case_dir = os.path.join(args.data_dir, case_name)
        text_file = os.path.join(case_dir, f"{case_name}_flair_text.txt")

        if not os.path.exists(text_file):
            continue

        with open(text_file, "r", encoding="utf-8") as f:
            original = f.read().strip()

        if args.dry_run:
            print(f"Case: {case_name}")
            print(f"Original ({len(original)} chars): {original[:100]}...")
            paraphrases = generate_paraphrases(original, n=1, model=args.model)
            print(f"Paraphrase: {paraphrases[0][:100]}...")
            break

        paraphrases = generate_paraphrases(original, n=args.n, model=args.model)

        for j, para in enumerate(paraphrases):
            out_path = os.path.join(case_dir, f"{case_name}_paraphrase_{j+1:02d}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(para)

        print(f"[{i+1}/{len(cases)}] {case_name}: {len(paraphrases)} paraphrases")


if __name__ == "__main__":
    main()
