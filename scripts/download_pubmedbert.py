"""Download PubMedBERT model and tokenizer to local directory.

Run this script on a machine with internet access before training.
The downloaded model will be saved to ./pretrained/pubmedbert/.

Usage:
    python scripts/download_pubmedbert.py
    python scripts/download_pubmedbert.py --output ./pretrained/pubmedbert
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Download PubMedBERT for offline use")
    parser.add_argument(
        "--output",
        type=str,
        default="./pretrained/pubmedbert",
        help="Output directory (default: ./pretrained/pubmedbert)",
    )
    args = parser.parse_args()

    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

    print(f"Downloading PubMedBERT from HuggingFace: {model_name}")
    print(f"Saving to: {os.path.abspath(args.output)}")

    from transformers import AutoModel, AutoTokenizer

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(args.output)
    print("Tokenizer saved.")

    print("Downloading model weights (~440MB)...")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(args.output)
    print("Model saved.")

    print(f"Done. Set text_model_path: \"{args.output}\" in your config YAML.")


if __name__ == "__main__":
    main()
