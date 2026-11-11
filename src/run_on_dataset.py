
import importlib.util
import sys
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_ID = "Horizon-Labs/punctuation-restoration-small"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_model():
    helper_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="punctuate.py"
    )

    spec = importlib.util.spec_from_file_location(
        "punctuate",
        helper_path
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules["punctuate"] = module
    spec.loader.exec_module(module)

    return module.Punctuator(MODEL_ID)


def clean_text(text):
    # Remove NULL placeholders, preserving word boundaries
    text = re.sub(r"\bNULL\b", " ", text)

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    model = load_model()

    files = sorted(DATA_DIR.glob("*.txt"))

    if not files:
        print("No text files found in data folder.")
        return

    for input_file in files:

        print(f"\nProcessing: {input_file.name}")

        text = input_file.read_text(encoding="utf-8")
        text = clean_text(text)

        punctuated_text = model(text)

        output_file = OUTPUT_DIR / (
            input_file.stem + "_punctuated.txt"
        )

        output_file.write_text(
            str(punctuated_text),
            encoding="utf-8"
        )

        print(f"Saved: {output_file.name}")

    print("\nAll files processed!")


if __name__ == "__main__":
    main()
