
import importlib.util
import sys

from huggingface_hub import hf_hub_download

MODEL_ID = "Horizon-Labs/punctuation-restoration-small"


def load_model():
    print("Downloading/loading punctuation model...")

    # Download the official punctuation helper
    helper_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="punctuate.py"
    )

    # Import the downloaded helper
    spec = importlib.util.spec_from_file_location(
        "punctuate",
        helper_path
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules["punctuate"] = module
    spec.loader.exec_module(module)

    # Load the pretrained model on CPU
    model = module.Punctuator(MODEL_ID)

    print("Model loaded successfully!")
    return model


def main():
    model = load_model()

    # First test input
    hindi_text = """
    भारत एक कृषि प्रधान देश है
    यहाँ अनेक प्रकार की फसलें उगाई जाती हैं
    क्या आप भारत के बारे में जानते हैं
    """

    print("\nOriginal text:")
    print(hindi_text)

    # Restore punctuation
    result = model(hindi_text)

    print("\nPunctuated text:")
    print(result)

    # Save output
    with open(
        "outputs/punctuated_test.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(str(result))

    print("\nOutput saved successfully!")


if __name__ == "__main__":
    main()
