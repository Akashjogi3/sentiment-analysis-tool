
from datasets import load_dataset
from pathlib import Path

# Download the IMDb movie review dataset
print("Downloading IMDb dataset. Please wait...")

dataset = load_dataset("stanfordnlp/imdb")

base_folder = Path("aclImdb")

# Create folders for training and testing
for split in ["train", "test"]:
    for sentiment in ["pos", "neg"]:
        (base_folder / split / sentiment).mkdir(
            parents=True,
            exist_ok=True
        )

# Save reviews as individual text files
for split in ["train", "test"]:
    print(f"Saving {split} reviews...")

    for index, item in enumerate(dataset[split]):
        sentiment = "pos" if item["label"] == 1 else "neg"

        file_path = (
            base_folder / split / sentiment / f"{index}.txt"
        )

        file_path.write_text(
            item["text"],
            encoding="utf-8"
        )

print("Dataset downloaded and saved successfully!")
print("You can now run sentiment_gui.py.")
