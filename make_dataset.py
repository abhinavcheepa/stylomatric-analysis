# make_dataset.py

import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from preprocess import clean_text

DATA_ROOT = "data_ocean"
OUTPUT_DIR = "dataset"

CHUNK_SIZE = 300
MIN_WORDS = 120

texts = []
labels = []

for author in os.listdir(DATA_ROOT):
    author_path = os.path.join(DATA_ROOT, author)

    if os.path.isdir(author_path):
        print(f"Processing author: {author}")

        for file in os.listdir(author_path):
            if file.endswith(".txt"):
                file_path = os.path.join(author_path, file)

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()

                    # ✅ CLEAN TEXT FIRST
                    content = clean_text(content)

                    words = content.split()

                    # Create chunks
                    for i in range(0, len(words), CHUNK_SIZE):
                        chunk_words = words[i:i + CHUNK_SIZE]

                        if len(chunk_words) >= MIN_WORDS:
                            chunk = " ".join(chunk_words)
                            texts.append(chunk)
                            labels.append(author.lower())

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

print(f"\nTotal samples created: {len(texts)}")

# Shuffle before split
combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("\n✅ Dataset created in /dataset folder")
print("\nSamples per author (train set):")
print(train_df["label"].value_counts())