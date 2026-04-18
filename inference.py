
import os
from preprocess import load_all_text, split_sentences
from pos_tagger import predict_pos

OUTPUT_FILE = os.environ.get("POS_OUTPUT_FILE", "results/predictions.txt")

output_dir = os.path.dirname(OUTPUT_FILE) or "results"
os.makedirs(output_dir, exist_ok=True)

# ── Choose method from ACL paper ─────────────────────────────────────
# "zero_shot"           → raw prediction, no fix
# "lookback"            → basic look-back fix (paper Section 4.3)
# "lookback_with_score" → best method from paper (+1% F1 over look-back)
METHOD = os.environ.get("POS_METHOD", "lookback_with_score")

print(f"Running inference with method : {METHOD}")
print(f"Output will be saved to       : {OUTPUT_FILE}\n")

# ── Load all text files from data/ folder ────────────────────────────
texts = load_all_text()

if not texts:
    print("No .txt files found in data/ folder.")
    print("Please add Hindi .txt files inside the data/ folder and try again.")
    exit()

total_sentences = 0
total_words     = 0

# ── Run POS tagging ──────────────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for text_idx, text in enumerate(texts):

        print(f"Processing file {text_idx + 1} of {len(texts)}...")

        sentences = split_sentences(text)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # predict POS tags using chosen method
            results = predict_pos(sent, method=METHOD)

            # write word TAB tag per line
            for word, tag in results:
                f.write(f"{word}\t{tag}\n")
                total_words += 1

            # blank line between sentences
            f.write("\n")
            total_sentences += 1

# ── Summary ──────────────────────────────────────────────────────────
print("\nInference completed!")
print(f"Total files processed  : {len(texts)}")
print(f"Total sentences tagged : {total_sentences}")
print(f"Total words tagged     : {total_words}")
print(f"Predictions saved to   : {OUTPUT_FILE}")


