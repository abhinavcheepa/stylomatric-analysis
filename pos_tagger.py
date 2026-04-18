# pos_tagger.py
# Loads the trained MuRIL model and predicts POS tags.
# Applies Look-back / Look-back-with-score fix from ACL paper.
# Does NOT retrain — loads from saved pos_model/ folder.

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from lookback import lookback, lookback_with_score

# ── Labels (same as paper) ───────────────────────────────────────────
label_list = [
    "ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN",
    "NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"
]
label2id = {l: i for i, l in enumerate(label_list)}
id2label  = {i: l for l, i in label2id.items()}

# ── Load trained model — NOT retraining from scratch ─────────────────
print("Loading POS model from pos_model/...")
tokenizer = AutoTokenizer.from_pretrained("pos_model")
model     = AutoModelForTokenClassification.from_pretrained("pos_model")
model.eval()   # set to evaluation mode — no dropout
print("Model loaded successfully!")


def predict_pos(sentence, method="lookback_with_score"):
    """
    Predict POS tags for a sentence using trained MuRIL model.

    Args:
        sentence : str  — input sentence
                   e.g. "राम बाजार जा रहा है"

        method   : str  — which prediction method to use:
                   "zero_shot"           → raw prediction, no fix
                   "lookback"            → basic look-back (paper Section 4.3)
                   "lookback_with_score" → best method from paper (default)

    Returns:
        list of (word, POS_tag) tuples
        e.g. [("राम", "PROPN"), ("बाजार", "NOUN"), ("जा", "VERB"), ...]
    """
    words = sentence.strip().split()

    if not words:
        return []

    # ── Tokenize ─────────────────────────────────────────────────────
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    word_ids = enc.word_ids()   # maps each token → original word index
                                # None for [CLS], [SEP], padding tokens

    # ── Run model ────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(**enc)

    logits      = outputs.logits[0]            # [seq_len, num_labels]
    predictions = torch.argmax(logits, dim=1)  # [seq_len] — best tag per token

    result = []

    # ── Method 1: Zero-shot ──────────────────────────────────────────
    # Raw prediction — just take first subword tag, no fix applied
    if method == "zero_shot":
        seen = set()
        for i, wid in enumerate(word_ids):
            if wid is not None and wid not in seen:
                tag = id2label[predictions[i].item()]
                result.append((words[wid], tag))
                seen.add(wid)

    # ── Method 2: Look-back ──────────────────────────────────────────
    # First subword tag applied to all subwords of the same word
    elif method == "lookback":
        raw_tags = [predictions[i].item() for i in range(len(word_ids))]
        fixed    = lookback(raw_tags, word_ids)

        seen      = set()
        valid_wids = [wid for wid in word_ids if wid is not None]
        for tag, wid in zip(fixed, valid_wids):
            if wid not in seen:
                result.append((words[wid], id2label[tag]))
                seen.add(wid)

    # ── Method 3: Look-back-with-score (BEST) ───────────────────────
    # Most confident subword tag applied to all subwords of the same word
    elif method == "lookback_with_score":
        fixed      = lookback_with_score(logits.numpy(), word_ids)
        seen       = set()
        valid_wids = [wid for wid in word_ids if wid is not None]
        for tag, wid in zip(fixed, valid_wids):
            if wid not in seen:
                result.append((words[wid], id2label[tag]))
                seen.add(wid)

    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Choose from: 'zero_shot', 'lookback', 'lookback_with_score'"
        )

    return result