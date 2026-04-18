# lookback.py
# Implementation of both Look-back methods from the ACL paper:
# Section 4.3 — "Investigating the impact of sub-word tokenization"
#
# WHY THIS IS NEEDED:
# When MuRIL/DistilBERT tokenizes a word, it may split it into
# multiple subword tokens. Each subword gets its own POS tag,
# which causes inconsistency. Example:
#   Angika word "dEkhAibae" → split into "dEkh" + "ibae"
#   "dEkh" gets VERB tag (correct)
#   "ibae" gets NOUN tag (wrong)
#
# SOLUTION FROM PAPER:
# Look-back        → use tag of FIRST subword for all subwords
# Look-back-with-score → use tag of MOST CONFIDENT subword for all subwords

import torch


def lookback(tags, word_ids):
    """
    Basic Look-back (paper Section 4.3).

    All subword tokens of a word get the POS tag of the FIRST subword.
    Reason: first subword is closest to the original Hindi/high-resource word,
    so its tag is most reliable.

    Args:
        tags     : list of predicted tag ids (one per token including subwords)
        word_ids : list of word indices from tokenizer.word_ids()
                   None = special token ([CLS], [SEP], padding)

    Returns:
        final_tags : list of tag ids, one per original word (subwords merged)

    Example:
        word_ids = [None, 0, 0, 1, 2, 2, None]
        tags     = [-100, 15, 7, 3, 8, 12, -100]
        output   = [15, 3, 8]   ← first subword tag used for each word
    """
    final_tags = []
    prev_word  = None
    prev_tag   = None

    for tag, wid in zip(tags, word_ids):
        if wid is None:
            continue                        # skip [CLS], [SEP], padding
        if wid != prev_word:
            prev_tag = tag
            final_tags.append(tag)          # first subword → keep its tag
        else:
            final_tags.append(prev_tag)     # other subwords → use first tag
        prev_word = wid

    return final_tags


def lookback_with_score(logits, word_ids):
    """
    Look-back-with-score (paper Section 4.3).

    Among ALL subword tokens of a word, find the one with the
    HIGHEST confidence score (max softmax probability).
    Assign that winning tag to ALL subwords of the word.

    This is the BETTER variant — achieves +1% F1 over basic look-back
    on average across Angika, Magahi, and Bhojpuri (Table 1 in paper).

    Args:
        logits   : numpy array or tensor of shape [seq_len, num_labels]
                   raw model output before softmax
        word_ids : list of word indices from tokenizer.word_ids()
                   None = special token

    Returns:
        final_tags : list of tag ids, one per original word (subwords merged)

    Example:
        Word "dEkhAibae" split into 2 subwords:
          subword 0 → logits → max confidence 0.91 → tag VERB  ✓ winner
          subword 1 → logits → max confidence 0.65 → tag NOUN
        Result → both subwords get VERB tag
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    # Softmax to convert logits → confidence probabilities
    probs = torch.softmax(logits, dim=-1)   # [seq_len, num_labels]

    # Step 1: Group token indices by their word id
    word_to_indices = {}
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in word_to_indices:
            word_to_indices[wid] = []
        word_to_indices[wid].append(idx)

    # Step 2: For each word → find subword with highest max confidence
    word_best_tag = {}
    for wid, indices in word_to_indices.items():
        # pick the subword token with the highest confidence score
        best_idx = max(indices, key=lambda i: probs[i].max().item())
        best_tag = probs[best_idx].argmax().item()
        word_best_tag[wid] = best_tag

    # Step 3: Build final tag list in word order (one tag per word)
    final_tags = []
    seen_words = set()
    for wid in word_ids:
        if wid is None:
            continue
        if wid not in seen_words:
            final_tags.append(word_best_tag[wid])
            seen_words.add(wid)

    return final_tags