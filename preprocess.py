# preprocess.py

import re

def clean_text(text):
    """
    Basic text normalization for stylometry.
    Works safely for Hindi + English text.
    """
    text = text.lower()

    # remove numbers
    text = re.sub(r'\d+', ' ', text)

    # remove punctuation but keep Hindi characters
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text