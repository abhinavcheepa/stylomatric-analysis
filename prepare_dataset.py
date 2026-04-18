import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from preprocess import clean_text   # if you already have this

def load_data(path):
    df = pd.read_csv(path)

    # Apply your preprocessing
    df['text'] = df['text'].apply(clean_text)

    return df['text'], df['label']


def get_ngram_features(train_texts, test_texts, method="tfidf", ngram=(1,2)):
    if method == "bow":
        vectorizer = CountVectorizer(ngram_range=ngram)
    else:
        vectorizer = TfidfVectorizer(ngram_range=ngram)

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer