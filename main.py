from prepare_dataset import load_data, get_ngram_features
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# ================== RESULT FOLDER SETUP ==================
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULT_DIR, run_id)
os.makedirs(RUN_DIR, exist_ok=True)

# ================== PATHS ==================
TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"

# ================== LOAD DATA ==================
train_texts, train_labels = load_data(TRAIN_PATH)
test_texts, test_labels = load_data(TEST_PATH)

# ================== ENCODE LABELS (IMPORTANT FOR XGBOOST) ==================
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
test_labels = le.transform(test_labels)

labels = le.classes_

# ================== HELPER FUNCTION ==================
def save_confusion_matrix(y_test, y_pred, labels, name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(os.path.join(RUN_DIR, f"confusion_matrix_{name}.png"))
    plt.close()

# ================== NGRAM SETTINGS ==================
ngram_settings = [
    ("Unigram", (1,1)),
    ("Bigram", (1,2)),
    ("Trigram", (1,3)),
]

# ================== MODELS ==================
models = {
    "LogisticRegression": LogisticRegression(max_iter=300),
    "LinearSVM": LinearSVC(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# ================== TRACK ACCURACY FOR GRAPH ==================
model_names = []
accuracies = []

# ================== TRAINING LOOP ==================
for ngram_name, ngram in ngram_settings:
    print(f"\n\n========== Using {ngram_name} ==========")

    X_train, X_test, vectorizer = get_ngram_features(train_texts, test_texts, "tfidf", ngram)

    # Save vocabulary
    vocab = vectorizer.get_feature_names_out()
    with open(os.path.join(RUN_DIR, f"vocabulary_{ngram_name}.txt"), "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n")

    for model_name, model in models.items():
        print(f"\nTraining {model_name} with {ngram_name}")

        model.fit(X_train, train_labels)
        y_pred = model.predict(X_test)

        acc = accuracy_score(test_labels, y_pred)
        print("Accuracy:", acc)

        # Save accuracy for graph
        model_names.append(f"{model_name}_{ngram_name}")
        accuracies.append(acc)

        # Save classification report
        report = classification_report(test_labels, y_pred, target_names=labels, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        report_path_csv = os.path.join(RUN_DIR, f"report_{model_name}_{ngram_name}.csv")
        report_path_txt = os.path.join(RUN_DIR, f"report_{model_name}_{ngram_name}.txt")

        report_df.to_csv(report_path_csv)

        with open(report_path_txt, "w", encoding="utf-8") as f:
            f.write(classification_report(test_labels, y_pred, target_names=labels))

        # Save confusion matrix image
        save_confusion_matrix(test_labels, y_pred, labels, f"{model_name}_{ngram_name}")

# ================== ACCURACY COMPARISON GRAPH ==================
plt.figure(figsize=(12,6))
plt.bar(model_names, accuracies)
plt.xticks(rotation=45)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "accuracy_comparison.png"))
plt.close()

print(f"\n✅ All results saved inside: {RUN_DIR}")