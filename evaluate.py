from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(classification_report(y_test, preds))
    return acc


def plot_results(results):
    df = pd.DataFrame(results, columns=["Ngram", "Model", "Accuracy"])

    plt.figure(figsize=(12,6))
    sns.barplot(data=df, x="Ngram", y="Accuracy", hue="Model")
    plt.title("N-gram vs Model Accuracy Comparison")
    plt.show()