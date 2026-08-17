import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def load_data():
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Place Fake.csv and True.csv inside the data/ folder."
        )

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)
    df = df.drop_duplicates()

    # Use both title and article body when available.
    title = df["title"].fillna("") if "title" in df else ""
    body = df["text"].fillna("") if "text" in df else ""

    df["content"] = (title.astype(str) + " " + body.astype(str)).map(clean_text)
    df = df[df["content"].str.len() > 20].copy()

    return df[["content", "label"]]


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=42),
    }

    results = []
    trained = {}

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=200000,
            )),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
        })
        trained[name] = (pipeline, pred)

        print(f"\n{name}")
        print(classification_report(
            y_test, pred, target_names=["Fake", "Real"], zero_division=0
        ))

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["Model"]
    best_pipeline, best_pred = trained[best_name]

    joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib")

    # Model comparison chart
    plot_df = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]]
    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Fake News Detection - Model Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=180)
    plt.close()

    # Confusion matrix for best model
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    # Save a small text summary for the report/PPT.
    summary = [
        f"Dataset rows used: {len(df)}",
        f"Training rows: {len(X_train)}",
        f"Test rows: {len(X_test)}",
        f"Best model by F1: {best_name}",
        "",
        results_df.to_string(index=False),
    ]
    (RESULTS_DIR / "training_summary.txt").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print("\nTraining complete.")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved to: {MODEL_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
