import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


DATA_PATH = "emails_priority.csv"
RANDOM_STATE = 42


def load_data(path: str):
    """Load and validate the email dataset."""
    df = pd.read_csv(path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    return df["text"], df["label"]


def build_model():
    """Create the TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    multi_class="multinomial",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate(model, X, y):
    """Train the model and print evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, predictions))

    return model


def predict_urgency(model, email_text: str):
    """
    Predict email priority and urgency score.

    Returns:
        predicted_class (int): 0 = Low, 1 = Medium, 2 = High
        urgency_score (float): 0–100
    """
    probabilities = model.predict_proba([email_text])[0]

    expected_priority = np.dot(probabilities, [0, 1, 2])
    urgency_score = round((expected_priority / 2) * 100, 1)

    predicted_class = int(np.argmax(probabilities))

    return predicted_class, urgency_score


def demo_predictions(model):
    """Run sample predictions for demonstration."""
    test_emails = [
        "Final exam has been rescheduled to tomorrow morning",
        "Career fair with tech companies next week",
        "Student club meeting tonight",
        "Your tuition payment is overdue",
        "Free donuts at the library",
    ]

    label_map = {0: "Low", 1: "Medium", 2: "High"}

    print("\n=== Sample Predictions ===")
    for email in test_emails:
        label, score = predict_urgency(model, email)

        print(f"\nEmail: {email}")
        print(f"Predicted Priority: {label_map[label]} ({label})")
        print(f"Urgency Score: {score}/100")


def main():
    X, y = load_data(DATA_PATH)
    model = build_model()
    trained_model = train_and_evaluate(model, X, y)
    demo_predictions(trained_model)


if __name__ == "__main__":
    main()
