import json
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Load knowledge base
with open("knowledge_base.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)


# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """Preprocess text using tokenization, stop-word removal and lemmatization."""

    text = text.lower()

    tokens = nltk.word_tokenize(text)

    processed_tokens = []

    for token in tokens:

        if token.isalnum() and token not in stop_words:

            lemma = lemmatizer.lemmatize(token)

            processed_tokens.append(lemma)

    return " ".join(processed_tokens)


def prepare_training_data():
    """Prepare training texts and intent labels."""

    training_texts = []
    intent_labels = []

    for intent, data in knowledge_base.items():

        if intent == "unknown":
            continue

        for pattern in data["patterns"]:

            processed_pattern = preprocess_text(pattern)

            training_texts.append(processed_pattern)
            intent_labels.append(intent)

    return training_texts, intent_labels


def load_test_data():
    """Load unseen test questions and their intent labels."""

    with open("test_dataset.json", "r", encoding="utf-8") as file:
        test_data = json.load(file)

    test_questions = []
    test_labels = []

    for item in test_data:

        test_questions.append(
            preprocess_text(item["question"])
        )

        test_labels.append(
            item["intent"]
        )

    return test_questions, test_labels


if __name__ == "__main__":

    print("=" * 60)
    print("             MODEL EVALUATION")
    print("=" * 60)

    # -----------------------------------
    # Prepare Training Data
    # -----------------------------------

    X, y = prepare_training_data()

    print(f"\nTotal training examples: {len(X)}")
    print(f"Total intent categories: {len(set(y))}")

    # -----------------------------------
    # TF-IDF Vectorization
    # -----------------------------------

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    X_tfidf = vectorizer.fit_transform(X)

    print(f"TF-IDF matrix shape: {X_tfidf.shape}")

    # -----------------------------------
    # Train Logistic Regression
    # -----------------------------------

    print("\nTraining Logistic Regression model...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_tfidf, y)

    print("Model training completed successfully.")

    # -----------------------------------
    # Load Test Dataset
    # -----------------------------------

    test_questions, test_labels = load_test_data()

    print(f"\nTotal test examples: {len(test_questions)}")

    # -----------------------------------
    # Transform Test Data
    # -----------------------------------

    X_test_tfidf = vectorizer.transform(
        test_questions
    )

    # -----------------------------------
    # Predictions
    # -----------------------------------

    predictions = model.predict(
        X_test_tfidf
    )

    # -----------------------------------
    # Evaluation Metrics
    # -----------------------------------

    accuracy = accuracy_score(
        test_labels,
        predictions
    )

    precision = precision_score(
        test_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        test_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        test_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("             EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # -----------------------------------
    # Classification Report
    # -----------------------------------

    print("\n" + "=" * 60)
    print("             CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            test_labels,
            predictions,
            zero_division=0
        )
    )

    # -----------------------------------
    # Confusion Matrix
    # -----------------------------------

    print("=" * 60)
    print("             CONFUSION MATRIX")
    print("=" * 60)

    matrix = confusion_matrix(
        test_labels,
        predictions
    )

    print(matrix)

    # -----------------------------------
    # Confusion Matrix Visualization
    # -----------------------------------

    labels = sorted(set(test_labels))

    plt.figure(figsize=(8, 6))

    plt.imshow(matrix)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Intent")
    plt.ylabel("Actual Intent")

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(labels)),
        labels
    )

    for i in range(len(labels)):
        for j in range(len(labels)):

            plt.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        "outputs/confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("\nConfusion matrix saved to:")
    print("outputs/confusion_matrix.png")

    print("\nModel evaluation completed successfully.")