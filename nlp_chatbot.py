
import json
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression


# Load knowledge base
with open("knowledge_base.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)


# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Preprocess text using:
    1. Lowercase conversion
    2. Punctuation removal
    3. Tokenization
    4. Stop-word removal
    5. Lemmatization
    """

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    tokens = nltk.word_tokenize(text)

    processed_tokens = []

    for token in tokens:

        if token not in stop_words:

            lemma = lemmatizer.lemmatize(token)

            processed_tokens.append(lemma)

    return " ".join(processed_tokens)


def prepare_training_data():
    """
    Extract training patterns and their corresponding intents.
    """

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


def calculate_similarity(
    user_input,
    training_texts,
    vectorizer,
    tfidf_matrix
):
    """
    Calculate cosine similarity between the user input
    and all training examples.
    """

    processed_input = preprocess_text(user_input)

    input_vector = vectorizer.transform(
        [processed_input]
    )

    similarity_scores = cosine_similarity(
        input_vector,
        tfidf_matrix
    )

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[
        0
    ][best_match_index]

    return best_match_index, best_score


def train_model(tfidf_matrix, labels):
    """
    Train Logistic Regression classifier.
    """

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(tfidf_matrix, labels)

    return model


def predict_intent(
    user_input,
    vectorizer,
    model
):
    """
    Predict the intent of a new user input.
    """

    processed_input = preprocess_text(
        user_input
    )

    input_vector = vectorizer.transform(
        [processed_input]
    )

    predicted_intent = model.predict(
        input_vector
    )[0]

    return predicted_intent


if __name__ == "__main__":

    # -----------------------------------
    # Prepare training data
    # -----------------------------------

    X, y = prepare_training_data()

    print("=" * 60)
    print("       NLP INTENT CLASSIFIER")
    print("=" * 60)

    print(
        f"Total training examples: {len(X)}"
    )

    print(
        f"Total intent categories: {len(set(y))}"
    )


    # -----------------------------------
    # TF-IDF Vectorization
    # -----------------------------------

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(X)

    print(
        f"TF-IDF matrix shape: {tfidf_matrix.shape}"
    )


    # -----------------------------------
    # Train Logistic Regression
    # -----------------------------------

    print("\nTraining Logistic Regression model...")

    model = train_model(
        tfidf_matrix,
        y
    )

    print("Model training completed successfully.")


    # -----------------------------------
    # Test predictions
    # -----------------------------------

    test_questions = [
        "what are the tuition fees",
        "do you provide accommodation",
        "how can I apply",
        "what courses are available",
        "tell me about career opportunities"
    ]

    print("\nTesting Logistic Regression predictions...")

    for question in test_questions:

        predicted_intent = predict_intent(
            question,
            vectorizer,
            model
        )

        print(
            f"\nQuestion: {question}"
        )

        print(
            f"Predicted intent: {predicted_intent}"
        )