
import json
import re
import random
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
    model,
    threshold=0.15
):
    """
    Predict intent and confidence score.

    If confidence is below the threshold,
    return 'unknown' intent.
    """

    processed_input = preprocess_text(
        user_input
    )

    input_vector = vectorizer.transform(
        [processed_input]
    )

    probabilities = model.predict_proba(
        input_vector
    )[0]

    best_index = probabilities.argmax()

    confidence = probabilities[best_index]

    predicted_intent = model.classes_[best_index]

    if confidence < threshold:
        predicted_intent = "unknown"

    return predicted_intent, confidence


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

def get_response(intent):
    """
    Get a response for the predicted intent
    from the knowledge base.
    """

    if intent == "unknown":
        return "Sorry, I don't have information about that."

    responses = knowledge_base.get(
        intent,
        {}
    ).get(
        "responses",
        []
    )

    if not responses:
        return "Sorry, I don't have a response for that."

    return random.choice(responses)


    # -----------------------------------
    # Test predictions
    # -----------------------------------



test_questions = [
    "what are the tuition fees",
    "do you provide accommodation",
    "how can I apply",
    "what courses are available",
    "tell me about career opportunities",
    "what is the weather today",
    "who is the prime minister",
    "how do I cook rice"
]

print("\nTesting Logistic Regression predictions...")

for question in test_questions:

    predicted_intent, confidence = predict_intent(
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

    print(
        f"Confidence score: {confidence:.4f}"
    )

    response = get_response(
    predicted_intent
)

    print(
    f"Response: {response}"
)