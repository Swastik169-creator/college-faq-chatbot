
import json
import re
import random
import nltk
import pickle

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

def save_model(
    model,
    vectorizer,
    filename="chatbot_tfidf.pkl"
):
    """
    Save the trained Logistic Regression model
    and TF-IDF vectorizer.
    """

    model_data = {
        "model": model,
        "vectorizer": vectorizer
    }

    with open(filename, "wb") as file:
        pickle.dump(model_data, file)

    print(
        f"\nModel saved successfully to: {filename}"
    )


def load_model(
    filename="chatbot_tfidf.pkl"
):
    """
    Load the saved model and TF-IDF vectorizer.
    """

    with open(filename, "rb") as file:
        model_data = pickle.load(file)

    model = model_data["model"]
    vectorizer = model_data["vectorizer"]

    print(
        f"Model loaded successfully from: {filename}"
    )

    return model, vectorizer


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

def test_novel_phrasings(
    model,
    vectorizer
):
    """
    Test the NLP model on five new phrasings
    that are different from the training patterns.
    """

    novel_questions = [
        "What steps should I follow to secure admission?",
        "How much money should I budget for my studies?",
        "Is there somewhere on campus where students can stay?",
        "Which academic programs can I choose from?",
        "What kind of career support do graduates receive?"
    ]

    print("\n" + "=" * 60)
    print("          NOVEL PHRASING TEST")
    print("=" * 60)

    for question in novel_questions:

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

    save_model(
    model,
    vectorizer
    )
    loaded_model, loaded_vectorizer = load_model()

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


def chat(
    model,
    vectorizer
):
    """
    Run the interactive NLP chatbot.
    """

    print("\n" + "=" * 60)
    print("          COLLEGEBOT - NLP COLLEGE ASSISTANT")
    print("=" * 60)

    print("\nType your question and press Enter.")
    print("Type 'bye' to exit.\n")

    while True:

        user_input = input("You: ").strip()

        # Handle empty input
        if not user_input:
            print("Bot: Please type something.\n")
            continue

        # Predict intent and confidence
        predicted_intent, confidence = predict_intent(
            user_input,
            vectorizer,
            model
        )

        # Debug information
        print(
            f"[NLP detected intent: {predicted_intent}]"
        )

        print(
            f"[Confidence: {confidence:.4f}]"
        )

        # Get response
        response = get_response(
            predicted_intent
        )

        print(
            f"Bot: {response}\n"
        )

        # Exit on farewell
        if predicted_intent == "farewell":
            print("Chat session ended.")
            break


test_novel_phrasings(
    loaded_model,
    loaded_vectorizer
)

chat(
    loaded_model,
    loaded_vectorizer
)