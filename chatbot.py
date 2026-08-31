
import json
import random
import re


# Load knowledge base
with open("knowledge_base.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)


def preprocess_text(text):
    """
    Convert text into a normalized form:
    - lowercase
    - remove punctuation
    - remove extra spaces
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def keyword_match(user_input, pattern):
    """
    Calculate keyword overlap between user input and a pattern.
    Returns a score between 0 and 1.
    """
    user_words = set(user_input.split())
    pattern_words = set(pattern.split())

    if not pattern_words:
        return 0

    common_words = user_words.intersection(pattern_words)

    return len(common_words) / len(pattern_words)


def regex_match(user_input, pattern):
    """
    Check whether the pattern matches the user input using regex.
    """
    try:
        return bool(re.search(pattern, user_input, re.IGNORECASE))
    except re.error:
        return False


def get_response(user_input):
    """
    Find the best matching intent using:
    1. Keyword matching
    2. Regex pattern matching
    """

    processed_input = preprocess_text(user_input)

    best_intent = None
    best_score = 0

    for intent, data in knowledge_base.items():

        if intent == "unknown":
            continue

        # -----------------------------
        # Keyword matching
        # -----------------------------
        for pattern in data["patterns"]:

            processed_pattern = preprocess_text(pattern)

            keyword_score = keyword_match(
                processed_input,
                processed_pattern
            )

            if keyword_score > best_score:
                best_score = keyword_score
                best_intent = intent

        # -----------------------------
        # Regex matching
        # -----------------------------
        regex_patterns = data.get("regex_patterns", [])

        for regex_pattern in regex_patterns:

            if regex_match(
                processed_input,
                regex_pattern
            ):
                best_score = 1
                best_intent = intent

    # -----------------------------
    # Response selection
    # -----------------------------
    if best_intent is not None and best_score >= 0.5:
        return random.choice(
            knowledge_base[best_intent]["responses"]
        )

    # -----------------------------
    # Fallback response
    # -----------------------------
    return random.choice(
        knowledge_base["unknown"]["responses"]
    )


def chatbot():
    """
    Start the CollegeBot conversation.
    """

    print("=" * 55)
    print("        CollegeBot - College FAQ Assistant")
    print("=" * 55)

    print(
        "Ask me about admissions, fees, courses, hostel,"
    )
    print(
        "placements and other college-related information."
    )
    print(
        "Type 'bye', 'exit' or 'quit' to end the chat."
    )
    print()

    while True:

        user_input = input("You: ").strip()

        if not user_input:
            print("Bot: Please enter a question.")
            continue

        processed_input = preprocess_text(user_input)

        # Exit commands
        if processed_input in [
            "bye",
            "goodbye",
            "exit",
            "quit"
        ]:
            print(
                "Bot:",
                random.choice(
                    knowledge_base["farewell"]["responses"]
                )
            )

            print("CollegeBot session ended.")
            break

        response = get_response(user_input)

        print("Bot:", response)
        print()


if __name__ == "__main__":
    chatbot()