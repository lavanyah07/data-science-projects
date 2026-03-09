import ollama
import re

# ------------------ System Prompt ------------------
SYSTEM_PROMPT = (
    "You are a sarcastic agent. "
    "Classify the sentiment strictly as Positive, Negative, or Neutral. "
    "Return only one word."
)

# ------------------ Sentiment Classification Function ------------------
def classify_sentiment(text, model="qwen3-vl:4b"):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )
    return response["message"]["content"].strip()

# ------------------ Main Program ------------------
print("Sentiment Analysis Agent")
print("Type 'exit' to stop\n")

while True:
    text = input("Enter text: ").strip()

    # 1️⃣ Empty input validation
    if not text:
        print("❌ Input cannot be empty.")
        continue

    # 2️⃣ Exit command validation (case-insensitive)
    if text.lower() == "exit":
        print("Agent stopped.")
        break

    # 3️⃣ Numeric-only validation
    if text.isdigit():
        print("❌ Numeric input is not valid for sentiment analysis.")
        continue

    # 4️⃣ Special-characters-only validation
    if not re.search(r"[A-Za-z]", text):
        print("❌ Input must contain alphabetic characters.")
        continue

    # 5️⃣ Minimum length validation
    if len(text) < 3:
        print("❌ Input is too short to analyze sentiment.")
        continue

    # 6️⃣ Maximum length validation
    if len(text) > 500:
        print("❌ Input is too long. Please enter shorter text.")
        continue

    # ✅ Valid input → Perform sentiment analysis
    sentiment = classify_sentiment(text)
    print("Predicted Sentiment:", sentiment)
    print("-" * 40)
