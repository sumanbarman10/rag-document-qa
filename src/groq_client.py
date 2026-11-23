
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"


def get_groq_client():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Check your .env file."
        )

    return Groq(api_key=api_key)


def test_groq_connection():

    client = get_groq_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with exactly: "
                    "Groq GPT-OSS connection successful."
                )
            }
        ],
        temperature=0,
        max_completion_tokens=100
    )

    answer = response.choices[0].message.content

    print("========== GROQ TEST ==========")
    print("Model:", MODEL_NAME)
    print("Response:", answer)


if __name__ == "__main__":
    test_groq_connection()
