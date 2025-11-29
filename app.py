from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(force=True)
    question = data.get("question", "")

    try:
        # Call OpenAI Chat API
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # lightweight, fast model
            messages=[
                {"role": "system", "content": "You are VP StudyBot, a helpful AI assistant."},
                {"role": "user", "content": question}
            ],
            max_tokens=200
        )

        # Safe extraction: avoid NoneType errors
        answer_text = (response.choices[0].message.content or "Sorry, I didn’t catch that.").strip()

        return jsonify({
            "answer": answer_text,
            "suggestions": [
                "Ask me another topic",
                "Try a fun fact",
                "Need help with studies?"
            ]
        })

    except Exception as e:
        return jsonify({
            "answer": "Oops, something went wrong while fetching the answer.",
            "suggestions": ["Ask another topic", "Help", "Product demo"],
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)