from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, re, os
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# -------------------------------
# API Clients and Keys
# -------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OWM_KEY = os.getenv("OPENWEATHERMAP_KEY")
ALPHA_KEY = os.getenv("ALPHAVANTAGE_KEY")
NEWS_KEY = os.getenv("NEWSAPI_KEY")

# -------------------------------
# Static Knowledge Base
# -------------------------------
KB = {
    "photosynthesis": "Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
    "gravity": "Gravity is a force that attracts two bodies toward each other.",
    "equation": "An equation states that two expressions are equal, typically with an equals sign.",
    "noun": "A noun names a person, place, thing, or idea."
}

ALIASES = {
    "gravitation": "gravity",
    "what is gravity": "gravity",
    "define noun": "noun",
    "photosynthesis definition": "photosynthesis"
}

# -------------------------------
# Intent Detection
# -------------------------------
def detect_intent(text):
    t = text.lower()
    if any(k in t for k in ["weather", "temperature", "forecast"]):
        return "weather"
    if any(k in t for k in ["stock", "share price", "price", "ticker"]):
        return "stock"
    if any(k in t for k in ["news", "headline", "trending"]):
        return "news"
    if any(k in t for k in ["define", "meaning", "synonym", "antonym"]):
        return "dictionary"
    return "kb"

# -------------------------------
# Weather API
# -------------------------------
def extract_city(text, default="Tiruchirappalli"):
    m = re.search(r"weather(?:\s+in\s+([a-zA-Z\s]+))?", text.lower())
    return m.group(1).strip().title() if m and m.group(1) else default

def get_weather_response(question):
    if not OWM_KEY:
        return jsonify({"ok": False, "answer": "Weather not configured. Missing OPENWEATHERMAP_KEY.", "suggestions": ["Help","Ask another topic","Top news"]}), 200
    city = extract_city(question)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OWM_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return jsonify({"ok": False, "answer": f"Couldn’t fetch weather for {city}.", "suggestions": ["Ask another topic","Product demo","Help"]}), 200
        data = r.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        hum = data["main"]["humidity"]
        return jsonify({
            "ok": True,
            "answer": f"Weather in {city}: {temp}°C, {desc}, humidity {hum}%.",
            "suggestions": ["Ask another topic","Check stock price","Top news"]
        }), 200
    except Exception:
        return jsonify({"ok": False, "answer": "Weather service timeout. Try again.", "suggestions": ["Top news","Stock price","Help"]}), 200

# -------------------------------
# Stock API (Alpha Vantage)
# -------------------------------
def extract_ticker(text):
    m = re.search(r"(?:stock|price)\s+(?:of\s+)?([A-Za-z\.:-]{1,10})", text.lower())
    return (m.group(1) or "").upper() if m else "TSLA"

def get_stock_response(question):
    if not ALPHA_KEY:
        return jsonify({"ok": False, "answer": "Stocks not configured. Missing ALPHAVANTAGE_KEY.", "suggestions": ["Weather","Top news","Help"]}), 200
    ticker = extract_ticker(question)
    # Build request URL and fetch quote
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_KEY}"
    try:
        r = requests.get(url, timeout=8)
        q = r.json().get("Global Quote", {})
        price = q.get("05. price")
        change = q.get("09. change")
        percent = q.get("10. change percent")
        if not price:
            return jsonify({"ok": False, "answer": f"Couldn’t fetch {ticker}. Try another symbol.", "suggestions": ["Top news","Weather","Help"]}), 200
        return jsonify({
            "ok": True,
            "answer": f"{ticker}: ${price} ({change} / {percent}).",
            "suggestions": ["Ask another topic","Top news","Weather"]
        }), 200
    except Exception:
        return jsonify({"ok": False, "answer": "Stock service timeout. Try again.", "suggestions": ["Weather","Top news","Help"]}), 200

# -------------------------------
# News API (NewsAPI)
# -------------------------------
def get_news_response(question):
    if not NEWS_KEY:
        return jsonify({"ok": False, "answer": "News not configured. Missing NEWSAPI_KEY.", "suggestions": ["Weather","Stock price","Help"]}), 200
    # Use US headlines for reliability
    url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=3&apiKey={NEWS_KEY}"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return jsonify({"ok": False, "answer": "Couldn’t fetch headlines right now.", "suggestions": ["Weather","Stock price","Help"]}), 200
        arts = r.json().get("articles", [])[:3]
        headlines = [f"- {a['title']}" for a in arts if a.get("title")]
        msg = "Top headlines:\n" + "\n".join(headlines) if headlines else "No headlines found."
        return jsonify({"ok": True, "answer": msg, "suggestions": ["Ask another topic","Weather","Stock price"]}), 200
    except Exception:
        return jsonify({"ok": False, "answer": "News service timeout. Try again.", "suggestions": ["Weather","Stock price","Help"]}), 200

# -------------------------------
# Dictionary API
# -------------------------------
def extract_word(text):
    m = re.search(r"(?:define|meaning(?:\s+of)?)\s+([a-zA-Z\-]+)", text.lower())
    return m.group(1) if m else None

def get_dictionary_response(question):
    word = extract_word(question) or "innovation"
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return jsonify({"ok": False, "answer": f"Couldn’t find a definition for '{word}'.", "suggestions": ["Ask another topic","Top news","Weather"]}), 200
        data = r.json()
        meanings = data[0].get("meanings", [])
        defs = meanings[0].get("definitions", []) if meanings else []
        first = defs[0].get("definition") if defs else "No definition available."
        return jsonify({"ok": True, "answer": f"{word.capitalize()}: {first}", "suggestions": ["Ask another topic","Synonyms","Top news"]}), 200
    except Exception:
        return jsonify({"ok": False, "answer": "Dictionary service timeout. Try again.", "suggestions": ["Top news","Weather","Help"]}), 200

# -------------------------------
# OpenAI Fallback
# -------------------------------
def get_openai_response(question):
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({"ok": False, "answer": "AI not configured. Missing OPENAI_API_KEY.", "suggestions": ["Help","Ask another topic","Top news"]}), 200
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
            max_tokens=180,
            temperature=0.7,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return jsonify({
            "ok": True,
            "answer": answer,
            "suggestions": ["Ask another topic","Product demo","Help"]
        }), 200
    except Exception:
        return jsonify({
            "ok": False,
            "answer": "AI assistant couldn’t answer right now.",
            "suggestions": ["Help","Product demo","Ask another topic"]
        }), 200

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return "✅ StudyBot backend is running. Use POST /answer to ask questions."

@app.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or request.form
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({
            "ok": False,
            "answer": "Please ask a topic. For example: photosynthesis, gravity, equation, noun.",
            "suggestions": ["photosynthesis", "gravity", "equation", "noun"]
        }), 200

    clean = question.lower().strip()
    if clean in ALIASES:
        clean = ALIASES[clean]

    intent = detect_intent(question)

    if intent == "kb":
        if clean in KB:
            return jsonify({"ok": True, "answer": KB[clean], "suggestions": ["Ask another topic","Product demo","Help"]}), 200
        for key in KB.keys():
            if key in clean:
                return jsonify({"ok": True, "answer": KB[key], "suggestions": ["Ask another topic","Product demo","Help"]}), 200
        return get_openai_response(question)

    elif intent == "weather":
        return get_weather_response(question)
    elif intent == "stock":
        return get_stock_response(question)
    elif intent == "news":
        return get_news_response(question)
    elif intent == "dictionary":
        return get_dictionary_response(question)

    return get_openai_response(question)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)