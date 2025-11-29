import requests

queries = [
    "photosynthesis",
    "weather in Tiruchirappalli",
    "stock price of TSLA",
    "top news",
    "define resilience",
    "Explain Newton's laws in simple words"
]

for q in queries:
    resp = requests.post("http://127.0.0.1:5000/answer", json={"question": q})
    print("\nQ:", q)
    print("Response:", resp.json())