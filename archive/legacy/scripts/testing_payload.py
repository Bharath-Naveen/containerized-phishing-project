import os
import requests

API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/responses"

payload = {
    "model": "gpt-4.1",
    "input": "Say hello in one sentence."
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

print("API key loaded:", API_KEY is not None)

r = requests.post(API_URL, headers=headers, json=payload, timeout=30)
print(r.status_code)
print(r.text)