import requests
import json

# Your deployed Vercel URL
base_url = "https://ipl-2026-predictor.vercel.app"
predict_url = f"{base_url}/predict"
metadata_url = f"{base_url}/metadata"

data = {
    "team1": "Chennai Super Kings",
    "team2": "Mumbai Indians",
    "venue": "MA Chidambaram Stadium, Chepauk, Chennai",
    "toss_winner": "Mumbai Indians",
    "toss_decision": "bat"
}

try:
    # 1. Test the home endpoint
    print(f"Testing Home: {base_url}/")
    home = requests.get(base_url)
    print(f"Response: {home.text}\n")

    # 2. Test the predict endpoint
    print(f"Testing Prediction: {predict_url}")
    response = requests.post(predict_url, json=data)
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

    # 3. Test metadata
    print(f"Testing Metadata: {metadata_url}")
    metadata = requests.get(metadata_url)
    print(f"Response: {json.dumps(metadata.json(), indent=2)}")

except Exception as e:
    print(f"Error: {e}")

