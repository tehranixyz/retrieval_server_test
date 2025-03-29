import argparse
import requests


parser = argparse.ArgumentParser(description="Launch the local judge server.")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address for the server")
parser.add_argument("--port", type=int, default=8000, help="Port for the server")

args = parser.parse_args()

# URL for your local FastAPI server
url = f"http://{args.host}:{args.port}/retrieve"

# Example payload
payload = {
    "queries": ["What is the capital of France?", "Explain neural networks."]
}

print(f"Sending POST request to {url}")
# Send POST request
response = requests.post(url, json=payload)

# Raise an exception if the request failed
response.raise_for_status()

# Get the JSON response
retrieved_data = response.json()

print("Response from server:")
for query, res in zip(payload['queries'], retrieved_data['responses']):
    print("===" * 20)
    print(f"Query:\n{query}")
    print(f"Response:\n{res}")