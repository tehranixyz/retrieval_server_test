import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

parser = argparse.ArgumentParser(description="Launch the local judge server.")
parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Path to LLM and Tokenizer")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address for the server")
parser.add_argument("--port", type=int, default=8000, help="Port for the server")

args = parser.parse_args()

# Model Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

# Verify request format
class QueryRequest(BaseModel):
    queries: List[str]

# Model response generation
def generate_llm_response(query: str, max_new_tokens: int = 100):
    inputs = tokenizer(query, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode full response
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the original prompt from the response
    generated_text = full_response[len(query):].strip()

    return generated_text

# FastAPI setup
app = FastAPI()
@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    responses = [generate_llm_response(query) for query in request.queries]
    return {"responses": responses}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
