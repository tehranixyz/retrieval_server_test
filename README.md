# Retrieval Server Test

## Install required packages
```
pip insall torch transformers fastapi pydantic uvicorn
```



## Launch server
You need two terminals. One for launching the server, another one for making a POST request to the server.
On first terminal:
```
python llm_retriever_server.py --model_path Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8000
```

## Make a request
On second terminal:
```
python request.py --host 127.0.0.1 --port 8000
```