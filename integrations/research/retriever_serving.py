# Adapted from Agent-RL/ReCall (re-search branch); see LICENSE.
from fastapi import FastAPI, HTTPException
import argparse
from pydantic import BaseModel
from typing import List, Tuple, Union
import asyncio

from flashrag.config import Config
from flashrag.utils import get_retriever

app = FastAPI()

retriever = None
retriever_semaphore = None
max_concurrency = 0
active_requests = 0

def init_retriever(args):
    global retriever, retriever_semaphore, max_concurrency
    if args.max_concurrency < 1:
        raise ValueError("--max_concurrency must be at least 1")

    config = Config(args.config)
    print("Initializing shared retriever (one model/index copy)")
    retriever = get_retriever(config)

    # The model, corpus and FAISS index are read-only while searching, so all
    # request threads can share them.  This number controls in-flight searches;
    # it must not control how many 60+ GB index copies are loaded.
    max_concurrency = args.max_concurrency
    retriever_semaphore = asyncio.Semaphore(max_concurrency)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retriever_instances": 1 if retriever is not None else 0,
        "concurrency": {
            "limit": max_concurrency,
            "active": active_requests,
            "available": max_concurrency - active_requests,
        },
    }

class QueryRequest(BaseModel):
    query: str
    top_n: int = 10
    return_score: bool = False

class BatchQueryRequest(BaseModel):
    query: List[str]
    top_n: int = 10
    return_score: bool = False

class Document(BaseModel):
    id: str
    contents: str

@app.post("/search", response_model=Union[Tuple[List[Document], List[float]], List[Document]])
async def search(request: QueryRequest):
    global active_requests
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    if not query or not query.strip():
        print(f"Query content cannot be empty: {query}")
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )

    async with retriever_semaphore:
        active_requests += 1
        try:
            if return_score:
                results, scores = await asyncio.to_thread(
                    retriever.search, query, top_n, return_score
                )
                return [Document(id=result['id'], contents=result['contents']) for result in results], scores
            else:
                results = await asyncio.to_thread(
                    retriever.search, query, top_n, return_score
                )
                return [Document(id=result['id'], contents=result['contents']) for result in results]
        finally:
            active_requests -= 1

@app.post("/batch_search", response_model=Union[List[List[Document]], Tuple[List[List[Document]], List[List[float]]]])
async def batch_search(request: BatchQueryRequest):
    global active_requests
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    async with retriever_semaphore:
        active_requests += 1
        try:
            if return_score:
                results, scores = await asyncio.to_thread(
                    retriever.batch_search, query, top_n, return_score
                )
                return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))], scores
            else:
                results = await asyncio.to_thread(
                    retriever.batch_search, query, top_n, return_score
                )
                return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))]
        finally:
            active_requests -= 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./retriever_config.yaml")
    parser.add_argument(
        "--max_concurrency",
        "--num_retriever",
        dest="max_concurrency",
        type=int,
        default=1,
        help="Maximum concurrent searches sharing one retriever instance",
    )
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()

    init_retriever(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
