import os
import json
import requests


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from huggingface_hub import snapshot_download

import sys
sys.path.append(os.path.dirname(__file__))
from rag import get_or_create_collection, retrieve

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
API_URL    = "https://router.huggingface.co/v1/chat/completions"
MODEL      = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct:featherless-ai")
INDEX_REPO = "jiviteshhp/earningsscribe-index"

app = FastAPI(title="EarningsScribe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
def build_rag_index():
    import glob
    collection = get_or_create_collection()
    if collection.count() > 0:
        print(f"Index already has {collection.count()} docs, skipping download.")
        return

    print("Downloading pre-built chroma index from HF dataset...")
    path = snapshot_download(
        repo_id=INDEX_REPO,
        repo_type="dataset",
        local_dir="/app/data",
        token=HF_TOKEN
    )
    print(f"Downloaded to: {path}")
    print(f"Files in /app/data: {glob.glob('/app/data/**/*', recursive=True)}")

    collection = get_or_create_collection()
    print(f"Index ready: {collection.count()} documents")

    print("Downloading pre-built chroma index from HF dataset...")
    snapshot_download(
        repo_id=INDEX_REPO,
        repo_type="dataset",
        local_dir="/app/data",
        token=HF_TOKEN
    )
    collection = get_or_create_collection()
    print(f"Index ready: {collection.count()} documents")


class TranscriptRequest(BaseModel):
    transcript: str
    ticker:     str = ""


def query_model(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":       MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  512,
        "temperature": 0.1,
        "stream":      False
    }
    print(f"Calling: {API_URL} with model {MODEL}")
    print(f"Token present: {bool(HF_TOKEN)}")
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:300]}")

    if response.status_code == 503:
        raise HTTPException(status_code=503, detail="Model is loading, retry in 20 seconds")
    if not response.ok:
        raise HTTPException(status_code=500, detail=f"HF API error {response.status_code}: {response.text}")

    result = response.json()
    return result["choices"][0]["message"]["content"]


def normalize_report(report: dict) -> dict:
    if "key_metrics" in report and isinstance(report["key_metrics"], list):
        flat = []
        for m in report["key_metrics"]:
            if isinstance(m, str):
                flat.append(m)
            elif isinstance(m, dict):
                label = m.get("metric_type") or m.get("name") or m.get("label") or ""
                value = m.get("amount") or m.get("value") or ""
                flat.append(f"{label}: {value}" if label and value else ": ".join(str(v) for v in m.values() if v))
        report["key_metrics"] = flat

    if "guidance" in report and isinstance(report["guidance"], dict):
        report["guidance"] = " | ".join(f"{k}: {v}" for k, v in report["guidance"].items())

    if "risks" in report and isinstance(report["risks"], list):
        flat = []
        for item in report["risks"]:
            if isinstance(item, str):
                flat.append(item)
            elif isinstance(item, dict):
                flat.append(item.get("description") or item.get("risk") or ": ".join(str(v) for v in item.values() if v))
        report["risks"] = flat

    if "company_summary" in report and isinstance(report["company_summary"], dict):
        report["company_summary"] = " ".join(str(v) for v in report["company_summary"].values() if v)

    return report


@app.get("/")
def root():
    return {"status": "EarningsScribe is running"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    collection = get_or_create_collection()
    return {"status": "ok", "model": MODEL, "rag_docs": collection.count()}



@app.post("/generate")
def generate(req: TranscriptRequest):
    collection   = get_or_create_collection()
    count        = collection.count()
    print(f"Chroma collection count: {count}")

    retrieved    = []
    context_text = ""
    if count > 0:
        retrieved = retrieve(collection, req.transcript[:200], top_k=3)
        print(f"Retrieved {len(retrieved)} contexts")
        for i, ctx in enumerate(retrieved):
            context_text += f"\n[Context {i+1} - {ctx['ticker']}]\n{ctx['document'][:200]}"
    else:
        print("No docs in index, skipping RAG retrieval")

    prompt = f"""You are a senior financial analyst writing a professional report.

RELEVANT CONTEXT:
{context_text}

TRANSCRIPT:
{req.transcript[:600]}

Write a JSON report. Every value must be a plain readable string or list of strings. No nested objects. Follow this exact structure:

{{
  "company_summary": "2-3 plain English sentences summarizing overall performance with specific numbers.",
  "key_metrics": ["Metric 1: value", "Metric 2: value"],
  "guidance": "One plain English sentence about management outlook for next quarter.",
  "risks": ["Risk 1", "Risk 2"],
  "sentiment": "positive"
}}

REPORT:"""

    raw_response = query_model(prompt)

    report = None
    try:
        report = json.loads(raw_response)
    except json.JSONDecodeError:
        start = raw_response.find("{")
        end   = raw_response.rfind("}") + 1
        if start != -1 and end > start:
            try:
                report = json.loads(raw_response[start:end])
            except Exception:
                pass
    if report is None:
        report = {"raw": raw_response}

    report = normalize_report(report)

    return {
        "report": report,
        "retrieved_context": [
            {
                "ticker":  r["ticker"],
                "date":    r["date"],
                "score":   r["score"],
                "preview": r["document"][:120]
            }
            for r in retrieved
        ]
    }


@app.get("/ui")
def ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
