import os
import json
import chromadb
from chromadb.utils import embedding_functions

# use /app in Docker, fallback to local for dev
BASE = "/app" if os.path.exists("/app") else os.path.expanduser("~/EarningScribe")

client = chromadb.PersistentClient(
    path=os.path.join(BASE, "data", "chroma_db")
)

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)

def get_or_create_collection():
    collection = client.get_or_create_collection(
        name="earnings_transcripts",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def build_index(collection):
    train_path = os.path.join(BASE, "data", "processed", "train.json")
    with open(train_path) as f:
        train_data = json.load(f)

    if collection.count() > 0:
        print(f"Index already has {collection.count()} documents. Skipping rebuild.")
        return

    print(f"Building index from {len(train_data)} training samples...")
    batch_size = 100
    documents, metadatas, ids = [], [], []

    for i, item in enumerate(train_data):
        documents.append(item["transcript"])
        metadatas.append({
            "ticker": item.get("ticker", ""),
            "date":   item.get("date", ""),
            "id":     item["id"]
        })
        ids.append(item["id"])

        if len(documents) == batch_size:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            documents, metadatas, ids = [], [], []
            print(f"  Indexed {i+1}/{len(train_data)} documents...")

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"Index built. Total documents: {collection.count()}")


def retrieve(collection, query, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "document": doc,
            "ticker":   meta.get("ticker", ""),
            "date":     meta.get("date", ""),
            "score":    round(1 - dist, 3)
        })

    return retrieved