# ask.py
import os
import json
import chromadb
from google import genai
from google.genai import types

# 1) Key
API_KEY = os.environ["GEMINI_API_KEY"]

# 2) Clients
client = genai.Client(api_key=API_KEY)
chroma = chromadb.PersistentClient()
notes = chroma.get_or_create_collection(name="notes", embedding_function=None)

# 3) Models
ANSWER_MODEL = "gemini-2.5-pro"
EMBED_MODEL = "text-embedding-004"

def embed_query(q: str):
    r = client.models.embed_content(model=EMBED_MODEL, contents=q)
    return r.embeddings[0].values

def search(question: str, k: int = 5):
    qvec = embed_query(question)
    res = notes.query(query_embeddings=[qvec], n_results=k, include=["metadatas"])
    records = [json.loads(m["record"]) for m in res["metadatas"][0]]
    return records

def build_context(records):
    blocks = []
    for r in records:
        kp = "; ".join(r.get("key_points", []))
        src = f"[{r['source']['doc']}, {r['source']['section']}]"
        blocks.append(
            f"Title: {r['title']}\nSummary: {r['summary']}\nKey points: {kp}\nSource: {src}"
        )
    return "\n\n".join(blocks)

def answer(question: str) -> str:
    ctx = build_context(search(question))
    system = (
        "You answer ONLY from the provided notes. "
        "Add inline citations like [doc, section]. "
        "If info is missing, say you don't have enough information."
    )
    prompt = f"{system}\n\nQuestion:\n{question}\n\nNotes:\n{ctx}"
    resp = client.models.generate_content(
        model=ANSWER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return resp.text.strip()

if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What are the key ideas?"
    print(answer(question))
