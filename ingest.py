# ingest.py
import os
import json
import uuid
import pathlib
from typing import List, Dict

import chromadb
from google import genai
from google.genai import types

# 1) Set your key. For first run, hardcode it. Later move to Run Config → Environment variables.
API_KEY = os.environ["GEMINI_API_KEY"]

# 2) Clients
client = genai.Client(api_key=API_KEY)
chroma = chromadb.PersistentClient()
notes = chroma.get_or_create_collection(name="notes", embedding_function=None)

# 3) Models
CONDENSE_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "text-embedding-004"

# 4) JSON schema for structured outputs
SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "definitions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"term": {"type": "string"}, "meaning": {"type": "string"}},
                "required": ["term", "meaning"],
            },
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "evidence": {"type": "string"}},
                "required": ["claim"],
            },
        },
        "qa": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"q": {"type": "string"}, "a": {"type": "string"}},
                "required": ["q", "a"],
            },
        },
        "source": {
            "type": "object",
            "properties": {"doc": {"type": "string"}, "section": {"type": "string"}},
            "required": ["doc", "section"],
        },
        "schema_version": {"type": "string"},
    },
    "required": ["title", "summary", "key_points", "source", "schema_version"],
}

def chunk(text: str, size: int = 3000, overlap: int = 300) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                out.append(buf)
            buf = (buf[-overlap:] + "\n\n" + p) if (overlap and len(buf) > overlap) else p
    if buf:
        out.append(buf)
    return out

def condense(raw_text: str, doc: str, section: str) -> Dict:
    prompt = f"""Read the text and output JSON only.
Follow the schema strictly. Keep summary 5-8 short sentences. Use plain language.

Text:
{raw_text}

Doc: {doc}
Section: {section}
"""
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=SCHEMA,
        temperature=0.2,
    )
    resp = client.models.generate_content(
        model=CONDENSE_MODEL,
        contents=prompt,
        config=cfg,
    )
    # If structured parsing is available, prefer it. Otherwise parse text.
    return resp.parsed if getattr(resp, "parsed", None) else json.loads(resp.text)

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [e.values for e in resp.embeddings]

def ingest(raw_text: str, doc_name: str = "notes.txt") -> None:
    for i, ch in enumerate(chunk(raw_text), start=1):
        record = condense(ch, doc=doc_name, section=f"chunk_{i}")
        record["schema_version"] = "v1"
        embed_text = " ".join([
            record.get("title", ""),
            record.get("summary", ""),
            " ".join(record.get("key_points", [])),
        ])
        vec = embed_texts([embed_text])[0]
        rid = f"{doc_name}:{i}:{uuid.uuid4().hex[:8]}"
        notes.add(
            ids=[rid],
            embeddings=[vec],
            metadatas=[{"record": json.dumps(record, ensure_ascii=False)}],
            documents=[embed_text],
        )

    print("Total chunks in collection:", notes.count())
    print("Sample record:", notes.peek()["metadatas"][:1])

    print("Ingested OK.")

if __name__ == "__main__":
    import sys
    # Run with parameter: my_text.txt
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_text_file>")
        raise SystemExit(1)
    path = pathlib.Path(sys.argv[1])
    raw = path.read_text(encoding="utf-8")
    ingest(raw, doc_name=path.name)
