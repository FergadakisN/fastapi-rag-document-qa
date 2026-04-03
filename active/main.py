import os
import re
import json
import numpy as np
import tiktoken
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")
CAMPUSAI_API_URL = os.getenv("CAMPUSAI_API_URL")
CAMPUSAI_CHAT_MODEL = os.getenv("CAMPUSAI_CHAT_MODEL", "Gemma 3 (Chat)")
EMBED_MODEL = "Nomic Embed Text"

MD_FILE = Path(__file__).parent.parent / "Nielsen2025Natural-2026-03-20.md"

client = OpenAI(api_key=CAMPUSAI_API_KEY, base_url=CAMPUSAI_API_URL)

# In-memory index built at startup
chunks: list[str] = []
chunk_embeddings: np.ndarray = None

# Section heading patterns (e.g. "6.4.1 Sentence embedding" or "Chapter 7")
SECTION_RE = re.compile(r'^\d+(?:\.\d+)* \S|^Chapter \d+$', re.MULTILINE)

# Nomic Embed Text has a hard 512-token context window.
# Its tokenizer counts ~1.71× more tokens than tiktoken cl100k_base.
# Use 250 so the server sees at most ~427 tokens — safely under 512.
MAX_TOKENS = 250
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _token_split(text: str, overlap_tokens: int = 50) -> list[str]:
    """Split a single text into ≤MAX_TOKENS pieces on token boundaries."""
    token_ids = _tokenizer.encode(text)
    if len(token_ids) <= MAX_TOKENS:
        return [text]

    pieces: list[str] = []
    step = MAX_TOKENS - overlap_tokens
    for start in range(0, len(token_ids), step):
        piece_ids = token_ids[start : start + MAX_TOKENS]
        pieces.append(_tokenizer.decode(piece_ids))
    return pieces


def load_and_chunk(max_chars: int = 800, overlap_paras: int = 1) -> list[str]:
    text = MD_FILE.read_text(encoding="utf-8")

    # Clean up PDF conversion artifacts before any searching/splitting
    text = re.sub(r'\(cid:\d+\)', '• ', text)  # bullet/special chars → •
    text = text.replace('\x0c', '\n')            # form-feed page breaks → newline

    # Always include the title/author page as the very first chunk
    preamble_chunk = text[:2000].strip()

    # Skip the table of contents — real content starts at "Chapter 1"
    match = re.search(r'^Chapter 1$', text, re.MULTILINE)
    text = text[match.start():] if match else text

    # Collapse runs of blank lines left after cleanup
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    raw_chunks: list[str] = []
    current: list[str] = []
    current_len: int = 0

    for para in paragraphs:
        is_heading = bool(SECTION_RE.match(para))
        para_len = len(para)

        # Start a new chunk at section headings or when max_chars would be exceeded
        if current and (is_heading or current_len + para_len > max_chars):
            raw_chunks.append("\n\n".join(current))
            # Keep last paragraph(s) as overlap, but not across headings
            current = current[-overlap_paras:] if not is_heading else []
            current_len = sum(len(p) for p in current)

        current.append(para)
        current_len += para_len

    if current:
        raw_chunks.append("\n\n".join(current))

    # Hard cap: split any chunk that still exceeds MAX_TOKENS
    result: list[str] = []
    for chunk in raw_chunks:
        result.extend(_token_split(chunk))

    # Prepend preamble (title/author page) as the very first chunk
    return _token_split(preamble_chunk) + result


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts in batches (CampusAI has a batch-size limit)."""
    all_embeddings: list[list[float]] = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
    return np.array(all_embeddings, dtype=np.float32)


def cosine_search(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    scores = (matrix / norms) @ q
    return np.argsort(scores)[::-1][:k].tolist()


def retrieve(question: str, k: int = 5) -> list[str]:
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    query_vec = np.array(response.data[0].embedding, dtype=np.float32)
    indices = cosine_search(query_vec, chunk_embeddings, k)
    return [chunks[i] for i in indices]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chunks, chunk_embeddings
    print(f"Loading document: {MD_FILE.resolve()}")
    text_preview = MD_FILE.read_text(encoding="utf-8")[:200].replace("\n", " ")
    print(f"  Preview: {text_preview}")
    print("Chunking document...")
    chunks = load_and_chunk()
    print(f"  {len(chunks)} chunks created. Embedding (this may take a minute)...")
    chunk_embeddings = embed_texts(chunks)
    print("  Index ready.")
    yield


app = FastAPI(lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


SYSTEM_PROMPT = """\
You are a question-answering assistant for the NLP textbook "Natural Language Processing" by Finn Årup Nielsen.
You MUST answer using ONLY the context excerpts provided in the user message.
Do NOT use any knowledge from your training data. Do NOT invent authors, titles, or facts.
If the answer is not present in the provided context, respond with "The provided context does not contain enough information to answer this question."

Respond with a JSON object containing exactly two fields:
- "answer": a direct answer drawn exclusively from the provided context
- "followup_questions": a list of 2-3 related questions the reader might ask next

Do not wrap the JSON in markdown code fences."""


@app.post("/api/v1/ask")
def ask(req: AskRequest):
    retrieved = retrieve(req.question)
    context = "\n\n---\n\n".join(retrieved)

    user_message = f"Context:\n{context}\n\nQuestion: {req.question}"

    response = client.chat.completions.create(
        model=CAMPUSAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON; strip markdown fences if the model added them anyway
    json_text = raw
    fence_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', raw)
    if fence_match:
        json_text = fence_match.group(1)

    try:
        data = json.loads(json_text)
        return {
            "answer": data.get("answer", raw),
            "followup_questions": data.get("followup_questions", []),
        }
    except json.JSONDecodeError:
        return {"answer": raw, "followup_questions": []}
