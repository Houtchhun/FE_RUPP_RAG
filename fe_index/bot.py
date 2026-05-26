import os
import re
import time
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("Missing TELEGRAM_TOKEN in .env")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

EMBEDDING_MODEL    = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL          = "llama-3.1-8b-instant"

MEMORY_K        = 5     # conversation turns to remember per user
TOP_K           = 5     # chunks returned after fusion / after reranking
RETRIEVAL_K     = 20    # candidates retrieved before reranking
MAX_DISTANCE    = 1.2   # cosine distance threshold
RRF_K           = 60    # RRF constant

# per-user conversation memory  { chat_id: ["User: ...", "Assistant: ...", ...] }
user_memory: Dict[int, List[str]] = defaultdict(list)

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    temperature=0.0,
    groq_api_key=GROQ_API_KEY,
    model_name=LLM_MODEL,
)

# ── Embeddings & ChromaDB ─────────────────────────────────────────────────────
logger.info("Loading ChromaDB vector store…")
embeddings  = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# ── Cross-encoder reranker ────────────────────────────────────────────────────
logger.info("Loading cross-encoder reranker…")
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", text).lower().split()


# ── BM25 Index ────────────────────────────────────────────────────────────────
logger.info("Initialising BM25 index…")
try:
    _all       = vectorstore.get()
    documents  = _all.get("documents", [])
    metadatas  = _all.get("metadatas", [])
    _tokenized = [_tokenize(d) for d in documents]
    bm25       = BM25Okapi(_tokenized) if _tokenized else None
    logger.info("BM25 ready — %d documents indexed.", len(documents))
except Exception as exc:
    logger.warning("BM25 init failed: %s", exc)
    documents, metadatas, bm25 = [], [], None

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a helpful assistant for the Faculty of Engineering (FE) at the \
Royal University of Phnom Penh (RUPP), Cambodia.

Rules you MUST follow:
1. Answer ONLY from the Retrieved Context provided below. Do NOT use outside knowledge.
2. If the answer is in the context, provide it completely and accurately.
3. Never say "not explicitly listed" or "not mentioned" when the information IS present.
4. If the context truly does not contain the answer, say exactly: "I don't have that information."
5. Do NOT fabricate, infer, or add any detail not present in the context.
6. Be concise, friendly, and respond in English only.
7. When describing a program, always include degree type, duration, and language of instruction \
   if present in the context.
8. For general knowledge questions unrelated to FE-RUPP (e.g. "what is machine learning", \
   "capital of France"), politely decline and redirect to FE-RUPP topics.

Conversation History:
{history}

Retrieved Context:
{context}

User Question: {question}

Answer:"""

WELCOME_MSG = (
    "Hello! I am the FE-RUPP assistant.\n"
    "Ask me anything about programs, admission, curriculum, faculty, and more.\n\n"
    "Commands:\n"
    "  /help  – Show help\n"
    "  /clear – Reset your conversation history"
)

HELP_MSG = (
    "FE-RUPP Chatbot — Help\n\n"
    "I can answer questions about the Faculty of Engineering at RUPP:\n"
    "• Available programs & curriculum\n"
    "• Admission requirements & process\n"
    "• Faculty research and vision\n\n"
    "Sample questions:\n"
    "  – What programs does FE offer?\n"
    "  – What does DSE focus on?\n"
    "  – Where is the FE located?\n\n"
    "Use /clear to reset your conversation history."
)

# ── Query normalisation maps ──────────────────────────────────────────────────
_YEAR_ALIASES = {
    "year 1":   "Foundation Year",
    "1st year": "Foundation Year",
    "year one": "Foundation Year",
}

_YEAR_TO_DB_TERM = {
    "foundation year": "Foundation Year Courses Semester",
    "year 2":          "Year 2 Courses Semester",
    "year 3":          "Year 3 Courses Semester",
    "year 4":          "Year 4 Courses Semester",
}

_ABBREV_TO_FULL: Dict[str, str] = {
    "FTE":  "Food Technology And Engineering",
    "DSE":  "Data Science And Engineering",
    "SCA":  "Automation And Supply Chain System Engineering",
    "EE":   "Environmental Engineering",
    "BIO":  "Bio Engineering Biotechnology",
    "ITE":  "Information Technology Engineering",
    "TEE":  "Telecommunication And Electronics Engineering",
    "MBFT": "Biotechnology and Food Technology Master Program",
    "MITE": "Master of Science in Information Technology Engineering",
}

_NAME_TO_ABBREV: Dict[str, str] = {
    "food technology and engineering":             "FTE",
    "food technology":                             "FTE",
    "data science and engineering":                "DSE",
    "data science":                                "DSE",
    "automation and supply chain":                 "SCA",
    "supply chain":                                "SCA",
    "environmental engineering":                   "EE",
    "bio engineering":                             "BIO",
    "biotechnology":                               "BIO",
    "information technology engineering":          "ITE",
    "information technology":                      "ITE",
    "telecommunication":                           "TEE",
    "electronics engineering":                     "TEE",
    "biotechnology and food technology master":    "MBFT",
    "master of science in information technology": "MITE",
}

_FOLLOWUP_SIGNALS = (
    "how about", "what about", "and in", "in year", "in semester",
    "how many subjects", "year 2", "year 3", "year 4",
    "they ", "their ", "it ", "this ", "that ", "what else",
    "can they", "do they", "what do they", "what can they",
)

_CURRICULUM_KEYWORDS = re.compile(
    r"\b(curriculum|course|courses|subject|subjects|class|classes|study|semester)\b",
    re.IGNORECASE,
)

# Master program identifiers — used to distinguish "Year 1" (master) vs "Foundation Year" (bachelor)
_MASTER_SIGNALS = (
    "mite", "mbft", "master", "biotechnology and food technology master",
    "master of science in information technology",
)


# ── Year filter detection ─────────────────────────────────────────────────────

def _detect_year_filter(query: str) -> Optional[str]:
    """Return ChromaDB 'year' metadata value if query specifies a year, else None."""
    q_lower = query.lower()

    if "foundation year" in q_lower:
        return "foundation"

    if "1st year" in q_lower or "year one" in q_lower:
        return "foundation"

    if "year 1" in q_lower:
        is_master = any(sig in q_lower for sig in _MASTER_SIGNALS)
        return "year_1" if is_master else "foundation"

    if "year 2" in q_lower:
        return "year_2"
    if "year 3" in q_lower:
        return "year_3"
    if "year 4" in q_lower:
        return "year_4"

    return None


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _rrf_fuse(rankings: List[List[int]]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


def _rerank(query: str, candidates: List[Tuple[str, str]], top_k: int) -> Tuple[List[str], List[str]]:
    """Cross-encoder rerank (query, doc) pairs and return top_k (chunks, sources)."""
    if len(candidates) <= 1:
        chunks  = [c[0] for c in candidates[:top_k]]
        sources = [c[1] for c in candidates[:top_k] if c[1]]
        return chunks, sources
    pairs   = [(query, c[0]) for c in candidates]
    scores  = cross_encoder.predict(pairs)
    ranked  = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    chunks  = [c[0] for _, c in ranked[:top_k]]
    sources = [c[1] for _, c in ranked[:top_k] if c[1]]
    return chunks, sources


def hybrid_search(
    query: str,
    top_k: int = TOP_K,
    year_filter: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Return (context_text, source_urls).

    Pipeline: BM25 + Vector → RRF → cross-encoder rerank (top_k).
    When year_filter is set, vector search is filtered to that year's curriculum
    chunks before reranking — eliminates wrong-year retrieval.
    """
    if year_filter:
        chroma_filter = {"year": {"$eq": year_filter}}
        vec_hits = vectorstore.similarity_search_with_score(
            query, k=top_k * 4, filter=chroma_filter
        )
        candidates = [
            (doc.page_content, doc.metadata.get("source", ""))
            for doc, score in vec_hits
            if score <= MAX_DISTANCE
        ]
        if candidates:
            chunks, sources = _rerank(query, candidates, top_k)
            return "\n\n---\n\n".join(chunks), list(dict.fromkeys(sources))
        logger.warning("Year filter '%s' returned no results — falling back to hybrid.", year_filter)

    # Full hybrid search (no year filter, or filter produced no results)
    if not bm25 or not documents:
        return "No documents are indexed yet.", []

    vec_hits = vectorstore.similarity_search_with_score(query, k=RETRIEVAL_K)
    vec_indices: List[int] = []
    for doc, score in vec_hits:
        if score <= MAX_DISTANCE:
            try:
                vec_indices.append(documents.index(doc.page_content))
            except ValueError:
                pass

    bm25_scores  = bm25.get_scores(_tokenize(query))
    bm25_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:RETRIEVAL_K]

    fused      = _rrf_fuse([vec_indices, bm25_indices])
    top_items  = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:RETRIEVAL_K]
    candidates = [
        (
            documents[idx],
            metadatas[idx].get("source", "") if metadatas and idx < len(metadatas) else "",
        )
        for idx, _ in top_items
    ]

    chunks, sources = _rerank(query, candidates, top_k)
    return "\n\n---\n\n".join(chunks), list(dict.fromkeys(sources))


# ── Program name resolution ───────────────────────────────────────────────────

def _resolve_program_full_name(query: str) -> str:
    q_lower = query.lower()
    for abbrev, full_name in _ABBREV_TO_FULL.items():
        if re.search(r"\b" + re.escape(abbrev) + r"\b", query):
            return full_name
    for typed_name, abbrev in _NAME_TO_ABBREV.items():
        if typed_name in q_lower:
            return _ABBREV_TO_FULL.get(abbrev, "")
    return ""


def _build_search_query(user_query: str, chat_id: int) -> str:
    """Normalise year/program aliases then prepend topic context for follow-up questions."""
    query = user_query
    for alias, replacement in _YEAR_ALIASES.items():
        query = re.sub(alias, replacement, query, flags=re.IGNORECASE)

    full_name = _resolve_program_full_name(query)

    if full_name and _CURRICULUM_KEYWORDS.search(query):
        for year_key, db_term in _YEAR_TO_DB_TERM.items():
            if year_key in query.lower():
                return f"{full_name} {db_term}"

    if full_name:
        query = f"{query} {full_name}"
        return query

    q_lower = query.lower()
    if any(sig in q_lower for sig in _FOLLOWUP_SIGNALS):
        buf    = user_memory[chat_id]
        recent = " ".join(buf[:2]) if len(buf) >= 2 else " ".join(buf)
        if recent:
            return f"{recent} {query}"
    return query


# ── Memory helpers ────────────────────────────────────────────────────────────

def _update_memory(chat_id: int, role: str, text: str) -> None:
    buf = user_memory[chat_id]
    buf.append(f"{role}: {text}")
    if len(buf) > MEMORY_K * 2:
        del buf[0]


def _get_history(chat_id: int) -> str:
    buf = user_memory[chat_id]
    return "\n".join(buf[:-1]) if len(buf) > 1 else "None"


# ── Telegram Handlers ─────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME_MSG)


async def help_cmd(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_MSG)


async def clear_cmd(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    user_memory[update.effective_chat.id].clear()
    await update.message.reply_text("Your conversation history has been cleared.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id    = update.effective_chat.id
    user_query = update.message.text.strip()

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    _update_memory(chat_id, "User", user_query)

    search_query = _build_search_query(user_query, chat_id)

    # Apply year filter only for curriculum queries that mention a specific year
    year_filter = None
    if _CURRICULUM_KEYWORDS.search(user_query):
        year_filter = _detect_year_filter(user_query)

    retrieved_context, sources = hybrid_search(search_query, year_filter=year_filter)
    history = _get_history(chat_id)

    prompt = SYSTEM_PROMPT.format(
        history=history,
        context=retrieved_context,
        question=user_query,
    )

    answer = None
    for attempt in range(3):
        try:
            response = llm.invoke(prompt)
            answer   = response.content.strip()
            break
        except Exception as exc:
            logger.warning("LLM attempt %d/3 failed: %s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    if answer is None:
        answer = "I'm currently experiencing high load. Please try again in a moment."

    _update_memory(chat_id, "Assistant", answer)

    valid_sources = [s for s in sources if s.startswith("http")][:2]
    if valid_sources:
        source_block = "\n\nSources:\n" + "\n".join(f"• {s}" for s in valid_sources)
        await update.message.reply_text(answer + source_block)
    else:
        await update.message.reply_text(answer)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help",  help_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot is running…")
    app.run_polling()


if __name__ == "__main__":
    main()
