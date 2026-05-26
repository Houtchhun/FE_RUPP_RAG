#!/usr/bin/env python3
"""
FE-RUPP RAG — Dual-Judge Evaluation
  Generator : Groq  (llama-3.1-8b-instant)
  Judge 1   : Groq  (llama-3.3-70b-versatile)
  Judge 2   : Cerebras (qwen-3-235b-a22b-instruct-2507)
  Metrics   : Relevance, Faithfulness, Correctness (0-1 each)

Usage:
    python evaluation/evaluate.py              # full run
    python evaluation/evaluate.py --reset      # delete cached results and restart
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from cerebras.cloud.sdk import Cerebras
from sentence_transformers import CrossEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DB_DIR      = ROOT / "fe_index" / "chroma_db"
GT_PATH     = Path(__file__).parent / "ground_truth.json"
RESULTS_DIR = Path(__file__).parent / "results"
RAW_JSON    = RESULTS_DIR / "raw_results.json"
LOG_FILE    = RESULTS_DIR / "evaluate.log"

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_GEN_MODEL       = "llama-3.1-8b-instant"
GROQ_JUDGE_MODEL     = "qwen/qwen3-32b"
CEREBRAS_JUDGE_MODEL = "qwen-3-235b-a22b-instruct-2507"

TOP_K           = 5
RETRIEVAL_K     = 20
MAX_DISTANCE    = 1.2
RRF_K           = 60
REQUEST_DELAY   = 1.0   # seconds between generation calls (8B is fast)
JUDGE_DELAY     = 1.5   # seconds between each judge call

# ── Threshold-based scoring ────────────────────────────────────────────────────
# Each metric must individually exceed its floor for a PASS verdict.
# Faithfulness has the strictest floor: a hallucinated answer must never PASS
# even when relevance and correctness are high.
RELEVANCE_FLOOR    = 0.60   # retrieved context must be on-topic
FAITHFULNESS_FLOOR = 0.70   # answer must be grounded in retrieved context
CORRECTNESS_FLOOR  = 0.50   # partial correctness is acceptable
OVERALL_PASS       = 0.70   # combined average must also exceed this
PARTIAL_THRESHOLD  = 0.40   # overall average floor for PARTIAL verdict

# ── Query normalisation ────────────────────────────────────────────────────────
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
}
_CURRICULUM_KW = re.compile(
    r"\b(curriculum|course|courses|subject|subjects|class|classes|study|semester)\b",
    re.IGNORECASE,
)
_MASTER_SIGNALS = (
    "mite", "mbft", "master", "biotechnology and food technology master",
    "master of science in information technology",
)


def _normalize_query(question: str) -> str:
    query = question
    for alias, replacement in _YEAR_ALIASES.items():
        query = re.sub(alias, replacement, query, flags=re.IGNORECASE)

    db_full_name = ""
    for abbrev, full_name in _ABBREV_TO_FULL.items():
        if re.search(r"\b" + re.escape(abbrev) + r"\b", query):
            db_full_name = full_name
            break
    if not db_full_name:
        for name, abbrev in _NAME_TO_ABBREV.items():
            if re.search(r"\b" + re.escape(name) + r"\b", query, re.IGNORECASE):
                db_full_name = _ABBREV_TO_FULL.get(abbrev, "")
                break

    if db_full_name and _CURRICULUM_KW.search(query):
        for year_key, db_term in _YEAR_TO_DB_TERM.items():
            if year_key in query.lower():
                return f"{db_full_name} {db_term}"

    if db_full_name:
        return f"{query} {db_full_name}"

    return query


def _detect_year_filter(question: str) -> Optional[str]:
    """Return ChromaDB 'year' metadata value when question targets a specific year."""
    q = question.lower()
    if "foundation year" in q:
        return "foundation"
    if "1st year" in q or "year one" in q:
        return "foundation"
    if "year 1" in q:
        is_master = any(sig in q for sig in _MASTER_SIGNALS)
        return "year_1" if is_master else "foundation"
    if "year 2" in q:
        return "year_2"
    if "year 3" in q:
        return "year_3"
    if "year 4" in q:
        return "year_4"
    return None


# ── Prompts ────────────────────────────────────────────────────────────────────
GENERATOR_PROMPT = """\
You are a helpful assistant for the Faculty of Engineering (FE) at the Royal University of Phnom Penh (RUPP), Cambodia.

Rules:
1. Answer ONLY from the Retrieved Context below. Do NOT use outside knowledge or fabricate details.
2. When describing a program, include degree type, duration, and language if available in the context.
3. If the context does not contain the answer, say "I don't have that information."
4. If the question asks about a general concept unrelated to FE-RUPP (e.g. "what is machine learning"), \
politely decline and redirect to FE-RUPP topics.
5. Be concise, accurate, and respond in English only.
6. List all courses or items mentioned in the context — do not truncate or summarise them away.

Retrieved Context:
{context}

Question: {question}

Answer:"""

JUDGE_PROMPT = """\
You are an expert evaluator for an AI chatbot serving the Faculty of Engineering (FE-RUPP), Cambodia.

Evaluate the bot response on three criteria, each scored 0.0 to 1.0:

1. Relevance    — Does the response directly address the question asked?
   Score 1.0 = fully answers the question; 0.5 = partially; 0.0 = off-topic.

2. Faithfulness — Is every fact in the response supported by the retrieved context?
   Score 1.0 = fully grounded in context; 0.5 = mostly grounded with minor additions; 0.0 = contradicts or ignores context.
   NOTE: If the response correctly declines an out-of-scope question, score faithfulness = 1.0.

3. Correctness  — Is the response factually accurate compared to the ground truth?
   Score 1.0 = all key facts match the ground truth.
   Score 0.5 = most facts match but some are missing or slightly off.
   Score 0.0 = key facts are wrong or missing.
   Special rule: for "general" out-of-scope questions, correctness = 1.0 if the bot correctly declines.

Output ONLY a valid JSON object with no extra text:
{{"relevance": 0.0, "faithfulness": 0.0, "correctness": 0.0}}

Question: {question}

Retrieved Context (what the bot had access to):
{context}

Ground Truth Answer:
{ground_truth}

Bot Response:
{response}

JSON scores:"""

DECLINE_SIGNALS = [
    "not related", "unrelated", "don't have", "cannot answer",
    "not able to", "outside my", "fe-rupp", "redirect", "focus on",
    "not within", "i'm here to", "please ask", "i cannot", "i can't",
    "not about fe", "only answer", "engineering topics",
    "general knowledge", "not specific to", "not a topic i",
]


# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv(ROOT / "fe_index" / ".env")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

log.info("Loading embedding model (%s)...", EMBEDDING_MODEL)
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

log.info("Loading ChromaDB from %s...", DB_DIR)
vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings_model)

log.info("Building BM25 index...")
_all      = vectorstore.get()
DOCUMENTS = _all.get("documents", [])
METADATAS = _all.get("metadatas", [])


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", text).lower().split()


BM25_INDEX = BM25Okapi([_tokenize(d) for d in DOCUMENTS]) if DOCUMENTS else None
log.info("BM25 ready — %d documents indexed.", len(DOCUMENTS))

log.info("Loading cross-encoder reranker (%s)...", CROSS_ENCODER_MODEL)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

log.info("Loading Groq generator (%s)...", GROQ_GEN_MODEL)
groq_gen = ChatGroq(
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY", ""),
    model_name=GROQ_GEN_MODEL,
)

log.info("Loading Groq judge (%s)...", GROQ_JUDGE_MODEL)
groq_judge = ChatGroq(
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY", ""),
    model_name=GROQ_JUDGE_MODEL,
)

log.info("Loading Cerebras judge (%s)...", CEREBRAS_JUDGE_MODEL)
cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY", ""))


# ── Retrieval ──────────────────────────────────────────────────────────────────

def _rrf_fuse(rankings: List[List[int]]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


def _rerank(query: str, candidates: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """Cross-encoder rerank (query, doc) pairs and return top TOP_K (chunks, sources)."""
    if len(candidates) <= 1:
        chunks  = [c[0] for c in candidates[:TOP_K]]
        sources = [c[1] for c in candidates[:TOP_K] if c[1]]
        return chunks, sources
    pairs   = [(query, c[0]) for c in candidates]
    scores  = cross_encoder.predict(pairs)
    ranked  = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    chunks  = [c[0] for _, c in ranked[:TOP_K]]
    sources = [c[1] for _, c in ranked[:TOP_K] if c[1]]
    return chunks, sources


def hybrid_search(
    query: str,
    year_filter: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """BM25 + Vector → RRF → cross-encoder rerank (top TOP_K).

    When year_filter is set, vector search is filtered to that year's curriculum
    chunks before reranking — eliminates wrong-year retrieval.
    """
    if year_filter:
        chroma_filter = {"year": {"$eq": year_filter}}
        vec_hits = vectorstore.similarity_search_with_score(
            query, k=TOP_K * 4, filter=chroma_filter
        )
        candidates = [
            (doc.page_content, doc.metadata.get("source", ""))
            for doc, score in vec_hits
            if score <= MAX_DISTANCE
        ]
        if candidates:
            chunks, sources = _rerank(query, candidates)
            return "\n\n---\n\n".join(chunks), list(dict.fromkeys(sources))
        log.warning("Year filter '%s' returned no results — falling back to hybrid.", year_filter)

    # Full hybrid search
    if not BM25_INDEX or not DOCUMENTS:
        return "No documents indexed.", []

    vec_hits = vectorstore.similarity_search_with_score(query, k=RETRIEVAL_K)
    vec_indices: List[int] = []
    for doc, score in vec_hits:
        if score <= MAX_DISTANCE:
            try:
                vec_indices.append(DOCUMENTS.index(doc.page_content))
            except ValueError:
                pass

    bm25_scores  = BM25_INDEX.get_scores(_tokenize(query))
    bm25_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:RETRIEVAL_K]

    fused      = _rrf_fuse([vec_indices, bm25_indices])
    top_items  = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:RETRIEVAL_K]
    candidates = [
        (
            DOCUMENTS[idx],
            METADATAS[idx].get("source", "") if METADATAS and idx < len(METADATAS) else "",
        )
        for idx, _ in top_items
    ]

    chunks, sources = _rerank(query, candidates)
    return "\n\n---\n\n".join(chunks), list(dict.fromkeys(sources))


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _call_groq_gen(prompt: str, retries: int = 3, base_wait: int = 15) -> str:
    for attempt in range(retries):
        try:
            resp = groq_gen.invoke(prompt)
            return resp.content.strip()
        except Exception as exc:
            wait = base_wait * (attempt + 1)
            log.warning("Groq gen attempt %d/%d failed: %s. Retry in %ds...", attempt + 1, retries, exc, wait)
            time.sleep(wait)
    return "[ERROR: Groq generation failed]"


def _call_groq_judge(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = groq_judge.invoke(prompt)
            return resp.content.strip()
        except Exception as exc:
            wait = 15 * (attempt + 1)
            log.warning("Groq judge attempt %d/%d: %s. Retry in %ds...", attempt + 1, retries, exc, wait)
            time.sleep(wait)
    return "{}"


def _call_cerebras(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = cerebras_client.chat.completions.create(
                model=CEREBRAS_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            wait = 5 * (attempt + 1)
            log.warning("Cerebras attempt %d/%d: %s. Retry in %ds...", attempt + 1, retries, exc, wait)
            time.sleep(wait)
    return "{}"


def _parse_scores(raw: str) -> Dict[str, float]:
    """Extract JSON scores from judge response, with regex fallback."""
    try:
        m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            r = float(obj.get("relevance", 0))
            f = float(obj.get("faithfulness", 0))
            c = float(obj.get("correctness", 0))
            r, f, c = max(0.0, min(1.0, r)), max(0.0, min(1.0, f)), max(0.0, min(1.0, c))
            return {"relevance": r, "faithfulness": f, "correctness": c, "overall": round((r + f + c) / 3, 4)}
    except Exception:
        pass
    def _extract(key: str) -> float:
        m2 = re.search(rf'"{key}"\s*:\s*([0-9.]+)', raw)
        return max(0.0, min(1.0, float(m2.group(1)))) if m2 else 0.5
    r, f, c = _extract("relevance"), _extract("faithfulness"), _extract("correctness")
    return {"relevance": r, "faithfulness": f, "correctness": c, "overall": round((r + f + c) / 3, 4)}


def _verdict(scores: dict) -> str:
    """Threshold-based verdict: EVERY metric must exceed its individual floor,
    AND the overall average must exceed OVERALL_PASS.
    This prevents a hallucinated answer (faithfulness=0) from passing just
    because relevance or correctness happened to be high."""
    r = scores["relevance"]
    f = scores["faithfulness"]
    c = scores["correctness"]
    o = scores["overall"]
    if (r >= RELEVANCE_FLOOR and
            f >= FAITHFULNESS_FLOOR and
            c >= CORRECTNESS_FLOOR and
            o >= OVERALL_PASS):
        return "pass"
    if o >= PARTIAL_THRESHOLD:
        return "partial"
    return "fail"


# ── Main evaluation loop ───────────────────────────────────────────────────────
_CAT_REMAP = {
    "major":         "program_overview",
    "location":      "program_overview",
    "contact":       "program_overview",
    "about_us":      "program_overview",
    "research":      "career",
    "job_prospects": "career",
    "curriculum":    "curriculum",
    "general":       "general",
}


def run_evaluation() -> List[Dict]:
    with open(GT_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    ground_truth = [
        {**item, "category": _CAT_REMAP.get(item["category"], item["category"])}
        for item in raw
    ]

    results: List[Dict] = []
    done_ids: set = set()
    if RAW_JSON.exists():
        with open(RAW_JSON, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        log.info("Resuming — %d / %d questions already done.", len(done_ids), len(ground_truth))

    total = len(ground_truth)
    for i, item in enumerate(ground_truth):
        qid      = item["id"]
        category = item["category"]
        question = item["question"]
        gt       = item["ground_truth_answer"]

        if qid in done_ids:
            continue

        log.info("[%3d/%d] Q%3d [%-14s] %s...", i + 1, total, qid, category, question[:50])

        # 1. Retrieve — apply year filter only for curriculum queries
        search_q    = _normalize_query(question)
        year_filter = _detect_year_filter(question) if category == "curriculum" else None
        context, _  = hybrid_search(search_q, year_filter=year_filter)

        # 2. Generate answer (Groq 8B)
        answer = _call_groq_gen(GENERATOR_PROMPT.format(context=context, question=question))
        time.sleep(REQUEST_DELAY)

        # 3. Build judge prompt — provide more context so judge can assess faithfulness properly
        judge_input = JUDGE_PROMPT.format(
            question=question,
            context=context[:3000],   # increased from 1500 → 3000 chars
            ground_truth=gt,
            response=answer,
        )

        # 4. Groq judge (70B)
        groq_raw    = _call_groq_judge(judge_input)
        groq_scores = _parse_scores(groq_raw)
        time.sleep(JUDGE_DELAY)

        # 5. Cerebras judge
        cerebras_raw    = _call_cerebras(judge_input)
        cerebras_scores = _parse_scores(cerebras_raw)
        time.sleep(JUDGE_DELAY)

        result = {
            "id":               qid,
            "category":         category,
            "question":         question,
            "ground_truth":     gt,
            "context_preview":  context[:400],
            "context_words":    len(context.split()),
            "year_filter":      year_filter,
            "answer":           answer,
            "groq_scores":      groq_scores,
            "cerebras_scores":  cerebras_scores,
            "groq_verdict":     _verdict(groq_scores),
            "cerebras_verdict": _verdict(cerebras_scores),
        }
        results.append(result)
        done_ids.add(qid)

        log.info(
            "  Groq: rel=%.2f faith=%.2f corr=%.2f -> %s | "
            "Cerebras: rel=%.2f faith=%.2f corr=%.2f -> %s",
            groq_scores["relevance"], groq_scores["faithfulness"], groq_scores["correctness"],
            result["groq_verdict"].upper(),
            cerebras_scores["relevance"], cerebras_scores["faithfulness"], cerebras_scores["correctness"],
            result["cerebras_verdict"].upper(),
        )

        with open(RAW_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete cached results and start fresh")
    args = parser.parse_args()

    if args.reset and RAW_JSON.exists():
        RAW_JSON.unlink()
        log.info("Cleared cached results. Starting fresh.")

    results = run_evaluation()
    log.info("Evaluation complete — %d questions processed.", len(results))
    log.info("Results saved to %s", RAW_JSON)
    log.info("Run  python evaluation/report.py  to print summary tables and charts.")
