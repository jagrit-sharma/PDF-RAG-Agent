"""Chunk-size experiment: which CHUNK_SIZE actually retrieves the right passage?

    python experiment.py

Builds one collection per candidate chunk size, then scores each against
questions whose answers were located by hand. Prints the evidence and a verdict.
"""

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

import config
import ingest

hf_logging.set_verbosity_error()

CHUNK_SIZES = [300, 500, 1000, 2000, 4000]
OVERLAP_RATIO = 0.1
TOP_K = 5

# question -> (source pdf, pages the answer is actually on)
QUESTIONS = {
    "What are the FEAT principles?": ("mas-feat-principles.pdf", [1, 6]),
    "How should a financial institution validate an AI model before deployment?": ("mas-ai-model-risk-mgmt.pdf", [29, 30, 31]),
    "What must be recorded in an AI model inventory?": ("mas-ai-model-risk-mgmt.pdf", [13, 14]),
}

# sizes → collections → token budget → rank per question → MRR → verdict

def buildCollections(model, client):
    pages = ingest.loadCorpus()
    chunk_sets, collections = {}, {}

    for size in CHUNK_SIZES:
        chunks = ingest.chunkPages(pages, size=size, overlap=int(size * OVERLAP_RATIO))
        col = client.get_or_create_collection(name=f"mas_docs_{size}")
        if col.count() != len(chunks):
            vectors = np.asarray(model.encode([c["text"] for c in chunks]), dtype=np.float32)
            col.upsert(
                ids=[c["chunk_id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                embeddings=vectors,
                metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
            )
        chunk_sets[size], collections[size] = chunks, col
        print(f"{size:>5} chars -> {len(chunks):>4} chunks, {col.count():>4} stored")

    return chunk_sets, collections


def tokenBudget(model, chunk_sets):
    limit = model.max_seq_length
    print(f"\n{config.EMBED_MODEL} truncates input at {limit} tokens.\n")
    print(f"{'size':>6} {'median tok':>11} {'max tok':>8} {'% truncated':>12} {'chars embedded':>15}")

    truncation = {}
    for size, chunks in chunk_sets.items():
        counts = sorted(len(model.tokenizer.encode(c["text"])) for c in chunks)
        median = counts[len(counts) // 2]
        truncation[size] = 100 * sum(n > limit for n in counts) / len(counts)
        embedded = min(size, int(size * limit / median))
        print(f"{size:>6} {median:>11} {counts[-1]:>8} {truncation[size]:>11.0f}% {embedded:>15}")

    return truncation


def hits(model, col, question, k=TOP_K):
    r = col.query(query_embeddings=[model.encode(question)], n_results=k)
    return [
        {"source": m["source"], "page": m["page"], "score": 1 - d / 2, "text": t}
        for m, d, t in zip(r["metadatas"][0], r["distances"][0], r["documents"][0])
    ]


def rankOfAnswer(hs, source, ok_pages):
    for i, h in enumerate(hs, 1):
        if h["source"] == source and any(abs(h["page"] - p) <= 1 for p in ok_pages):
            return i
    return None


def rankTable(model, collections):
    ranks = {size: {} for size in collections}

    for question, (source, ok_pages) in QUESTIONS.items():
        print("=" * 92)
        print("Q:", question)
        print("   answer lives in:", source, "p" + "/".join(map(str, ok_pages)))
        print("=" * 92)
        print(f"{'size':>6} {'top1 cos':>9}  {'top-1 returned':<42} {'rank of real answer':>19}")

        for size, col in collections.items():
            hs = hits(model, col, question)
            rank = rankOfAnswer(hs, source, ok_pages)
            ranks[size][question] = rank
            top = f"{hs[0]['source'][:26]} p{hs[0]['page']}"
            print(f"{size:>6} {hs[0]['score']:>9.3f}  {top:<42} {(rank or 'not in top 5'):>19}")
        print()

    return ranks


def showTopHits(model, collections, question, preview=240):
    print("=" * 92)
    print(f"TOP HIT AT EACH SIZE — {question!r}")
    print("=" * 92)
    for size, col in collections.items():
        h = hits(model, col, question)[0]
        print(f"--- {size} chars | {h['source']} p{h['page']} | cos {h['score']:.3f} ---")
        print("   " + " ".join(h["text"].split())[:preview] + "...\n")


def truncationProof(model, chunk_sets):
    head = " ".join(chunk_sets[max(CHUNK_SIZES)][0]["text"].split())[:1500]
    a = model.encode(head + " " + "zebra quantum pancake " * 30)
    b = model.encode(head + " " + "entirely different trailing content " * 30)

    print("=" * 92)
    print("PROOF: text past the token limit is invisible to retrieval")
    print("=" * 92)
    print(
        "same 1500-char head, completely different tails ->",
        "IDENTICAL embedding" if np.allclose(a, b) else "different embeddings",
    )
    print()


def verdict(ranks, truncation):
    mrr = {
        size: sum(1 / r if r else 0 for r in per_question.values()) / len(QUESTIONS)
        for size, per_question in ranks.items()
    }
    best = max(mrr, key=lambda s: (mrr[s], -truncation[s]))

    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"{'size':>6} {'MRR@5':>7} {'truncated':>11}   ranks per question")
    for size in CHUNK_SIZES:
        per_q = " ".join(f"{ranks[size][q] or '-':>2}" for q in QUESTIONS)
        print(f"{size:>6} {mrr[size]:>7.3f} {truncation[size]:>10.0f}%   {per_q}")

    print(f"\nBest: CHUNK_SIZE = {best}  (MRR {mrr[best]:.3f}, {truncation[best]:.0f}% truncated)")
    print(f"Currently in config.py: CHUNK_SIZE = {config.CHUNK_SIZE}")
    return best, mrr


def main():
    model = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=config.DB_PATH)

    chunk_sets, collections = buildCollections(model, client)
    truncation = tokenBudget(model, chunk_sets)
    ranks = rankTable(model, collections)
    showTopHits(model, collections, "What are the FEAT principles?")
    truncationProof(model, chunk_sets)
    verdict(ranks, truncation)


if __name__ == "__main__":
    main()
