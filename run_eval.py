import json
import os
from datetime import datetime

import config
import retrieve

EVAL_PATH = os.path.join(config.BASE_DIR, "eval", "questions.jsonl")

SHORT = {
    "mas-ai-model-risk-mgmt.pdf": "mrm",
    "mas-ai-risk-guidelines-consultation.pdf": "consult",
    "mas-feat-principles.pdf": "feat",
}

# golden set → search each question → rank the first correct hit → hit@5, MRR@5

def loadQuestions(path=EVAL_PATH):
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))

    ids = [row["id"] for row in data]
    assert len(set(ids)) == len(ids), "duplicate id in the golden set"

    unverified = [row["id"] for row in data if row["type"] == "answerable" and not row["verified"]]
    assert not unverified, f"unverified answerable rows: {unverified}"

    return data

def isCorrect(hit, row):
    return (hit["source"] == row["source"]) and (any(hit["page_start"] <= p <= hit["page_end"] for p in row["pages"]))

def rankOfAnswer(hits, row):
    for (rank, hit) in enumerate(hits, 1):
        if isCorrect(hit, row): return rank
    return None

def evaluate(rows, model, collection):
    res = []
    for r in rows:
        hits = retrieve.search(r["question"], model, collection)
        res.append(
            {
                "row": r,
                "rank": rankOfAnswer(hits, r) if r["type"] == "answerable" else None,
                "top1": hits[0]["score"],
                "top1_where": hits[0]["source"] + " " + hits[0]["pages"]
            }
        )
    return res

def short(source):
    return SHORT.get(source, source)

def showConfig(collection):
    print(f"run          {datetime.now():%Y-%m-%d %H:%M}")
    print(f"embed model  {config.EMBED_MODEL}")
    print(f"extraction   {config.EXTRACTION_MODE}")
    print(f"chunk size   {config.CHUNK_SIZE} / overlap {config.CHUNK_OVERLAP}")
    print(f"collection   {config.COLLECTION}  ({collection.count()} chunks)")
    print(f"top k        {config.TOP_K}")
    print()

def showRows(results):
    print(f"{'id':<9}{'type':<14}{'expected':<20}{'rank':>5}  {'top1':>6}  top-1 returned")
    print("-" * 86)

    for r in results:
        row = r["row"]
        expected = f"{short(row['source'])} {row['pages']}" if row["source"] else "(nothing)"

        if row["type"] != "answerable":
            rank = "n/a"
        elif r["rank"]:
            rank = str(r["rank"])
        else:
            rank = "MISS"

        where = f"{short(r['top1_where'].rsplit(' ', 1)[0])} {r['top1_where'].rsplit(' ', 1)[1]}"
        print(f"{row['id']:<9}{row['type']:<14}{expected:<20}{rank:>5}  {r['top1']:>6.3f}  {where}")

def summarise(results):
    answerable = [r for r in results if r["row"]["type"] == "answerable"]
    found = [r for r in answerable if r["rank"]]

    hitAtK = len(found) / len(answerable)
    mrr = sum(1 / r["rank"] for r in found) / len(answerable)

    print("\n" + "=" * 86)
    print("RETRIEVAL")
    print("=" * 86)
    print(f"  hit@{config.TOP_K}   {len(found)}/{len(answerable)}   {hitAtK:.3f}")
    print(f"  MRR@{config.TOP_K}                {mrr:.3f}")
    print(f"  missed  {[r['row']['id'] for r in answerable if not r['rank']] or 'none'}")

    return {"hit@k": hitAtK, "mrr": mrr}

def thresholdCheck(results):
    print("\n" + "=" * 86)
    print("TOP-1 SCORE BY TYPE - can a relevance threshold separate them?")
    print("=" * 86)

    byType = {}
    for kind in ("answerable", "unanswerable", "epistemic"):
        scores = sorted(r["top1"] for r in results if r["row"]["type"] == kind)
        byType[kind] = scores
        print(f"  {kind:<14} n={len(scores):<3} min {scores[0]:.3f}   max {scores[-1]:.3f}")

    floor = min(byType["answerable"])
    ceiling = max(byType["unanswerable"])

    print(f"\n  lowest answerable   {floor:.3f}")
    print(f"  highest unanswerable {ceiling:.3f}")
    if floor > ceiling:
        print(f"  -> separable, gap {floor - ceiling:.3f}")
    else:
        print(f"  -> THEY OVERLAP by {ceiling - floor:.3f}. No single score threshold works.")

def main():
    data = loadQuestions()
    print(f"{len(data)} questions loaded")

    model = retrieve.loadModel()
    collection = retrieve.loadCollection()
    showConfig(collection)

    results = evaluate(data, model, collection)
    showRows(results)
    summarise(results)
    thresholdCheck(results)

if __name__ == "__main__":
    main()