import sys

import chromadb
from sentence_transformers import SentenceTransformer

import config

SAMPLE_QUESTIONS = [
    "What are the FEAT principles?",
    "How should a financial institution validate an AI model before deployment?",
    "What must be recorded in an AI model inventory?",
]

# question → vector → query → hits → show

def loadModel():
    return SentenceTransformer(config.EMBED_MODEL)


def loadCollection():
    client = chromadb.PersistentClient(path=config.DB_PATH)
    return client.get_collection(name=config.COLLECTION)


def search(question, model, collection, n_results=5):
    results = collection.query(
        query_embeddings=[model.encode(question)],
        n_results=n_results,
    )

    return [
        {
            "chunk_id": chunk_id,
            "source": meta["source"],
            "page": meta["page"],
            "text": doc,
            "distance": d,
            "score": 1 - d / 2,
        }
        for chunk_id, doc, meta, d in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def show(hits, preview=200):
    for rank, h in enumerate(hits, 1):
        text = " ".join(h["text"].split())
        print(f"{rank}. {h['source']}  p{h['page']}   d={h['distance']:.4f}  cos={h['score']:.3f}")
        print(f"   {text[:preview]}...")
        print()


def main(questions):
    model = loadModel()
    collection = loadCollection()

    for question in questions:
        print("=" * 78)
        print("Q:", question)
        print("=" * 78)
        show(search(question, model, collection))


if __name__ == "__main__":
    main(sys.argv[1:] or SAMPLE_QUESTIONS)
