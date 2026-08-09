import os
from bisect import bisect_right
from itertools import groupby

import chromadb
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config

PAGE_SEP = "\n"

# corpus → pages → chunks → vectors → store

def readPDF(file):
    reader = PdfReader(file)
    source = os.path.basename(file)
    return [
        {
            "source": source,
            "page": i + 1,
            "text": page.extract_text(),
        }
        for i, page in enumerate(reader.pages)
    ]


def loadCorpus(corpus_dir=config.CORPUS_DIR):
    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".pdf"))
    assert files, f"no PDFs found in {corpus_dir}"

    pages = []
    for file in files:
        pages.extend(readPDF(os.path.join(corpus_dir, file)))

    print(f"{len(files)} PDFs -> {len(pages)} pages")
    return pages


def chunkPages(pages, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    assert overlap < size, "overlap must be < size, or the window never advances"

    chunks = []

    for source, group in groupby(pages, key=lambda p: p["source"]):
        doc_pages = list(group)

        parts, offsets, page_numbers, cursor = [], [], [], 0
        for p in doc_pages:
            offsets.append(cursor)
            page_numbers.append(p["page"])
            text = p["text"] or ""
            parts.append(text)
            cursor += len(text) + len(PAGE_SEP)
        full_text = PAGE_SEP.join(parts)

        start, c = 0, 0
        while start < len(full_text):
            end = min(start + size, len(full_text))
            # a 500-char window crosses a page break ~20% of the time, so record
            # the span rather than the page the chunk happens to start on
            page_start = page_numbers[bisect_right(offsets, start) - 1]
            page_end = page_numbers[bisect_right(offsets, end - 1) - 1]
            chunks.append(
                {
                    "chunk_id": f"{source}_p{page_start}_c{c}",
                    "source": source,
                    "page_start": page_start,
                    "page_end": page_end,
                    "text": full_text[start:end],
                }
            )
            c += 1
            start += size - overlap

    return chunks


def embedChunks(chunks):
    model = SentenceTransformer(config.EMBED_MODEL)
    vectors = np.asarray(
        model.encode([c["text"] for c in chunks], show_progress_bar=True),
        dtype=np.float32,
    )

    assert len(vectors) == len(chunks), "vectors and chunks are out of alignment"
    print(f"{len(vectors)} vectors, {vectors.shape[1]} dimensions each")
    return vectors


def store(chunks, vectors):
    client = chromadb.PersistentClient(path=config.DB_PATH)
    collection = client.get_or_create_collection(name=config.COLLECTION)
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[
            {
                "source": c["source"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
            }
            for c in chunks
        ],
    )

    print(f"{config.COLLECTION}: {collection.count()} chunks stored")
    print("all collections:", [c.name for c in client.list_collections()])


def main():
    pages = loadCorpus()
    chunks = chunkPages(pages)

    lengths = [len(c["text"]) for c in chunks]
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print(f"chunk length: min {min(lengths)}, max {max(lengths)}")

    vectors = embedChunks(chunks)
    store(chunks, vectors)


if __name__ == "__main__":
    main()
