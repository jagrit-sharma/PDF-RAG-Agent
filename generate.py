import re
import sys

import anthropic

import config
import retrieve

SAMPLE_QUESTIONS = [
    "What are the FEAT principles?",
    "What must be recorded in an AI model inventory?",
]

SYSTEM_PROMPT = """You answer questions about Monetary Authority of Singapore publications on AI governance and model risk management, for readers who need to verify every statement against the source documents.

The user message contains numbered passages retrieved from those documents. Answer using only those passages. Do not draw on knowledge from your training, even when you are confident it is correct — an answer that cannot be traced to a passage is unusable here, whether or not it happens to be true.

Cite the passage each claim comes from using its bracketed number, placed at the end of the sentence containing the claim: [2]. If a sentence draws on more than one passage, list each: [2][4]. Cite the number only, not the filename or page — those are attached automatically from the retrieval metadata, and anything you write yourself there could be wrong without a reader being able to tell.

The passages are the closest matches to the question, which does not mean they answer it. If they do not, say so plainly and stop. If they cover only part of the question, answer that part and state which part is not covered — do not close the gap with general knowledge. A short answer that marks its own limits is more useful here than a complete-looking one.

Answer in prose, under 150 words unless the question genuinely needs more. Begin with the answer itself, with no preamble restating the question or referring to "the provided passages"."""

MARKER = re.compile(r"\[(\d+)\]")

# question → hits → context → Claude → answer → resolved sources

def buildContext(hits, start=1):
    blocks = []
    for i, hit in enumerate(hits, start):
        text = " ".join(hit["text"].split())
        blocks.append(f"[{i}] {hit['source']}, {hit['pages']}\n{text}\n\n")

    return "".join(blocks).strip()


def askClaude(client, question, context, system=SYSTEM_PROMPT):
    return client.messages.create(
        model=config.CHAT_MODEL,
        max_tokens=config.MAX_TOKENS,  # caps thinking + answer together
        system=system,
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    )


def answerText(response):
    return "\n".join(b.text for b in response.content if b.type == "text")


def sourcesUsed(answer, hits):
    cited = sorted({int(n) for n in MARKER.findall(answer)})

    lines = []
    for n in cited:
        if 1 <= n <= len(hits):
            hit = hits[n - 1]
            lines.append(f"[{n}] {hit['source']}, {hit['pages']}")
        else:
            lines.append(f"[{n}] INVALID — only {len(hits)} passages were supplied")

    unused = [n for n in range(1, len(hits) + 1) if n not in cited]
    return lines, unused


def answerQuestion(question, model, collection, client):
    hits = retrieve.search(question, model, collection, n_results=config.TOP_K)
    response = askClaude(client, question, buildContext(hits))
    answer = answerText(response)
    sources, unused = sourcesUsed(answer, hits)

    return answer, sources, unused, response.usage


def show(answer, sources, unused, usage):
    print(answer)
    print("\nSources")
    print("\n".join(sources) if sources else "(none cited)")
    print(f"\nretrieved but never cited: {unused}")
    print(f"tokens: {usage.input_tokens} in / {usage.output_tokens} out\n")


def main(questions):
    model = retrieve.loadModel()
    collection = retrieve.loadCollection()
    client = anthropic.Anthropic()

    for question in questions:
        print("=" * 78)
        print("Q:", question)
        print("=" * 78)
        show(*answerQuestion(question, model, collection, client))


if __name__ == "__main__":
    main(sys.argv[1:] or SAMPLE_QUESTIONS)
