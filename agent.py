import ast
import operator
import sys

import anthropic

import config
import generate
import retrieve

SAMPLE_QUESTIONS = [
    "Hello",
    "What must be recorded in an AI model inventory?",
    "What is 17% of 4,200?",
]

SEARCH_TOOL = {
    "name": "search_documents",
    "description": (
        "Searches three Monetary Authority of Singapore publications: an information paper on "
        "AI model risk management (observations from a thematic review of banks, December 2024), "
        "a consultation paper on AI risk management guidelines for financial institutions, and "
        "the FEAT principles on fairness, ethics, accountability and transparency.\n\n"
        "Use this for any question about what MAS expects of financial institutions regarding AI "
        "governance, model risk management, AI inventories, risk materiality assessment, model "
        "validation, monitoring, or the FEAT principles.\n\n"
        "Do not use it for general knowledge, for arithmetic, or for questions about a specific "
        "institution's own systems and policies — these documents describe regulatory "
        "expectations, not the practices of any individual firm."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query. Phrase it as the topic being looked up rather than as the "
                    "user's literal question, and use the vocabulary of the documents."
                ),
            }
        },
        "required": ["query"],
    },
}

CALC_TOOL = {
    "name": "calculate",
    "description": (
        "Evaluates a single arithmetic expression and returns the number. Supports "
        "+ - * / ** % and parentheses on plain numbers.\n\n"
        "Use this whenever a question requires arithmetic — percentages, totals, "
        "differences, ratios — rather than working the sum out yourself.\n\n"
        "It does no lookups: pass literal numbers only, never variable names, units "
        "or currency symbols. Write percentages as decimals, e.g. 17% of 4200 is "
        "'0.17 * 4200'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression, e.g. '0.17 * 4200'.",
            }
        },
        "required": ["expression"],
    },
}

TOOLS = [SEARCH_TOOL, CALC_TOOL]

# generate.SYSTEM_PROMPT cannot be reused: it opens "the user message contains numbered passages", which is false on turn one here, and "answer only from those passages", which fights calculate. A pipeline prompt supplies context; an agent prompt supplies tools.
SYSTEM_PROMPT = """You answer questions about Monetary Authority of Singapore publications on AI governance and model risk management, for readers who need to verify every statement against the source documents.

Nothing is retrieved for you. If a question concerns what MAS expects of financial institutions, call search_documents before answering, and call it again with different wording if the first results are thin. If a question needs arithmetic, call calculate rather than working the sum out yourself. If it needs neither, answer directly and briefly.

Search results come back as numbered passages. Every claim about MAS guidance must come from those passages and nothing else. Do not draw on knowledge from your training, even when you are confident it is correct — an answer that cannot be traced to a passage is unusable here, whether or not it happens to be true.

Cite the passage each claim comes from using its bracketed number, placed at the end of the sentence containing the claim: [2]. If a sentence draws on more than one passage, list each: [2][4]. The numbers run continuously across every search you make, so [6] is the sixth passage you have been shown, not the first result of your second search. Cite the number only, not the filename or page — those are attached automatically from the retrieval metadata, and anything you write there yourself could be wrong without a reader being able to tell.

The passages are the closest matches to your query, which does not mean they answer the question. If they do not, say so plainly and stop. If they cover only part of it, answer that part and state which part is not covered — do not close the gap with general knowledge. These documents describe regulatory expectations, so a question about what one particular institution actually does cannot be answered from them at all; say that rather than answering the nearest question you can.

Answer in prose, under 150 words unless the question genuinely needs more. Begin with the answer itself, with no preamble restating the question or referring to "the provided passages"."""

MAX_TURNS = 5

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

# question → model → tool request → we execute → model → answer


def evalNode(node):
    """Walk the parse tree ourselves. Anything not on the whitelist is refused."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evalNode(node.left), evalNode(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evalNode(node.operand))

    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression):
    # not eval(). this string is model output - an untrusted source - and eval() would make it arbitrary code execution. ast.literal_eval will not do either: it rejects operators, so it cannot evaluate "0.17 * 4200"
    try:
        return str(evalNode(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"could not evaluate {expression!r}: {e}"


def dispatch(block, hits, seen, model, collection):
    """The model asked; we decide whether to honour it and we run it."""
    if block.name == "search_documents":
        found = retrieve.search(block.input.get("query", ""), model, collection, n_results=config.TOP_K)

        # two parallel searches overlapped 2-3 chunks of 5 every time it was measured, and a repeated chunk under a second marker reads as a second source. drop repeats before they are numbered
        fresh = [h for h in found if h["chunk_id"] not in seen]
        seen.update(h["chunk_id"] for h in fresh)

        context = generate.buildContext(fresh, start=len(hits) + 1)
        hits.extend(fresh)
        return context or "No new passages - the earlier results already cover this."

    if block.name == "calculate":
        return calculate(block.input.get("expression", ""))

    return f"unknown tool: {block.name}"


def callModel(client, messages):
    return client.messages.create(
        model=config.CHAT_MODEL,
        max_tokens=config.MAX_TOKENS,  # caps thinking + answer together
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    )


def runAgent(question, model, collection, client, max_turns=MAX_TURNS):
    hits, seen, tool_calls = [], set(), []
    messages = [{"role": "user", "content": question}]
    calls = in_tok = out_tok = 0
    capped = True  # only cleared by the model choosing to stop

    for _ in range(max_turns):
        response = callModel(client, messages)
        calls += 1
        in_tok += response.usage.input_tokens
        out_tok += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            capped = False
            break

        # the assistant turn goes back whole - thinking blocks are signed, and rebuilding the list by hand invalidates the turn
        messages.append({"role": "assistant", "content": response.content})

        # every tool_use block from this turn, answered in ONE user message. a tool_use left without a tool_result is a 400 on the next call
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_calls.append((block.name, block.input))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": dispatch(block, hits, seen, model, collection),
                }
            )

        messages.append({"role": "user", "content": results})

    answer = generate.answerText(response)
    sources, unused = generate.sourcesUsed(answer, hits)

    return {
        "answer": answer,
        "sources": sources,
        "unused": unused,
        "tools": [name for name, _ in tool_calls],
        "calls": calls,
        "max_turns": max_turns,
        "capped": capped,  # True = we stopped it, it did not finish
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def show(run):
    if run["capped"]:
        print(f"!! hit the {run['max_turns']}-turn cap - this is whatever it had mid-thought\n")

    print(run["answer"])
    print("\nSources")
    print("\n".join(run["sources"]) if run["sources"] else "(none cited)")
    print(f"\nretrieved but never cited: {run['unused']}")
    print(f"tools: {run['tools'] or '(none)'}")
    print(f"api calls: {run['calls']} | tokens: {run['input_tokens']} in / {run['output_tokens']} out\n")


def main(questions):
    model = retrieve.loadModel()
    collection = retrieve.loadCollection()
    client = anthropic.Anthropic()

    for question in questions:
        print("=" * 78)
        print("Q:", question)
        print("=" * 78)
        show(runAgent(question, model, collection, client))


if __name__ == "__main__":
    main(sys.argv[1:] or SAMPLE_QUESTIONS)
