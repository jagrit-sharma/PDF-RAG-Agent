import anthropic

import config
import generate
import retrieve
import run_eval

TYPES = ("unanswerable", "epistemic")

def abstentionRows(types=TYPES):
    rows = [row for row in run_eval.loadQuestions() if row["type"] in types]
    assert rows, f"no rows of type {types}"
    return rows

def showRow(row, score, answer, sources, unused, usage):
    print("=" * 78)
    print(f"{row['id']}  {row['type']}   top-1 {score:.3f}")
    print("Q:", row["question"])
    print("=" * 78)
    print(answer)
    print("\nSources")
    print("\n".join(sources) if sources else "(none cited)")
    print(f"\nretrieved but never cited: {unused}")
    print(f"tokens: {usage.input_tokens} in / {usage.output_tokens} out")
    print("\nverdict:\n")

def showTotals(rows, in_tok, out_tok):
    print("=" * 78)
    print(f"{len(rows)} rows   {in_tok} in / {out_tok} out tokens")

def main():
    rows = abstentionRows()
    model = retrieve.loadModel()
    collection = retrieve.loadCollection()
    client = anthropic.Anthropic()

    run_eval.showConfig(collection)
    print(f"chat model   {config.CHAT_MODEL}\n")

    in_tok = out_tok = 0
    for row in rows:
        score = retrieve.search(row["question"], model, collection, n_results=config.TOP_K)[0]["score"]
        answer, sources, unused, usage = generate.answerQuestion(row["question"], model, collection, client)
        in_tok += usage.input_tokens
        out_tok += usage.output_tokens
        showRow(row, score, answer, sources, unused, usage)

    showTotals(rows, in_tok, out_tok)

if __name__ == "__main__":
    main()
