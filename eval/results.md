# Evaluation: retrieval and abstention

Observed output from the retrieval and abstention runs.

| measured | script | raw output |
|---|---|---|
| retrieval | `run_eval.py` | `baseline.txt`, `after-clean.txt`, `after-layout.txt` |
| abstention | `run_abstention.py` | `abstention.txt` |

Both score against `questions.jsonl`: 20 questions with answers located by hand in the PDFs. 14 are answerable and measure retrieval, where a hit means one of the top 5 chunks covers the expected pages. The remaining 6 have no answer in the corpus and measure abstention, scored by hand.

Two questions under test, and what the runs show:

1. Does cleaning up PDF text extraction improve retrieval? It depends which part. Removing repeated lines and collapsing whitespace gains 0.143 hit@5. Switching to layout extraction loses 0.143. Applied together they cancel.
2. When the corpus cannot answer, does the system say so? Yes, on all 6 rows, decided at generation rather than by a retrieval score threshold.

---

## Results

Retrieval, three ingest configurations:

| run | extraction | cleaning | hit@5 | MRR@5 | misses |
|---|---|---|---|---|---|
| baseline | plain | off | 0.643 | 0.455 | 5 |
| **A** | plain | **on** | **0.786** | **0.592** | 3 |
| B | layout | on | 0.643 | 0.320 | 5 |

Abstention, shipped configuration:

| question type | n | result |
|---|---|---|
| `unanswerable` | 3 | 3/3 refused |
| `epistemic` | 3 | 3/3 answered the covered part, named the uncovered part |

Shipped: run A, `EXTRACTION_MODE = "plain"` with cleaning on.

Conditions: 14 Aug 2026, `all-MiniLM-L6-v2`, CHUNK_SIZE 500, overlap 50, TOP_K 5, `claude-opus-5` for generation. 485 chunks at baseline, 463 after cleaning, 462 under layout. Abstention run: 8,009 input and 1,956 output tokens, $0.09.

---

## Three runs instead of one

The change was planned as a single variable: layout extraction, whitespace collapse and header stripping applied together.

Measured that way, baseline against run B gives 0.643 against 0.643, which reads as no effect. Measured separately, cleaning contributes +0.143 and layout extraction -0.143. The two effects are close to equal and opposite.

MRR@5 separates them more clearly than hit@5, being sensitive to rank rather than presence: 0.455, 0.592, 0.320. Layout extraction scores below the untreated baseline.

---

## Effect of cleaning

Two questions moved from a miss to rank 1 (`inv-02`, `gov-01`), and one from rank 2 to rank 1 (`inv-03`).

`vm-02` shows the mechanism:

```
                     baseline          after cleaning
chunk holding it     mrm p31           mrm pp30-31
its score            0.289             0.330
its rank             5                 1
```

The gain is not a demotion of boilerplate. The four chunks above it at baseline were ordinary content, and the answer sentence was not inside the chunk that came back: that chunk held a footnote and a URL.

Stripping repeated lines removes 5,194 characters and whitespace collapse removes another 4,845, which shifts every chunk boundary after them and takes the corpus from 485 chunks to 463. The sentence "validation should generally be done by individuals not responsible for development or use" then falls inside one chunk, which scores 0.330 and ranks first. Cleaning works here by re-chunking, not by removing competitors.

Boilerplate text is not inert either. The running header `"Artificial Intelligence Model Risk Management | 46"` embedded on its own scores 0.573 against `un-03` and 0.493 against `val-01`, placing it in the same range as genuine content.

Detection is automatic rather than a fixed list: a line is treated as boilerplate when it appears on at least half of a document's pages after digits and whitespace are normalised.

Digit normalisation is what makes footers detectable. The FEAT paper's footer reads `"Monetary Authority of Singapore 3"`, then `"... 4"`, so counted literally it appears once per page and no frequency test finds it. Masking digits collapses the fifteen variants into one line at 14 of 15 pages. That paper's running header carries no page number and was detectable either way.

---

## Effect of layout extraction

Measured against run A, so that cleaning is held constant. Two questions fell out of the top 5: `inv-02` from rank 1 and `vm-01` from rank 5. `gov-01` fell from rank 1 to rank 5, `vm-02` from 1 to 5, `tp-01` from 1 to 3 and `val-01` from 3 to 4. One improved: `gen-01`, from rank 4 to rank 1.

Layout mode produces cleaner text. Split words, the `Ris k` and `use d` pattern, drop from 53 to 20, counting any one or two letter token that is not a corpus word but joins with the word before it to form one. Retrieval quality still fell.

The probable cause is that layout mode preserves the visual column structure of the page, reordering prose and shifting every chunk boundary. These runs do not establish it.

A measurable improvement in extraction quality produced worse retrieval. Text quality is therefore not a valid proxy for retrieval quality.

---

## Score thresholds and abstention

Rejecting a question when its top retrieval score falls below a cut-off does not work on this corpus.

Answerable and unanswerable scores overlap by 0.361 at baseline, 0.381 after cleaning and 0.407 under layout. The overlap does not close. The lowest scoring answerable question is `vm-02` in all three runs (0.330, 0.330, 0.303); the highest scoring unanswerable question is `un-03`, on the EU AI Act, at 0.690, 0.711 and 0.710. Any cut-off that rejects `un-03` also rejects answerable questions.

Abstention is handled at generation instead. All three unanswerable questions were refused, including `un-03`, which outscores 11 of the 14 answerable questions.

The three epistemic questions ask what one institution does in practice, which the corpus cannot state; it records only regulatory expectations. Each answer covered the part the corpus supports and identified the part it does not:

- `ep-01` gives inventory scope and linkages, then states that the passages do not cover AI embedded in vendor products.
- `ep-02` gives the trigger for the oversight committee, then notes that the assessment method is an open consultation question.
- `ep-03` converts the question into testable criteria, then states that compliance is internal evidence absent from the corpus.

This matches the system prompt, which requires a partial answer to name what is not covered. A binary refused or answered metric would record all three as failures, so the two types are scored separately.

The six verdicts in `abstention.txt` were labelled by hand. At six rows an LLM judge would require validation against the same labels, so the labels are the measurement.

---

## Remaining failures

`feat-01`, `mat-01` and `gen-02` miss in all three runs.

- `feat-01` requests the FEAT principles. The corpus does not list them in one place, so no single chunk answers it.
- `mat-01` and `gen-02` return the correct topic from the wrong location. `mat-01` expects the model risk paper p15 and returns the consultation paper's materiality section at p18. `gen-02` expects consultation p21 and returns consultation p8.

---

## Limits

- Sample size is 14 answerable questions and 6 abstention rows. One question is 7 points of hit@5, sufficient to detect a large regression but not a small gain.
- Boilerplate detection is conservative. The consultation paper carries four alternate running headers that each appear on only 2 of its 30 pages, so all four survive. A threshold low enough to catch them also strips real headings such as `AI Inventory` and `AI Identification`, which appear on exactly 2 pages themselves.
- Chunk count does not explain the layout result. Runs A and B produce 463 and 462 chunks from 207,606 and 207,300 characters, so the fall cannot be a chunk size effect. It is part of the cleaning result, where the corpus goes from 485 chunks to 463.

---

## Unreachable passage

Observed on the shipped build while running the pipeline end to end.

MAS's list of inventory attributes sits almost entirely in one chunk, `mas-ai-risk-guidelines-consultation.pdf_p17_c91`. Its neighbours bracket it: `c90` ends one clause short, mid-word, at *"should capture key attributes to enable effective governance and oversight, as well as risk mana"*, and `c92` resumes after the list.

That chunk is present in the collection, and `collection.get` returns it, but it does not rank high enough to be retrieved. Ranks out of 25:

| query | c91 rank | c91 score | c90 rank | c90 score | top-1 |
|---|---|---|---|---|---|
| What must be recorded in an AI model inventory? | 23 | 0.529 | 3 | 0.639 | mrm 0.702 |
| AI inventory key attributes captured | not in top 25 | - | 1 | 0.633 | c90 0.633 |
| inventory attributes purpose scope model type dependencies | 7 | 0.378 | 2 | 0.439 | mrm 0.474 |
| what do we need to write down about our models? | 14 | 0.357 | not in top 25 | - | mrm 0.480 |

`c90` outranks `c91` on three of the four queries while containing less of the answer. The third query is the clearest case: `c91` contains **purpose**, **scope**, **model type** and **dependencies** verbatim, `c90` contains only *inventory*, and `c90` still ranks higher at 0.439 against 0.378.

The effect is visible in cost. The same inventory question was put to `agent.py` twice:

```
                    searches   API calls   passages   in tok   out tok    cost
run 1                      2           2          8    4,560       486   $0.035
run 2                      6           4         16   13,132       884   $0.088
```

Run 2 searched six times, each time receiving passages that reference the list without containing it, and answered with the material available. Agent spend on a single question is therefore a distribution rather than a fixed figure. The two runs used different corpora, since run 1 predates cleaning, so these figures are not a controlled comparison.
