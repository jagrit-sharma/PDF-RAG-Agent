# PDF RAG Agent

A retrieval augmented generation agent that answers questions from a collection of PDF documents. It reads the PDFs once and builds a searchable index of their text. Asked a question, it finds the passages most likely to contain the answer and a language model writes the answer from those passages alone. Every claim carries a citation back to the file and page it came from, and when the documents do not hold the answer the agent says so rather than inventing one.

The corpus here is three Monetary Authority of Singapore publications on AI governance: an information paper on AI model risk management, drawn from a thematic review of banks in December 2024, a consultation paper proposing AI risk management guidelines for financial institutions, and the FEAT principles on fairness, ethics, accountability and transparency. Regulatory documents suit this problem. They are long, heavily cross referenced, and read by people who need to check a statement against its source rather than take an answer on trust.

Every change to the pipeline was measured rather than assumed.

## What it does

Three MAS PDFs, 95 pages, are split into 463 passages and embedded locally. A question retrieves the five closest passages, and those passages are the only material the model is allowed to use. Citation numbers are resolved back to filename and page from the retrieval metadata rather than written by the model, so a reader can check each sentence against the source.

When the corpus does not contain an answer, the system says so instead of producing one.

```
$ python generate.py "Who should carry out model validation, and why should it not be the developer?"

Validation should be performed by personnel or functions that are competent — possessing
the necessary expertise — and objective, and that are independent of the development and
deployment teams [1]. In practice this typically means an independent unit reviewing the
AI development process and documentation, assessing that the AI performs and behaves as
intended, and carrying out pre-deployment checks [4]. Validation may also take the form of
independent checks of the development process, whether by an independent party or another
peer developer [3].

On the rationale, the passages cite the US Federal Reserve/OCC SR Letter 11-7, which states
that validation should generally be done by individuals who are not responsible for
development or use and who have no stake in whether a model is determined to be valid [2].
Relatedly, the process is meant to provide effective challenge to developers, covering areas
such as conceptual soundness of design, data input suitability and quality, implementation
integrity, and evaluation measures and performance thresholds [1].

Note that remediation of issues found during validation is typically proposed by developers
rather than validators [4], and that validation frequency may be scaled to the materiality
and complexity of the model [5]. The passages do not elaborate further on the reasoning
behind independence beyond the SR Letter 11-7 reference.

Sources
[1] mas-ai-risk-guidelines-consultation.pdf, p25
[2] mas-ai-model-risk-mgmt.pdf, pp30-31
[3] mas-ai-model-risk-mgmt.pdf, pp8-9
[4] mas-ai-model-risk-mgmt.pdf, p30
[5] mas-feat-principles.pdf, p8

retrieved but never cited: []
tokens: 1302 in / 406 out
```

The closing sentence is the system marking its own limit, and the empty `retrieved but never cited` line reports that all five retrieved passages were used. The token counts are the billed ones: 1,302 in covers the system prompt, the five passages and the question, and 406 out is the answer. At Opus 5 rates that is about $0.017 for this call.

## Results

| measure | value |
|---|---|
| retrieval, hit@5 | 0.786 |
| retrieval, MRR@5 | 0.592 |
| abstention | 6 of 6 |
| generation cost per question | about $0.015 |

Measured against a 20 question set whose answers were located by hand in the PDFs. Method, the three ingest configurations compared, and every figure are in [eval/results.md](eval/results.md).

## Quickstart

Tested on Python 3.13.

```bash
git clone https://github.com/jagrit-sharma/PDF-RAG-Agent.git
cd PDF-RAG-Agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python ingest.py
```

`ingest.py` has to run before anything else. The vector store is not committed, so a fresh clone builds it. The first run also downloads the embedding model, 87 MB, and caches it. It prints:

```
3 PDFs -> 95 pages
95 pages -> 463 chunks
chunk length: min 56, max 500
463 vectors, 384 dimensions each
mas_docs_500: 463 chunks stored
all collections: ['mas_docs_500']
```

Then ask it something:

```bash
python agent.py "What must be recorded in an AI model inventory?"
```

An API key is needed only by `generate.py`, `agent.py` and `run_abstention.py`. Ingestion (`ingest.py`), retrieval (`retrieve.py`) and evaluation (`run_eval.py`) make no API calls and cost nothing.

## Dependencies

Six direct dependencies, pinned in `requirements.txt`:

```
anthropic==0.120.2            # Claude client, used by generate.py, agent.py, run_abstention.py
sentence-transformers==5.6.1  # runs all-MiniLM-L6-v2, turning text into 384 dimension vectors
chromadb==1.5.9               # local vector store, holds the chunks and runs the nearest neighbour search
pypdf==6.14.2                 # extracts text from the corpus PDFs, in plain or layout mode
numpy==2.2.6                  # holds the embedding matrix as float32 on the way into Chroma
transformers==5.14.1          # underlies sentence-transformers, imported directly to quiet its warnings
```

PyTorch is not pinned. It arrives as a dependency of `sentence-transformers` at roughly 380 MB, which is most of the install size.

## Commands

```bash
python ingest.py                         # build the vector store, required once
python retrieve.py "<question>"          # top 5 passages with scores, no API call
python generate.py "<question>"          # cited answer from the top 5 passages
python agent.py "<question>"             # the model chooses: search, calculate, or neither
python run_eval.py                       # hit@5 and MRR@5 against the golden set
python run_abstention.py                 # answers for the 6 non-answerable rows
python eval/showpage.py <prefix> <page>  # print one corpus page
python experiment.py                     # chunk size sweep, rebuilds 5 collections
```

`<prefix>` is any leading part of a corpus filename:

```bash
python eval/showpage.py mas-ai-model 15  # AI model risk management, 50 pages
python eval/showpage.py mas-ai-risk 8    # AI risk guidelines consultation, 30 pages
python eval/showpage.py mas-feat 3       # FEAT principles, 15 pages
```

A prefix matching more than one file, such as `mas-ai`, prints the first in alphabetical order.

`retrieve.py`, `generate.py` and `agent.py` fall back to built-in sample questions when none is given.

## Repository layout

```
.
├── config.py                  every setting: model ids, chunk size, top k, paths
├── ingest.py                  PDFs to pages to chunks to vectors to Chroma
├── retrieve.py                question to top k passages with scores
├── generate.py                passages to a cited answer
├── agent.py                   tool loop, written by hand, with search and calculate
├── run_eval.py                hit@5 and MRR@5 against the golden set
├── run_abstention.py          answers for the non-answerable rows, labelled by hand
├── experiment.py              the chunk size sweep that set CHUNK_SIZE
├── requirements.txt
├── LICENSE
├── corpus/                    the three MAS PDFs
│   ├── mas-ai-model-risk-mgmt.pdf
│   ├── mas-ai-risk-guidelines-consultation.pdf
│   └── mas-feat-principles.pdf
├── chroma_db/                 built by ingest.py, not committed
└── eval/
    ├── questions.jsonl        the golden set, 20 questions
    ├── baseline.txt           raw run output, cleaning off
    ├── after-clean.txt        raw run output, cleaning on, this is what ships
    ├── after-layout.txt       raw run output, layout extraction
    ├── abstention.txt         raw run output, with hand labels
    ├── results.md             what the runs showed
    └── showpage.py            print one corpus page, used to locate answers by hand
```

## The corpus

| document | pages | what it is |
|---|---|---|
| `mas-ai-model-risk-mgmt.pdf` | 50 | information paper, observations from a thematic review of banks |
| `mas-ai-risk-guidelines-consultation.pdf` | 30 | consultation paper proposing guidelines, with open questions to industry |
| `mas-feat-principles.pdf` | 15 | the FEAT principles on fairness, ethics, accountability and transparency |

95 pages in total. Two properties of these documents shape every decision that follows.

**They record expectations, not practice.** The corpus can state what MAS expects a financial institution to do. It cannot state what any particular institution actually does. A similarity search cannot see that distinction, because a question about one firm's own inventory uses the same vocabulary as the guidance about inventories. This reappears in the evaluation set as its own question type and in the tool description the agent reads before deciding to search.

**They are typeset for print.** Nearly every page carries a running header, most carry a footer with a page number, footnotes interrupt sentences mid-paragraph, and extraction splits words across line breaks. None of that is content, and all of it lands in the text that a naive extraction produces. Removing it is the one change in this project that measurably improved retrieval.

## How it works

```
corpus/*.pdf
     │   ingest.py      extract text, strip boilerplate, chunk, embed
     ▼
chroma_db/              463 chunks, 384 dimensions each, written once
     │   retrieve.py    encode the question, nearest neighbour search
     ▼
top 5 passages
     │   generate.py    number them, one API call, resolve citations
     ▼
answer with sources
```

`ingest.py` runs once. Everything after it only reads the stored collection.

`agent.py` sits beside this pipeline rather than inside it. The pipeline always searches. The agent decides whether to search at all, and may search several times before it answers. Both share `retrieve.py` and the citation resolution in `generate.py`.

Two rules hold across the codebase.

**Every setting lives in `config.py`.** Model ids, chunk size, overlap, top k, extraction mode and paths. A setting buried in a function body is invisible to the evaluation harness, which will then report two runs as having identical conditions when they did not. That is why `EXTRACTION_MODE` is a config value and not an argument inside `ingest.readPDF`.

**`retrieve.py` owns loading.** Every module downstream calls `retrieve.loadModel()` and `retrieve.loadCollection()` instead of constructing its own embedding model or Chroma client. `ingest.py` is the writer and `experiment.py` builds five collections of its own, so those two are the exceptions.

### 1. Ingestion (`ingest.py`)

```
PDF → pages → clean → chunks → vectors → Chroma
```

- **Extraction** uses pypdf in `plain` mode. A `layout` mode also exists and was measured against it. Layout produces visibly cleaner text and worse retrieval, so plain ships.
- **Boilerplate removal is frequency based, not a fixed list.** Any line appearing on at least half of a document's pages is dropped, which generalises to a PDF nobody has looked at. Digits are masked first, so `Monetary Authority of Singapore 3` and `... 4` count as one line. Without masking, a footer carrying a page number appears exactly once per page and no frequency test can find it.
- **Chunk size was measured, not guessed.** `all-MiniLM-L6-v2` truncates at 256 tokens, so text past that point is invisible to search however large the chunk is. At 500 characters the median chunk is 100 tokens and 3 of 463 exceed the limit. At 2,000 most would.
- **Chunks record a page span, not a start page.** 98 of 463 chunks, 21%, cross a page break. A citation naming only the page a chunk began on would be wrong one time in five.
- **Embedding runs locally**, 384 dimensions per chunk. Local embedding costs nothing per run, which is what made comparing three ingest configurations affordable.
- **The collection name carries the chunk size**, `mas_docs_500`, so changing `CHUNK_SIZE` writes to a new collection instead of silently mixing two chunkings in one index.

### 2. Retrieval (`retrieve.py`)

```
question → vector → nearest neighbour search → top 5 passages
```

- **The question is embedded with the same model as the chunks.** Query and document have to sit in the same vector space or the distances mean nothing.
- **The reported score is cosine similarity.** Chroma returns a squared L2 distance and the score is `1 - d/2`. That identity holds only because `all-MiniLM-L6-v2` emits unit vectors, which it does.
- **Top 5 is a cost decision as much as a quality one.** Five passages is roughly 1,300 input tokens per question. More passages means more context and a larger bill on every call.
- **Each hit carries its source, page span, text and score.** Everything the citation layer needs travels with the hit, so nothing downstream has to query the store a second time.
- **Scores rank well and threshold badly.** Answerable and unanswerable questions overlap in score in every configuration measured, which is why abstention is not a cut-off. See [eval/results.md](eval/results.md).

### 3. Generation (`generate.py`)

```
passages → numbered context → one API call → answer → resolved sources
```

- **The passages are numbered and handed over as the user message.** The system prompt permits no other evidence, including the model's own training knowledge, even when it is confident and correct.
- **The model cites numbers, never filenames.** Filenames and pages are attached afterwards from the retrieval metadata, so a citation cannot be wrong about where it came from even if the model misremembers the source.
- **Citations are validated against the range supplied.** A number outside it is rendered as invalid rather than quietly printed as though it resolved.
- **A partial answer must name the part it cannot cover.** This is why questions about one firm's own practices come back split, answering what the corpus supports and marking the rest, rather than refused outright.
- **Passages retrieved but never cited are reported.** It shows how much of what retrieval returned the answer actually used.

### 4. The agent (`agent.py`)

```
question → model → tool call → we execute → model → answer
```

- **Two tools, and the model chooses.** `search_documents` and `calculate`, or neither if the question needs neither.
- **The routing lives in the tool descriptions, not in code.** `search_documents` states that these documents describe regulatory expectations rather than any one firm's practices, so the model does not reach for the corpus on questions it cannot answer.
- **The loop is written by hand rather than taken from a framework.** Two API rules drive its shape: the assistant turn goes back unmodified because thinking blocks are signed, and every `tool_use` block must be answered by a `tool_result` in a single user message or the next call fails.
- **`calculate` walks the parse tree against a whitelist of operators instead of calling `eval`.** The expression is model output, which is untrusted input, and `eval` on it would be arbitrary code execution.
- **Passages are deduplicated across searches.** Two searches overlapped by 2 or 3 chunks out of 5 every time it was measured, and a repeated chunk under a new number reads as a second independent source.
- **A five turn cap, and the result records why it stopped.** A run cut off at the cap is marked, so a truncated answer is never mistaken for a finished one.

### 5. Evaluation (`run_eval.py`, `run_abstention.py`)

```
golden set → search each question → rank the first correct hit → hit@5, MRR@5
```

- **20 questions with answers located by hand in the PDFs**, and the harness refuses to run if any answerable row is not marked verified. A question set whose answers were found by the system under test measures nothing.
- **Rows are keyed on source and page range, never on `chunk_id`.** Chunk ids are rebuilt on every re-ingest, so a chunk id key would match nothing after the first configuration change and report a collapse that was an artifact of the key.
- **Three question types.** 14 `answerable` rows measure retrieval; 3 `unanswerable` and 3 `epistemic` rows measure abstention.
- **Both metrics divide by every answerable row**, so a miss counts as zero rather than being dropped from the denominator.
- **Retrieval scoring is deterministic and free**, so a change can be re-measured as often as needed.
- **Abstention is labelled by hand.** At six rows a model judge would itself need validating against those same labels, so the labels are the measurement.

What the runs showed, across three ingest configurations, is in [eval/results.md](eval/results.md).

## Analysis

The three ingest configurations compared, what each change did to retrieval and why, the abstention results, the failures that remain and the limits of the measurement are written up in [eval/results.md](eval/results.md).

## License

MIT, in [LICENSE](LICENSE), covering the code in this repository.

The three PDFs in `corpus/` are publications of the Monetary Authority of Singapore, included here as source material for the retrieval task. They remain the property of MAS and are not covered by the MIT license.
