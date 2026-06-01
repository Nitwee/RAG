*This project has been created as part of the 42 curriculum by qrios.*

# RAG Against the Machine

## Description

This project implements a Retrieval-Augmented Generation (RAG) system for
answering questions about the vLLM codebase.

The application indexes source files and documentation from `vllm-0.10.1`,
retrieves the most relevant chunks for a question, and uses the retrieved
context to generate a grounded answer with a local language model.

The first objective is to build a strong retrieval pipeline. The evaluator
mainly checks whether the system retrieves the expected source locations for
each question.

## Architecture

The project is organized around five main steps:

1. Ingestion: read useful files from the vLLM repository.
2. Chunking: split Python and Markdown files into searchable chunks.
3. Indexing: build a BM25 or TF-IDF index from the chunks.
4. Retrieval: return the top-k chunks matching a user question.
5. Answering: send the question and retrieved context to an LLM.

Expected workflow:

```text
question -> retrieval -> context augmentation -> LLM -> grounded answer
```

## Instructions

Install dependencies:

```bash
make install
```

Run the CLI:

```bash
make run
```

Index the vLLM repository:

```bash
uv run python -m student index --repo_path vllm-0.10.1 --max_chunk_size 2000
```

Search a single question:

```bash
uv run python -m student search "What endpoint loads a LoRA adapter?" --k 10
```

Search a dataset:

```bash
uv run python -m student search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results
```

Evaluate retrieval results with the provided moulinette:

```bash
./moulinette/moulinette_pkg/moulinette-ubuntu evaluate_student_search_results \
  data/output/search_results/dataset_docs_public.json \
  datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
  --k 10 \
  --max_context_length 2000 \
  --threshold 0.80
```

## Chunking Strategy

Python files should be chunked around logical code units when possible, such as
classes and functions. If structural parsing fails, the fallback is text-based
chunking with a configurable maximum chunk size.

Markdown files should be chunked around headings and sections, then split again
if a section exceeds the configured maximum size.

Each chunk must keep its original source metadata:

- `file_path`
- `first_character_index`
- `last_character_index`

## Retrieval Method

The baseline retrieval method will be BM25. It is simple, fast, and well suited
for source-code and documentation search because exact identifiers, function
names, paths, and configuration keys often matter.

The system may later be extended with TF-IDF, query expansion, reranking, or
embedding-based retrieval.

## Performance Analysis

The target metrics are:

- Docs questions: Recall@5 >= 80%
- Code questions: Recall@5 >= 50%
- Indexing time: <= 5 minutes
- Warm retrieval: <= 90 seconds for 1000 questions

Performance should be measured with the provided public datasets before trying
private evaluation data.

## Design Decisions

The project starts with a retrieval-first design. Answer generation depends on
the quality of retrieved context, so the first milestone is to produce valid
search result JSON files and improve Recall@5.

The implementation uses Pydantic models to validate input and output formats,
Python Fire for the command-line interface, and tqdm for long-running progress
bars.

## Challenges

The main difficulty is producing chunks that are small enough for evaluation and
LLM context limits, but large enough to preserve useful information.

Another important detail is path normalization. The evaluator expects source
paths compatible with the dataset annotations, usually under
`data/raw/vllm-0.10.1/...`.

## Resources

- vLLM documentation: https://docs.vllm.ai/
- Python `ast` module documentation: https://docs.python.org/3/library/ast.html
- Pydantic documentation: https://docs.pydantic.dev/
- Python Fire documentation: https://github.com/google/python-fire
- BM25 overview: https://en.wikipedia.org/wiki/Okapi_BM25

AI was used to clarify the project requirements, design the initial project
structure, and explain the RAG pipeline. All generated content must be reviewed,
tested, and understood before submission.
