*This project has been created as part of the 42 curriculum by qrios.*

# RAG Against the Machine

## Description

This project implements a Retrieval-Augmented Generation (RAG) system for
answering questions about the `vllm-0.10.1` codebase.

The application reads the vLLM repository, splits source files into searchable
chunks, builds a persistent BM25 index, retrieves the most relevant source
locations for each question, and can generate grounded answers with a local
language model.

The main graded objective is retrieval quality: the evaluator checks whether
the system returns source ranges that overlap with the expected annotations.

## System Architecture

The pipeline is organized as follows:

1. Ingestion: `FilesReader` scans the repository and keeps supported Python,
   documentation, configuration, shell, C/CUDA, and template files.
2. Chunking: `PyChunker` uses Python AST nodes for functions and classes, while
   `TxtChunker` handles the remaining text-like files.
3. Indexing: `BM25Retriever` tokenizes chunk paths and contents, builds a BM25
   index, and stores it under `data/processed/`.
4. Retrieval: CLI commands load the persisted index and return top-k source
   ranges for a query or a dataset.
5. Answering: `LLMAnswerer` builds a source-grounded prompt and uses
   `Qwen/Qwen3-0.6B` by default for answer generation.
6. Evaluation: `Evaluator` computes a local recall@k score, and the provided
   moulinette validates the final JSON files.

```text
question -> retrieval -> source context -> LLM -> grounded answer
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

Run tests and static checks:

```bash
make test
make lint
```

Index the vLLM repository with the default BM25 backend:

```bash
uv run python -m student index \
  --repo_path data/raw/vllm-0.10.1 \
  --max_chunk_size 2000
```

The optional vector index is only built when requested:

```bash
uv run python -m student index \
  --repo_path data/raw/vllm-0.10.1 \
  --max_chunk_size 2000 \
  --method hybrid
```

## Example Usage

Search one question:

```bash
uv run python -m student search \
  "What endpoint loads a LoRA adapter?" \
  --k 10 \
  --method bm25
```

Search the public documentation dataset:

```bash
uv run python -m student search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results \
  --method bm25
```

Evaluate documentation retrieval with the provided moulinette:

```bash
./moulinette-ubuntu evaluate_student_search_results \
  --student_answer_path data/output/search_results/dataset_docs_public.json \
  --dataset_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
  --k 10 \
  --max_context_length 2000 \
  --threshold 0.80
```

Search and evaluate the public code dataset:

```bash
uv run python -m student search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory data/output/search_results \
  --method bm25

./moulinette-ubuntu evaluate_student_search_results \
  --student_answer_path data/output/search_results/dataset_code_public.json \
  --dataset_path datasets_public/public/AnsweredQuestions/dataset_code_public.json \
  --k 10 \
  --max_context_length 2000 \
  --threshold 0.50
```

Generate an answer for one question:

```bash
uv run python -m student answer \
  "How can vLLM serve an OpenAI-compatible API?" \
  --k 10 \
  --method bm25
```

Generate answers from a search result file:

```bash
uv run python -m student answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer
```

## Chunking Strategy

The maximum chunk size is configurable through `--max_chunk_size` and is
validated to stay within the subject limit of 2000 characters.

Python files are parsed with the `ast` module. Top-level classes and functions
become chunks when possible. Oversized functions or methods are split into
smaller ranges while preserving original character offsets. If a Python file
cannot be parsed, it falls back to text chunking.

Text-like files, including Markdown, reStructuredText, TOML, YAML, shell
scripts, templates, and C/CUDA sources, are split into bounded character ranges.
The splitter prefers newline boundaries when a chunk would otherwise exceed the
configured size.

Every retrieved source keeps the metadata required by the evaluator:

- `file_path`
- `first_character_index`
- `last_character_index`

## Retrieval Method

The default retrieval method is BM25, implemented with `bm25s`. Each document
indexed by BM25 contains both the chunk path and the chunk content, which helps
queries that mention source filenames, identifiers, configuration keys, or API
names.

The BM25 index and chunk metadata are saved under:

```text
data/processed/bm25_index/
data/processed/chunks/chunks.json
```

Embedding and hybrid retrieval classes are present as optional experiments, but
BM25 is the default path used for the public evaluation scores below.

## Performance Analysis

The public datasets were evaluated with `k=10`, `max_context_length=2000`, and
BM25 retrieval.

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Docs public | 0.510 | 0.750 | 0.820 | 0.870 | 0.800 |
| Code public | 0.370 | 0.530 | 0.550 | 0.620 | 0.500 |

Both public thresholds pass:

- Documentation questions: Recall@5 = 0.820
- Code questions: Recall@5 = 0.550

Warm retrieval on the public datasets processes 100 questions in less than one
second after the BM25 index is loaded, which is comfortably under the required
90 seconds for 1000 questions.

## Design Decisions

The project is retrieval-first because answer generation depends directly on
the quality of the retrieved sources.

BM25 is used as the baseline because exact matches are valuable for codebase
question answering: function names, class names, endpoint names, command-line
flags, file paths, and configuration fields often appear verbatim in questions.

Pydantic models validate datasets, search results, answers, and chunk metadata.
Python Fire provides the required CLI, and tqdm progress bars are used for
long-running dataset processing and indexing operations.

The index command builds BM25 by default to keep the mandatory path fast and
lightweight. Embedding-based retrieval remains optional because it requires
additional model downloads and memory.

## Challenges

The main challenge is balancing chunk size and context quality. Small chunks
improve source overlap scoring but can lose surrounding explanation; large
chunks preserve context but may miss the 5% overlap threshold or exceed context
limits.

Path normalization is also important. Retrieved paths must match the dataset
annotations, so chunks preserve their original paths under
`data/raw/vllm-0.10.1/...`.

Another challenge is keeping the default workflow reproducible without relying
on large model downloads. For that reason, BM25 is the stable default and local
LLM generation is isolated in the answer commands.

## Resources

- vLLM documentation: https://docs.vllm.ai/
- Python `ast` module: https://docs.python.org/3/library/ast.html
- Pydantic documentation: https://docs.pydantic.dev/
- Python Fire documentation: https://github.com/google/python-fire
- BM25 overview: https://en.wikipedia.org/wiki/Okapi_BM25
- Qwen model family: https://huggingface.co/Qwen

AI was used to clarify the project requirements, review edge cases, improve the
README structure, and identify missing documentation. All generated content was
reviewed and tested before submission.
