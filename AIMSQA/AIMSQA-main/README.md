# Fine-grained context retrieval and QA pipeline

This directory is a self-contained, cleaned release of the pipeline accompanying the paper.

## Pipeline

For every statement, the 11-label distilled ModernBERT classifier first selects a local context window for each criterion. The LLM answers the corresponding questions from these contexts. If the classifier finds no context for a criterion, the fine-tuned SentenceBERT model retrieves the three most relevant five-sentence chunks and the LLM answers from those instead.

## Layout

```text
paper_code_release/
├── pipeline.py                 # runnable end-to-end pipeline
├── models.py                   # 11-label context classifier
├── prompts.py                  # labels, criteria, and QA prompt
├── requirements.txt
├── data/
│   ├── AU.csv                  # statement sentences
│   └── questions.csv           # paper questions
├── original_code/              # exact copies retained for provenance
│   ├── main_gemma.py           # original main experiment entry point
│   ├── functions.py
│   ├── models2-Copy1.py        # model class actually referenced by main
│   └── questions.csv
├── training/
│   ├── train_sentencebert_original.py # exact copy of ft-sb/main.py
│   ├── generator*.py           # original retrieval/QA experiments
│   └── AU_train.csv            # local-only SentenceBERT fine-tuning data
├── models/
│   ├── modernbert/             # classifier backbone and tokenizer
│   ├── context_classifier.pth  # fine-tuned 11-label weights
│   └── sentencebert_retriever/ # fallback dense retriever
└── outputs/
```

## Run

Python 3.10+ is recommended. Install Ollama separately, make sure its server is running, and pull the generation model used by the experiment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull gemma3:27b
python pipeline.py --statement-id 12532
```

For a fast end-to-end check before a long run:

```bash
python pipeline.py --statement-id 248 --max-questions 3 --llm gemma3:1b \
  --output outputs/smoke_test.csv
```

Run the complete dataset by omitting `--statement-id`. Use `--device cpu`, `--device cuda`, or `--device mps` to override automatic device selection. Outputs are written to `outputs/answers.csv` by default.

The input statement CSV must contain `statement_id`, `sentence_orig_idxs`, and `sentence`. A custom question CSV may use the simple columns `label` and `question`; labels must match those defined in `prompts.py`.

## Reproducibility notes

- The default classifier threshold is `0.7`.
- Classifier contexts use a two-sentence window on either side.
- SentenceBERT fallback uses five-sentence chunks, stride two, and top-3 retrieval.
- LLM temperature is fixed at `0.0` for stable QA.
- Unparseable LLM answers are retried up to five times. If classifier-based QA still cannot be parsed, the pipeline retries QA with SentenceBERT-retrieved contexts, matching the original fallback intent.
- The included model files are large (about 3.1 GB total). For a public Git repository, publish them through Git LFS or a model repository rather than ordinary Git objects.

## Source provenance

`pipeline.py` is the cleaned, configurable implementation. To make the release auditable, `original_code/main_gemma.py` is a byte-for-byte copy of the repository's original `main_gemma.py`; the original FT source is likewise preserved as `training/train_sentencebert_original.py`.

The fallback model bundled under `models/sentencebert_retriever` is specifically `ft-sb/output_retriever_win5`, which is the path referenced by `main_gemma.py`. It is not the similarly named `output_retriever` model.

`training/AU_train.csv` is intentionally excluded from the public Git repository. The local source corpus contains unreviewed contact and credential-like text copied from source statements; review/redact it and confirm redistribution rights before publishing it separately.

The preserved historical scripts contain machine-specific paths and code-version drift, so they are included as experimental provenance rather than the recommended entry points. In particular, the original main imports `SentenceBERTForDownstreamTask`, which is present in `models2-Copy1.py` but absent from the later `models2.py`, and its historical `criterion_mapping` call no longer matches the later helper signature. Those issues are resolved in `pipeline.py`.
