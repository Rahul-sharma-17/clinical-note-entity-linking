# Clinical Note Entity Linking to a Medical Ontology

## Phase 1

From the project root with Python 3.12:

```powershell
python -m src.fetch_public_data
python -m unittest discover -s tests -v
```

If you use uv, replace `python` with `uv run --python 3.12 python`. The pipeline uses Python's standard library. It checks SHA-256 for both downloads and reuses verified files in `data/raw/`.

The three interchangeable CSVs are written to `data/processed/`. The additional `splits.csv` preserves the upstream train, development (validation), and test partitions.

| File | Columns |
| --- | --- |
| notes.csv | note_id, text |
| annotations.csv | note_id, start, end, concept_id |
| terminology.csv | concept_id, concept_name, synonyms |
| splits.csv | note_id, split |

Offsets are zero-based Python character indices with an exclusive end. The converter joins each PubTator title and abstract with a single space and checks each original annotation against its source surface form. IDs are loaded as strings. Synonyms use a pipe separator, with an empty field when none are available.

[BC5CDR on Hugging Face](https://huggingface.co/datasets/bigbio/bc5cdr) supplies PubMed titles, abstracts, and MeSH annotations. The [pinned MONDO release](https://github.com/monarch-initiative/mondo/releases/tag/v2026-11-03) supplies MeSH and SNOMED CT exact-match cross-references, disease labels, and exact synonyms. Only disease annotations that resolve unambiguously to one SNOMED CT code are exported. Terminology labels and aliases are MONDO terms, and the converter builds them from the ontology independently of all corpus mentions. The resulting public benchmark has partial disease annotation coverage. This limits the interpretation of mention-level scores, especially when a prediction matches a source disease annotation that the crosswalk dropped.

When the DrivenData files become available in the stated schema, point `NOTES_PATH`, `ANNOTATIONS_PATH`, and `TERMINOLOGY_PATH` in `src/config.py` to them. Point `SPLITS_PATH` to a matching split manifest or set it to `None` to use a deterministic 80/20 train/validation split. The fetch command always writes the public dataset to `data/processed/`.

## Phase 2

Run `python -m src.baseline` from the project root to evaluate the exact-string baseline on the configured validation split. The matcher uses case-insensitive literal names and synonyms with word boundaries. It takes the longest match at each eligible position, scans left to right without overlapping predictions, and emits confidence `1.0`. If several concepts share a string, a preferred name wins over a synonym, with ties resolved by lexically sorting concept IDs.

Mention F1 scores exact note IDs and character spans. Linking Accuracy measures the fraction of detected exact spans with a correct concept ID. End-to-End F1 scores exact note, span, and concept ID combinations. F1 is micro-averaged across notes. A metric with a zero denominator returns `0.0`.

| Validation notes | Gold links | Predictions | Mention F1 | Linking Accuracy | End-to-End F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 2,587 | 4,170 | 0.392482 | 0.864253 | 0.339204 |

## Phase 3

Run `python -m src.ner_predictor --fallback` to benchmark the standard-library mention detector. It learns a dictionary of annotated surfaces from the training split and returns character offsets in the original note. The public validation split scored **0.691033 Mention F1** across 500 notes, compared with **0.392482** for Phase 2. Warm CPU inference averaged **0.474 ms per note** with a **0.836 ms p95**.

For cloud training, copy the project and its processed CSVs to Colab or Kaggle, enable a GPU and internet access, and run these commands from the project root. The runtime needs CUDA-enabled PyTorch 2.4 or newer.

```sh
python -m pip install "transformers>=4.48,<5" safetensors
python -m notebooks.train_ner_colab
```

The script fine-tunes DistilBERT for BIO tagging using overlapping token windows and exports FP16 safetensors and the tokenizer to `models/ner_model/`. Copy the entire directory back to the local project. Install CPU PyTorch and the same Transformers dependencies locally, then run `python -m src.ner_predictor --model-dir models/ner_model` to measure CPU latency and check the exported model against the Phase 2 validation score. The default command loads that directory if present and uses the training dictionary if it is absent. The exported-model comparison appears in Phase 5.

## Phase 4

Run `python -m src.entity_linker --lexical` to reproduce the dependency-free CPU benchmark. The retriever ranks concepts by the best character-trigram similarity among their names and synonyms. It gives exact aliases priority and returns up to five distinct concepts. Confidence estimates come from smoothed training-set correctness rates by candidate rank and similarity band. They are fitted on gold mention spans, so their reliability on automatically detected spans may differ.

| Validation gold spans | Recall@1 / standalone Linking Accuracy | Recall@5 | Top candidate Brier score | Peak process RAM |
| ---: | ---: | ---: | ---: | ---: |
| 2,587 | 0.596444 | 0.682644 | 0.155644 | 59.3 MiB |

To build the semantic index, use the same cloud setup as Phase 3 and run from the project root.

```sh
python -m pip install "transformers>=4.48,<5" safetensors numpy
python -m notebooks.build_linker_colab
```

The CUDA script embeds terminology names and synonyms with SapBERT and exports an FP16 NumPy array, local encoder files, and a confidence table fitted on training spans to `models/linker_index/`. Copy the entire directory back to the local project. With CPU PyTorch and the same dependencies installed locally, run `python -m src.entity_linker --index-dir models/linker_index` for the semantic benchmark. The index must match the configured terminology. When rebuilding, choose a new output directory with `--output`. The default evaluation command uses the semantic directory if it exists and otherwise uses the lexical retriever. The evaluated SapBERT pipeline results appear in Phase 5.

## Phase 5

Run `python -m src.pipeline` from the project root to evaluate the complete pipeline on the configured validation split. For a single note, call `src.pipeline.predict(note_text)`. Each result contains the original offsets and surface text, plus the top concept ID, concept name, and confidence. A detected span with no candidate has null concept fields and confidence. It still contributes to mention detection and to the Linking Accuracy denominator when its span matches gold.

| System | Mention F1 | Linking Accuracy | End-to-End F1 |
| --- | ---: | ---: | ---: |
| Exact String Lookup Baseline | 0.392482 | 0.864253 | 0.339204 |
| Training dictionary + lexical linker | 0.691033 | 0.615006 | 0.424989 |
| DistilBERT + SapBERT | 0.584502 | 0.682449 | 0.398893 |
| Training dictionary + SapBERT | 0.691033 | 0.661132 | 0.456864 |

These scores use all 500 public validation notes. Linking Accuracy is conditional on exact span matches, so systems can have different sets of mentions in that denominator. The pipeline and review app use the training dictionary with SapBERT when the index is present and with the lexical linker otherwise. The benchmark prints which detector and retriever ran.

## Phase 6

Install the review app dependencies and start the local server from the project root.

```sh
python -m pip install -r app/requirements.txt
python -m app.review_ui
```

Open [Clinical Note Review](http://127.0.0.1:8000). Select a validation note or choose **Paste a note**, enter text, and select **Analyze note**. The highlighted mentions and review table come from the configured Phase 5 pipeline.

Select **Approve** or **Reject** for each mention. Select a pressed button again to clear the decision. Decisions stay in the browser tab when you switch notes. Both download formats include only the decided mentions for the current note, along with their offsets, concept IDs, and confidence. Download the reviews before refreshing or closing the tab.

Use the **Preferred Term** dropdown to select one of up to three retrieved concepts. Downloads use the selected concept and confidence. The low-confidence checkbox filters the display while keeping decisions and exports intact. **Metrics** displays the latest saved validation run. Run `python -m src.pipeline` to regenerate `results/benchmark_metrics.json`.

For the browser tests, install Playwright with `python -m pip install playwright`. The tests use Microsoft Edge on Windows and Playwright Chromium elsewhere. If needed outside Windows, install Chromium with `python -m playwright install chromium`. Run `python -m unittest discover -s tests -v`.


## Import the official challenge data

The public dataset is the default. Once you have the challenge files, place
`mimic-iv_notes_training_set.csv`, `train_annotations.csv`, and the unpacked international
RF2 release together in `data/official/raw/`. From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.import_challenge data/official/raw
$env:CODER_REVIEW_DATASET = "official"
.\.venv\Scripts\python.exe -m src.pipeline
.\.venv\Scripts\python.exe -m app.review_ui
```

Stop an existing review server before starting the official one. This profile has
its own split, model directory and benchmark file. Initially it uses the lexical
linker. For dense retrieval, build the SapBERT index with the same profile set
in the GPU runtime and copy the result into `models/official/linker_index/`.
Set `CODER_REVIEW_DATASET` to `public` to use the public benchmark again.

Following [Team SNOBERT's preprocessing](https://github.com/drivendataorg/snomed-ct-entity-linking/blob/main/2nd%20Place/src/preprocess.py),
the importer gathers the finding, procedure and body structure hierarchies through
active RF2 is-a relationships and includes the three extra finding IDs listed by Team SNOBERT. It uses active English fully specified names as
display labels and active English synonyms as aliases. It preserves original
text and exclusive-end offsets and splits whole notes deterministically (80/20).
These are local validation scores. The importer validates spans and concept
references before publishing the CSVs and refuses to overwrite an existing dataset.
