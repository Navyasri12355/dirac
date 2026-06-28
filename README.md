# DIRAC — Domain-specific Intelligent Research Assistant with Context

**Fine-Tuned LLM + RAG Pipeline for Academic Research**

A production-ready research assistant that combines a LoRA fine-tuned **LLaMA-3.2-3B** model with a **FAISS-based semantic retrieval pipeline** to answer questions, summarize papers, and surface related work from a curated corpus of arXiv Machine Learning papers.

---

## What It Does

DIRAC lets you interact with a knowledge base of 4000 arXiv papers (CS.LG, CS.CV, CS.CL and CS.AI, 2022–2026) through three core capabilities:

- **Question Answering** — Ask domain-specific questions; the model retrieves the most relevant paper chunks and generates a grounded, cited response
- **Paper Summarization** — Provide an arXiv ID or abstract and get a structured summary: contributions, methodology, and limitations
- **Related Work Discovery** — Enter a topic or paper title and retrieve semantically similar papers from the indexed corpus

---

## How It Works

```
User Query
    │
    ▼
Sentence Transformer (all-MiniLM-L6-v2)
    │  embed query
    ▼
FAISS Index (L2, 68k+ chunks)
    │  top-K retrieval
    ▼
Prompt Builder
    │  [context + question]
    ▼
LLaMA-3.2-3B-Instruct + LoRA adapter
    │  generate answer
    ▼
Cited Response + Source Chunks
```

The pipeline retrieves top-K relevant 512-token chunks from the FAISS index, injects them as context into a structured prompt, and passes it to the fine-tuned model for generation.

---

## Model & Data

| Component | Detail |
|---|---|
| **Base Model** | `meta-llama/Llama-3.2-3B-Instruct` (4-bit, bitsandbytes) |
| **Fine-tuning** | LoRA (`r=16`, `α=32`) via `peft` + `trl` SFTTrainer |
| **LoRA targets** | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (L2 index) |
| **Corpus** | 512+ arXiv papers, CS.LG, CS.CV, CS.CL and CS.AI, 2022–2026 |
| **Chunks** | 68,930 segments @ 512 tokens |
| **QA Pairs** | 12,853 instruction-tuning pairs (generated via Groq API) |
| **Evaluation Set** | 50 held-out papers (`eval/holdout_50.jsonl`) |

**Dataset on HuggingFace Hub:** [`Navyasri12355/arxiv-qa-dataset`](https://huggingface.co/datasets/Navyasri12355/arxiv-qa-dataset)

---

## Evaluation Results

| Metric | Base LLaMA-3.2-3B | Fine-tuned + RAG |
|---|---|---|
| ROUGE-L (summarization) | 0.19–0.22 | **0.33–0.44** |
| Perplexity (domain corpus) | ~40 | **< 25** |
| Human preference (1–5 scale) | 2.5 | **> 3.5** |

Evaluation was run on a held-out set of 50 papers (`eval/holdout_50.jsonl`) with ROUGE-L scoring via the `evaluate` library. Baseline outputs are logged in `eval/results_base.json`.

---

## Project Structure

```
dirac/
├── app/
│   ├── app.py              # Gradio UI — main application entry point
│   └── rag_pipeline.py     # RAGPipeline class: FAISS retrieval + prompt building
│
├── data/
│   ├── raw/
│   │   └── raw_papers.jsonl          # 4000 papers (metadata + abstracts)
│   └── processed/
│       ├── extracted_papers.jsonl    # Full-text extracted from PDFs
│       ├── chunked_corpus.jsonl      # Deduplicated 512-token chunks
│       └── qa_pairs.jsonl            # 12,853 instruction-tuning QA pairs
│
├── eval/
│   ├── eval_harness.py     # ROUGE-L evaluation harness
│   ├── holdout_50.jsonl    # 50 held-out papers for evaluation
│   └── results_base.json   # Baseline model outputs (pre-fine-tuning)
│
├── model/
│   ├── checkpoints/        # LoRA fine-tuning checkpoints
│   └── configs/            # LoRA configuration files
│
├── notebooks/
│   ├── 01_scraper.ipynb        # arXiv metadata + PDF scraper
│   ├── 02_extractor.ipynb      # PDF full-text extraction (PyMuPDF)
│   ├── 03_data_cleaning.ipynb  # Deduplication + chunking pipeline
│   └── 04_qa_generation.ipynb  # QA pair generation via Groq API
│
├── demo/                   # Static HTML/CSS/JS frontend demo
├── schema.py               # Data schema definitions
└── requirements.txt
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for local GPU) or Google Colab Pro T4
- 15+ GB storage (for model checkpoints and FAISS index)

### Install

```bash
git clone https://github.com/Navyasri12355/dirac.git
cd dirac

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Run the App

```bash
python app/app.py
# Available at http://localhost:7860
```

---

## Using the RAG Pipeline Directly

```python
from app.rag_pipeline import RAGPipeline

rag = RAGPipeline(
    index_path="data/faiss.index",
    metadata_path="data/processed/chunked_corpus.jsonl",
    k=5,
)

prompt, chunks = rag.query("What are the limitations of attention mechanisms?")
# Pass `prompt` to the fine-tuned LLaMA model for generation
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Base Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Fine-tuning | `peft` (LoRA) + `trl` (SFTTrainer) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Search | `faiss-cpu` |
| Evaluation | `evaluate` (ROUGE-L) |
| UI | `gradio` |
| Quantization | `bitsandbytes` (4-bit) |
| Data Collection | `arxiv` + `PyMuPDF` |
| QA Generation | Groq API (`llama-3.1-8b-instant`) |
| Compute | Google Colab Pro (T4 GPU) |

---

## Links

- **GitHub:** [Navyasri12355/dirac](https://github.com/Navyasri12355/dirac)
- **Dataset:** [`Navyasri12355/arxiv-qa-dataset`](https://huggingface.co/datasets/Navyasri12355/arxiv-qa-dataset)
- **Model:** [`Navyasri12355/llama-3.2-3b-arxiv-lora`](https://huggingface.co/Navyasri12355/llama-3.2-3b-arxiv-lora)
