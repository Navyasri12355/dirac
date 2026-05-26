# Domain-Specific Research Assistant — arxiv ML Papers
**Fine-Tuned LLM + RAG Pipeline for Academic Research** | Team Project | 4-Week Implementation

---

## 📋 Project Overview

Build a **RAG-powered research assistant** using a LoRA fine-tuned LLaMA-3.2-3B model on arXiv Machine Learning papers, with a live Gradio interface for paper summarization, Q&A, and related work discovery.

| Item | Detail |
|---|---|
| **Goal** | Production-ready research assistant combining fine-tuned LLM + semantic search |
| **Domain** | arXiv papers (CS.LG + CS.AI, last 2 years) |
| **Base Model** | `meta-llama/Llama-3.2-3B-Instruct` + LoRA adaptation |
| **Dataset** | 800–1k papers, 10k–15k QA pairs |
| **Final Output** | Live Gradio app + HuggingFace Hub models + evaluation report |

---

## ✅ Progress Status

### Phase 1: Setup & Data Collection (Week 1) — **✓ COMPLETE**
- [x] Project scaffolding & shared GitHub repo setup
- [x] arXiv scraper built — 500+ papers collected with metadata
- [x] PDF text extraction pipeline — full-text processed with fallback handling
- [x] Base model validated on Colab T4 GPU — baseline Gradio stub working

**Status:** `data/raw/raw_papers.jsonl` and `data/processed/extracted_papers.jsonl` ready

---

### Phase 2: Dataset Curation & Fine-Tuning (Week 2) — **50% IN PROGRESS**

#### Completed ✓
- [x] **Data cleaning & chunking** — Deduplicated corpus chunked into 512-token segments
  - Output: `data/processed/chunked_corpus.jsonl`
- [x] **QA pair generation** — 10k–15k structured QA pairs generated via Groq API
  - Output: `data/processed/qa_progress.jsonl`
  - HuggingFace Hub: `Navyasri12355/arxiv-qa-dataset`

#### In Progress 🔄
- [ ] **LoRA fine-tuning** — Preparing SFTTrainer with `r=16`, `lora_alpha=32`
  - Target: final loss < 1.5 over 2–3 epochs
  - **ETA:** Next 2 days
- [ ] **FAISS index construction** — Embedding all chunks with sentence-transformers
  - Target: L2 index ready for retrieval testing
  - **ETA:** Day after fine-tuning checkpoint
- [ ] **Baseline ROUGE evaluation** — Comparing base vs. fine-tuned outputs
  - Using held-out 50-paper evaluation set
  - **ETA:** Following FAISS completion

---

### Phase 3: RAG Integration & Evaluation (Week 3) — **NOT STARTED**
- [ ] RAG retrieval pipeline (FAISS querying + prompt templating)
- [ ] End-to-end pipeline integration (model + retrieval + Gradio UI)
- [ ] Fine-tuned model evaluation (ROUGE-L, human preference scoring)
- [ ] Out-of-corpus paper handling (live arXiv ID lookup)

**ETA:** Week 3 (pending Phase 2 completion)

---

### Phase 4: Polish, Deployment & Report (Week 4) — **NOT STARTED**
- [ ] Gradio UI polish & HuggingFace Spaces deployment
- [ ] Model card & dataset card publication
- [ ] Final evaluation report (8–12 pages)
- [ ] GitHub cleanup & documentation
- [ ] Presentation slides & demo video

**ETA:** Week 4

---

## 📁 Project Structure

```
dirac/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── research_assistant_implementation_plan.md   # Full roadmap
├── schema.py                                    # Data schema definitions
│
├── app/
│   ├── app.py                                   # Gradio UI (main application)
│   └── rag_pipeline.py                         # RAG retrieval + inference
│
├── data/
│   ├── raw/
│   │   └── raw_papers.jsonl                    # ✓ 500+ papers (metadata only)
│   └── processed/
│       ├── extracted_papers.jsonl              # ✓ Full text extraction
│       ├── chunked_corpus.jsonl                # ✓ Chunked + deduplicated
│       └── qa_progress.jsonl                   # ✓ Generated QA pairs
│
├── eval/
│   ├── eval_harness.py                         # ROUGE-L evaluation
│   └── sample_qas.json                         # Sample test questions
│
├── model/
│   ├── checkpoints/                            # Fine-tuning checkpoints (to be populated)
│   └── configs/                                # LoRA + training configs
│
├── notebooks/
│   ├── 01_scraper.ipynb                       # arXiv scraper
│   ├── 02_extractor.ipynb                     # PDF text extraction
│   ├── 03_data_cleaning.ipynb                 # Deduplication & chunking
│   └── 04_qa_generation.ipynb                 # QA pair generation
│
└── scripts/                                     # Utility scripts (to be added)
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for local GPU) or Google Colab Pro T4
- 15+ GB storage (for model checkpoints and data)

### Installation

```bash
# Clone repo
git clone https://github.com/Navyasri12355/dirac.git
cd dirac

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Once Phase 3 is complete, start Gradio UI:
python app/app.py

# The app will be accessible at http://localhost:7860
```

---

## 📊 Evaluation Targets

| Metric | Base Model | Fine-tuned + RAG | Status |
|---|---|---|---|
| ROUGE-L (summarization) | 0.18–0.22 | > 0.30 | 🔄 Computing |
| Perplexity (domain corpus) | 35–50 | < 25 | ⏳ Pending |
| Human preference (1–5 scale) | 2.5 | > 3.5 | ⏳ Pending |
| Related work relevance | N/A | > 70% | ⏳ Pending |

---

## 🎯 Key Deliverables

- [x] **Dataset:** `Navyasri12355/arxiv-qa-dataset` on HuggingFace Hub ✓ (In HF Hub)
- [ ] **Model:** `Navyasri12355/llama-3.2-3b-arxiv-lora` (LoRA adapter + model card) — In progress
- [ ] **App:** Live Gradio demo on HuggingFace Spaces — Pending Phase 3
- [ ] **Evaluation:** `eval/results_*.json` + human preference sheet — Pending Phase 2
- [ ] **Report:** Final writeup (8–12 pages, PDF) — Pending Phase 4
- [ ] **Repository:** Clean code + `v1.0` release tag — Pending Phase 4

---

## 📝 Dataset Information

**arxiv-qa-dataset** (HuggingFace Hub: `Navyasri12355/arxiv-qa-dataset`)

- **Size:** 10,000–15,000 structured QA pairs
- **Source:** arXiv papers (CS.LG, CS.AI, 2023–2025)
- **Format:** Instruction-tuning ready
  ```json
  {
    "instruction": "Summarize the main contribution of this paper",
    "input": "[paper abstract/section text]",
    "output": "[reference answer]"
  }
  ```
- **Generation method:** Groq API (`llama-3.1-8b-instant`)
- **Question types:** Problem statement, contributions, methodology, limitations

---

## 📚 Tech Stack

| Component | Technology |
|---|---|
| Base Model | `meta-llama/Llama-3.2-3B-Instruct` (4-bit, bitsandbytes) |
| Fine-tuning | `peft` (LoRA) + `trl` (SFTTrainer) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Search | `faiss-cpu` |
| Evaluation | `evaluate` (ROUGE-L), manual human preference scoring |
| UI | `gradio` (local) + HuggingFace Spaces (deployment) |
| Compute | Google Colab Pro (T4 GPU) + Google Drive storage |
| Data Collection | `arxiv` Python package + `PyMuPDF` |

---

## 📖 Documentation

- **Full Implementation Plan:** See [research_assistant_implementation_plan.md](research_assistant_implementation_plan.md)
- **Data Schema:** See [schema.py](schema.py)
- **Notebooks:** See [notebooks/](notebooks/) for step-by-step walkthroughs
- **Evaluation Harness:** See [eval/eval_harness.py](eval/eval_harness.py)

---

## ✨ Key Milestones Completed

- ✅ **Week 1:** 500+ papers collected, full-text extracted, environment validated
- ✅ **Week 2 (Partial):** Corpus chunked (10k+ chunks), 10k–15k QA pairs generated
- 🔄 **Week 2 (In Progress):** Fine-tuning & FAISS indexing
- ⏳ **Week 3:** RAG pipeline + end-to-end integration
- ⏳ **Week 4:** Deployment, reports, presentation

---

## 📞 Contacts & Links

- **GitHub:** [dirac](https://github.com/Navyasri12355/dirac)
- **Dataset HF Hub:** [`Navyasri12355/arxiv-qa-dataset`](https://huggingface.co/datasets/Navyasri12355/arxiv-qa-dataset)
- **Model HF Hub:** (Coming soon — `Navyasri12355/llama-3.2-3b-arxiv-lora`)
- **Live Demo:** (Coming soon — HuggingFace Spaces)

---

**Last Updated:** May 26, 2026 | Phase 2, 50% Complete
