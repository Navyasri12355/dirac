# Domain-Specific Research Assistant — Implementation Plan
### Fine-Tuned LLM + RAG Pipeline | Team of 3 | 4 Weeks

---

## Project Overview

| Item | Detail |
|---|---|
| **Goal** | Build a RAG-powered research assistant using a LoRA fine-tuned LLaMA-3.2-3B on arXiv ML papers |
| **Labs covered** | Lab 1 (Fine-tuning) + Lab 6 (Custom Dataset) + Lab 11 (Research Assistant) |
| **Team size** | 3 members |
| **Duration** | 4 weeks |
| **Compute** | Google Colab Pro (T4 GPU) + Google Drive + GitHub |
| **Final output** | Live Gradio app + HuggingFace model/dataset + evaluation report |

---

## Team Roles

| Member | Role | Primary Responsibility |
|---|---|---|
| **M1** | Data Engineer | arXiv scraping, PDF extraction, dataset curation, QA pair generation |
| **M2** | Model Engineer | LoRA fine-tuning, FAISS index, RAG retrieval pipeline |
| **M3** | App & Eval | Gradio UI, ROUGE evaluation, human preference scoring, final report |

---

## Phase 1 — Setup & Data Collection (Week 1)

### Milestone 1.1 — Project scaffolding `(Day 1–2)` `All members`
- [x] Create shared GitHub repo with agreed folder structure (`/data`, `/model`, `/app`, `/eval`, `/notebooks`)
- [x] Set up shared Google Drive folder for dataset storage and model checkpoints
- [x] Each member sets up Colab environment: install `transformers`, `peft`, `trl`, `faiss-cpu`, `arxiv`, `PyMuPDF`, `gradio`, `sentence-transformers`, `evaluate`, `datasets`, `bitsandbytes`
- [x] Agree on domain: **CS.LG + CS.AI** (last 2 years), target ~800–1000 papers
- [x] Define shared data schema: `{paper_id, title, abstract, body_text, url, year, category}`

### Milestone 1.2 — arXiv scraper `(Day 2–4)` `M1`
- [x] Build scraper using the `arxiv` Python package — query by category and date range
- [x] Collect metadata: paper ID, title, abstract, authors, PDF URL
- [x] Download PDFs for top 500 papers (filter by citation count or recency)
- [x] Store raw downloads in `/data/raw/`

> **Checkpoint:** 500+ paper metadata records collected and stored as `raw_papers.jsonl`

### Milestone 1.3 — PDF text extraction `(Day 3–5)` `M1`
- [x] Extract full text using `PyMuPDF (fitz)` — page by page
- [x] Fallback to abstract-only for papers with garbled LaTeX extraction
- [x] Strip references section, figure captions, and math-heavy blocks
- [x] Output: `extracted_papers.jsonl` — one record per paper with clean text

### Milestone 1.4 — Baseline environment validation `(Day 3–5)` `M2 + M3`
- [x] **M2:** Load `meta-llama/Llama-3.2-3B-Instruct` in 4-bit on Colab T4 — confirm it fits in memory
- [x] **M2:** Run a simple inference pass to verify the base model responds correctly
- [x] **M3:** Build a minimal Gradio stub (text input → hardcoded output) as UI skeleton
- [x] **M3:** Set up ROUGE-L evaluation harness using the `evaluate` library on 5 sample outputs

> **Checkpoint:** Base model loads and infers. Gradio stub is live. Eval harness returns ROUGE scores.

### Phase 1 Team Sync
- Review `extracted_papers.jsonl` quality together
- Confirm base model fits on Colab without OOM errors
- Align on QA pair generation prompt template (used in Phase 2)

---

## Phase 2 — Dataset Curation & Fine-Tuning (Week 2)

### Milestone 2.1 — Data cleaning & chunking `(Day 8–10)` `M1`
- [x] Deduplicate papers by abstract similarity (cosine > 0.95 threshold)
- [x] Chunk each paper body into 512-token segments with 64-token overlap using `RecursiveCharacterTextSplitter`
- [x] Assign chunk IDs: `{paper_id}_chunk_{n}`
- [x] Output: `chunked_corpus.jsonl` — ready for embedding and fine-tuning

### Milestone 2.2 — QA pair generation `(Day 8–11)` `M1`
- [x] For each abstract, generate 3–4 structured QA pairs using Groq API (llama-3.1-8b-instant model)
- [x] Question types: problem statement, main contribution, methodology, limitations
- [x] Target: **10,000–15,000 QA pairs** total
- [x] Format as instruction-tuning pairs: `{"instruction": "...", "input": "...", "output": "..."}`
- [x] Push final dataset to HuggingFace Hub: `Navyasri12355/arxiv-qa-dataset`

> **Checkpoint:** Dataset published on HuggingFace Hub. QA pairs reviewed and sampled for quality.

### Milestone 2.3 — LoRA fine-tuning `(Day 9–12)` `M2`
- [x] Configure LoRA: `r=16`, `lora_alpha=32`, `target_modules=["q_proj","v_proj"]`, `dropout=0.05`
- [x] Load dataset from HuggingFace Hub and apply chat template formatting
- [ ] Run `SFTTrainer` for 2–3 epochs; use `save_steps=50` to checkpoint to Drive
- [ ] Monitor training loss — target final loss < 1.5
- [ ] Save final adapter weights: `llama-3.2-3b-arxiv-lora/`
- [ ] Push adapter to HuggingFace Hub with a model card

> **Checkpoint:** Fine-tuned adapter saved and accessible. Training loss curve logged.

### Milestone 2.4 — FAISS index construction `(Day 10–12)` `M2`
- [ ] Embed all chunks using `sentence-transformers` (`all-MiniLM-L6-v2`)
- [ ] Build FAISS `IndexFlatL2` over all chunk embeddings
- [ ] Save index to Drive: `faiss_index.bin` + `chunk_metadata.jsonl`
- [ ] Test: query index with a sample question, confirm top-3 retrieved chunks are relevant

### Milestone 2.5 — Baseline ROUGE evaluation `(Day 10–13)` `M3`
- [ ] Hold out 50 papers not seen during training as evaluation set
- [ ] Generate summaries from **base model** (no fine-tuning, no RAG) for all 50 papers
- [ ] Compute ROUGE-L scores against reference abstracts
- [ ] Log baseline scores to `eval/results_base.json`
- [ ] Begin Gradio UI development — wire text input to base model inference

> **Checkpoint:** Baseline ROUGE-L score recorded. Gradio UI calls base model live.

### Phase 2 Team Sync
- M1 demos dataset on HuggingFace Hub
- M2 shares training loss curve and sample fine-tuned outputs
- M3 shares baseline ROUGE scores — agree on target improvement (aim for +5–10 ROUGE-L points)

---

## Phase 3 — RAG Integration & Evaluation (Week 3)

### Milestone 3.1 — RAG retrieval pipeline `(Day 15–17)` `M2`
- [ ] Build retrieval function: embed query → FAISS top-k search → return chunk texts + paper IDs
- [ ] Construct RAG prompt template: `[CONTEXT: {chunks}]\n\nQuestion: {query}\nAnswer:`
- [ ] Test retrieval quality on 10 sample research questions
- [ ] Tune `k` (try k=3 and k=5) — pick based on context window fit and output quality

### Milestone 3.2 — End-to-end pipeline integration `(Day 16–18)` `M2 + M3`
- [ ] **M2:** Expose inference function: `generate(query, mode) → response` supporting three modes
- [ ] **M3:** Wire Gradio UI to fine-tuned model + RAG pipeline
- [ ] Implement three UI modes:
  - **Summarize** — input: arXiv paper ID → output: structured summary
  - **Q&A** — input: free-form question → RAG retrieval + answer
  - **Related work** — input: topic/title → output: list of relevant papers with rationale
- [ ] Display retrieved source chunks in a collapsible UI panel
- [ ] Test end-to-end: paste a paper ID → get summary in < 30 seconds on Colab

> **Checkpoint:** Full pipeline live — paper ID in, structured summary out. All three modes functional.

### Milestone 3.3 — Fine-tuned model evaluation `(Day 17–19)` `M3`
- [ ] Re-run ROUGE-L evaluation on same 50-paper held-out set using **fine-tuned + RAG** model
- [ ] Log scores to `eval/results_finetuned_rag.json`
- [ ] Side-by-side comparison table: base model vs fine-tuned vs fine-tuned+RAG
- [ ] Conduct human preference test: 20 sample outputs scored on 5-point Likert scale
  - Criteria: factual accuracy, coherence, domain relevance, conciseness
- [ ] Collect scores from all 3 team members + 2 external reviewers if possible

### Milestone 3.4 — arXiv paper ID integration `(Day 18–20)` `M1 + M3`
- [ ] **M1:** Build `fetch_paper(arxiv_id)` utility — fetches abstract + metadata on-the-fly for papers not in corpus
- [ ] **M3:** Wire utility to Gradio — users can paste any arXiv ID, not just corpus papers
- [ ] Test with 5 recent papers outside the training corpus

> **Checkpoint:** Evaluation complete. Human preference results collected. UI handles out-of-corpus papers.

### Phase 3 Team Sync
- Review full evaluation results together
- Identify failure cases — where does the model still underperform?
- Finalize feature scope — no new features after this sync

---

## Phase 4 — Polish, Report & Submission (Week 4)

### Milestone 4.1 — App polish & deployment `(Day 22–24)` `M3`
- [ ] Clean up Gradio UI: add title, description, example inputs, and usage instructions
- [ ] Add error handling for invalid paper IDs and empty queries
- [ ] Deploy to HuggingFace Spaces for a permanent shareable demo link
- [ ] Record 2-minute demo video: real research query → summarization → Q&A → related work

### Milestone 4.2 — Model card & dataset card `(Day 22–24)` `M1 + M2`
- [ ] **M2:** Write HuggingFace model card — training data, LoRA config, eval scores, intended use, limitations
- [ ] **M1:** Write HuggingFace dataset card — collection method, domain, size, schema, QA pair generation prompt
- [ ] Both cards published and linked from GitHub repo README

### Milestone 4.3 — Final evaluation report `(Day 22–25)` `M3`
- [ ] Write report sections:
  - Introduction & motivation
  - Dataset curation methodology
  - Model architecture & fine-tuning setup
  - RAG pipeline design
  - Quantitative results (ROUGE-L, perplexity, human preference)
  - Error analysis & failure cases
  - Conclusions & future work
- [ ] Include evaluation tables and training loss chart
- [ ] Target length: 8–12 pages

### Milestone 4.4 — Code cleanup & documentation `(Day 23–25)` `All members`
- [ ] Each member documents their own notebooks with inline comments and section headers
- [ ] Write `README.md`: project overview, setup instructions, how to run each component
- [ ] Tag final GitHub release: `v1.0`
- [ ] Confirm all external links work: HuggingFace model, dataset, Spaces demo

### Milestone 4.5 — Presentation prep `(Day 25–27)` `All members`
- [ ] Build 10–12 slide deck covering: problem, pipeline diagram, dataset, training, results, demo, conclusions
- [ ] Each member owns their section (M1: data, M2: model, M3: eval + demo)
- [ ] Dry-run presentation with timing — aim for 12–15 minutes + Q&A
- [ ] Rehearse live demo as backup for any Colab connectivity issues

> **Final checkpoint:** HF Spaces demo live, report submitted, repo tagged, slides ready.

---

## Evaluation Summary Targets

| Metric | Base model | Target (fine-tuned + RAG) |
|---|---|---|
| ROUGE-L (summarization) | ~0.18–0.22 | > 0.30 |
| Perplexity on domain corpus | ~35–50 | < 25 |
| Human preference score (1–5) | ~2.5 | > 3.5 |
| Related work retrieval relevance | N/A | > 70% judged relevant |

---

## Final Deliverables Checklist

- [ ] `team-name/arxiv-ml-qa` — dataset on HuggingFace Hub (800–1k papers, 2k+ QA pairs)
- [ ] `team-name/llama-3.2-3b-arxiv-lora` — fine-tuned adapter on HuggingFace Hub with model card
- [ ] `team-name/research-assistant` — live Gradio app on HuggingFace Spaces
- [ ] `eval/` folder — `results_base.json`, `results_finetuned_rag.json`, human preference sheet
- [ ] Final written report (PDF, 8–12 pages)
- [ ] GitHub repo — clean codebase, README, tagged `v1.0`
- [ ] 2-minute demo video

---

## Key Dependencies Between Members

```
M1 (dataset ready) ──────────► M2 (fine-tuning starts)
                                      │
M2 (adapter + FAISS ready) ──────────► M3 (wires UI to model)
                                              │
M3 (Gradio stub ready by end W2) ────────────► Integration in W3 is swap-in only
```

M1's QA pairs unlock M2's training. M2's adapter and FAISS index unlock M3's full UI wiring. Plan syncs at the end of each phase to unblock the next.

---

## Tech Stack Quick Reference

| Component | Library / Tool |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Quantization | `bitsandbytes` (4-bit NF4) |
| Fine-tuning | `peft` (LoRA) + `trl` (SFTTrainer) |
| Data collection | `arxiv` + `PyMuPDF` |
| Dataset hosting | `datasets` + HuggingFace Hub |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector search | `faiss-cpu` |
| Evaluation | `evaluate` (ROUGE-L) |
| UI | `gradio` + HuggingFace Spaces |
| Experiment tracking | `wandb` (optional) |
| Compute | Google Colab Pro (T4 GPU) + Google Drive |
