# Multi-Domain Support Triage Agent (HackerRank Orchestrate 24 hr Hackathon)

An automated, terminal-based AI agent that triages and resolves support tickets for **HackerRank**, **Claude**, and **Visa** using a Retrieval-Augmented Generation (RAG) pipeline backed by a hybrid search engine.

---

## Table of Contents

1. [Overview & Approach](#1-overview--approach)
2. [Project Structure](#2-project-structure)
3. [Technical Architecture](#3-technical-architecture)
4. [Workflow — Step by Step](#4-workflow--step-by-step)
5. [Key Technical Features](#5-key-technical-features)
6. [Output Schema](#6-output-schema)
7. [Setup & Installation](#7-setup--installation)
8. [Running the Agent](#8-running-the-agent)

---

## 1. Overview & Approach

The core idea is to build a **grounded support agent** that never hallucinates. Instead of letting the LLM answer from memory, the agent is forced to work only from retrieved support documentation.

This is achieved by implementing a **Retrieval-Augmented Generation (RAG)** system:

1. The support corpus (hundreds of Markdown articles across three domains) is indexed into a hybrid search engine.
2. For every incoming support ticket, the most relevant documentation snippets are retrieved.
3. These snippets are passed as context to an LLM, which then classifies, routes, and responds to the ticket — strictly grounded in the retrieved content.
4. The LLM makes a final decision: **Reply** (if the docs contain an answer) or **Escalate** (if the case is too risky or unsupported).

This approach ensures:
- **High accuracy**: Responses are always grounded in real product documentation.
- **Safe escalation**: Edge cases (stolen cards, site outages, security issues) are automatically escalated to a human.
- **Scope enforcement**: Out-of-domain queries are gracefully declined.

---

## 2. Project Structure

```
.
├── code/
│   ├── agent.py        # LLM reasoning engine (Groq)
│   ├── indexer.py      # Hybrid search: FAISS + TF-IDF with RRF
│   ├── main.py         # Entry point: orchestrates indexing & batch processing
│   
│
├── data/
│   ├── hackerrank/     # HackerRank support articles (.md)
│   ├── claude/         # Claude support articles (.md)
│   └── visa/           # Visa support articles (.md)
│
├── support_tickets/
│   ├── support_tickets.csv         # Full input ticket dataset
│   ├── sample_support_tickets.csv  # Smaller sample for testing
│   └── output.csv                  # Generated triage results
│
├── .env                # API keys (not committed)
└── requirements.txt    # Python dependencies
```

---

## 3. Technical Architecture

The system is composed of three modules that work in a sequential pipeline:

### `indexer.py` — Hybrid Retrieval Engine

Implements a **dual-index search system** for maximum retrieval accuracy:

| Index Type | Method | Strength |
|---|---|---|
| **Dense** | `all-MiniLM-L6-v2` embeddings via FAISS | Semantic / conceptual matching |
| **Sparse** | TF-IDF vectorizer (sklearn) | Exact keyword / product name matching |

The two result sets are merged using **Reciprocal Rank Fusion (RRF)** to produce a single, re-ranked list of the most relevant document paragraphs.

**Key design choices:**
- Articles are split into **paragraphs** (not full files) and short paragraphs (< 30 words) are discarded — this gives the LLM tightly focused context.
- FAISS uses **`IndexFlatIP`** (Inner Product) on L2-normalized vectors, which is mathematically equivalent to **Cosine Similarity**.
- Company metadata is extracted from the file path (`data/hackerrank/...` → `HackerRank`) and used to filter search results to the correct domain.

### `agent.py` — LLM Reasoning Engine

Uses **Groq's `llama-3.1-8b-instant`** to reason over the retrieved context and produce a structured JSON response.

The system prompt strictly enforces:
- Answer only from the provided context.
- Classify the `request_type` into one of: `product_issue`, `feature_request`, `bug`, `invalid`.
- Set `status` to `replied` or `escalated`.
- Escalate if any of the following are detected: stolen items, site outage, security/fraud, or a request for human intervention.

### `main.py` — Orchestrator

The entry point that ties everything together:
- Initializes the `HybridIndexer` and builds the index from the `data/` directory.
- Reads tickets from an input CSV and iterates through them.
- For each ticket: retrieves context → calls the agent → saves the result.
- Handles **checkpointing** (saves progress to `.tmp` file) for resilience against interruptions.
- Enforces a **rate-limiting delay** between API calls.

---

## 4. Workflow — Step by Step

```
                          ┌─────────────────┐
                          │  data/ (corpus)  │
                          │  .md articles    │
                          └────────┬────────┘
                                   │ Split into paragraphs
                                   │ Filter < 30 words
                                   ▼
                    ┌──────────────────────────┐
                    │      HybridIndexer       │
                    │  ┌────────┐ ┌──────────┐ │
                    │  │ FAISS  │ │  TF-IDF  │ │
                    │  │(Dense) │ │ (Sparse) │ │
                    │  └────────┘ └──────────┘ │
                    └──────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Ticket: "How do I reset     │
                    │   my HackerRank password?"   │
                    └──────────────┬──────────────┘
                                   │ hybrid_search(query, company)
                                   │ → RRF merges dense + sparse
                                   ▼
                    ┌──────────────────────────────┐
                    │   Top-K relevant paragraphs   │
                    │   (filtered by company)       │
                    └──────────────┬───────────────┘
                                   │ Passed as context
                                   ▼
                    ┌──────────────────────────────┐
                    │        SupportAgent           │
                    │  Groq: llama-3.1-8b-instant  │
                    │                              │
                    │  • Classify request type     │
                    │  • Reply or Escalate?         │
                    │  • Generate grounded response │
                    └──────────────┬───────────────┘
                                   │ Structured JSON
                                   ▼
                    ┌──────────────────────────────┐
                    │       output.csv              │
                    │  status, response,            │
                    │  product_area, request_type   │
                    └──────────────────────────────┘
```

---

## 5. Key Technical Features

### Hybrid Search with Reciprocal Rank Fusion (RRF)
Dense models capture semantics but miss exact keywords. Sparse models match keywords but miss paraphrases. RRF fuses both ranked lists using the formula:

```
RRF_score(doc) = Σ  1 / (k + rank)
```

where `k=60` is a constant. Documents that rank highly in *both* lists are strongly promoted.

### FAISS Cosine Similarity
Embeddings are L2-normalized before being added to a `faiss.IndexFlatIP` index. This allows Inner Product search to act as Cosine Similarity, which is the standard metric for sentence transformer models.

### Paragraph-Level Granularity
Full articles are chunked into paragraphs by splitting on `\n\n`. Paragraphs shorter than 30 words are dropped. This ensures the LLM receives targeted, dense information rather than irrelevant boilerplate.

### Checkpointing & Fault Tolerance
After processing each ticket, the result is immediately written to a `.tmp` checkpoint file. On restart, the agent detects this file and resumes from where it left off — preventing duplicate API calls and data loss.

### Strict Grounding Enforcement
The system prompt explicitly prohibits the LLM from using its own knowledge. All responses must be derived from the retrieved context, making the system auditable and traceable to specific support documentation.

---

## 6. Output Schema

The agent generates a CSV with the following columns:

| Column | Description | Values |
|---|---|---|
| `Issue` | Original support ticket text | (string) |
| `Subject` | Ticket subject line | (string) |
| `Company` | Identified company | `HackerRank`, `Claude`, `Visa` |
| `status` | Triage decision | `replied` or `escalated` |
| `product_area` | Best-fit product category | (string) |
| `response` | User-facing reply or escalation message | (string) |
| `justification` | Internal reasoning and context used | (string) |
| `request_type` | Classification of the request | `product_issue`, `feature_request`, `bug`, `invalid` |

---

## 7. Setup & Installation

### Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/) (free tier available)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure API Key
Create a `.env` file in the **root of the repository** and add:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 8. Running the Agent

Run from the **repository root**:

```bash
# Full dataset
python code/main.py support_tickets/support_tickets.csv support_tickets/output.csv


The agent will:
1. Load and index the knowledge base.
2. Process each ticket sequentially with a short delay between requests.
3. Save results incrementally to avoid data loss.
4. Print a progress bar showing completion status.
```
---
## 9. Output

<img width="966" height="753" alt="Screenshot 2026-05-03 002953" src="https://github.com/user-attachments/assets/afb0a9c0-f343-4260-a3aa-b7d7d33cf8a9" />

