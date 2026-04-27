# Agent-Observability-Evaluation

Hands-on modules for building a **LangGraph/LangChain FinTech support agent** and then making it **observable** (LangSmith tracing) and **safer** (input/output guardrails).

## Why this project exists

LLM agents are powerful, but in real systems you quickly need:

- **Observability**: see routing decisions, retrieval context, latency, and token usage so you can debug and improve quality.
- **Guardrails**: reduce data leakage and unsafe responses using layered defenses (fast rules + ML + LLM validators).

This repo is a small, self-contained playground that demonstrates both—using a mock “SecureBank” customer support scenario.

## Tech stack

- **Python**: runtime
- **LangChain + LangGraph**: agent orchestration (supervisor + specialist agents)
- **Chroma**: local/in-process vector store for RAG over policy docs
- **OpenAI**: chat model + embeddings (and Moderation API in guardrails demo)
- **LangSmith**: tracing / observability
- **Guardrails AI**: output validation (regex + LLM-based validators)
- **Presidio**: local PII detection + redaction (ML/NER)

See `requirements.txt` for the full dependency list.

## High-level architecture

The shared agent lives in `fintech_support_agent.py` and is used by multiple exercises:

- **Supervisor**: classifies intent (`policy` | `account_status` | `escalation`)
- **Policy agent (RAG)**: retrieves from `documents/*.md` and answers *only* from that context
- **Account agent**: looks up a mock account (by `ACC-XXXXX`) and summarizes status/transactions
- **Escalation agent**: empathetic handoff to human support channels

The exercises then wrap this agent with:

- **Observability** (`observability/main.py`): run queries and inspect traces in LangSmith
- **Guardrails** (`guardrails/main.py`): apply layered input/output safety checks

## Setup

### Prerequisites

- Python 3.10+ recommended
- An OpenAI API key
- (For observability) a LangSmith API key

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Copy the template and fill in real values:

```bash
cp .env.example .env
```

Required:

- **`OPENAI_API_KEY`**: used by the agent (LLM + embeddings) and by guardrails demos

For LangSmith tracing:

- **`LANGCHAIN_TRACING_V2=true`**
- **`LANGCHAIN_API_KEY`**
- **`LANGCHAIN_PROJECT`** (any name you want for grouping runs)

## Observability (LangSmith) — how to run

This module demonstrates basic LangSmith tracing, trace inspection, error tracing, and tagged runs.

Run:

```bash
python observability/main.py
```

What you should see locally:

- Logs showing `Intent`, a truncated `Response`, and `Retrieved sources` per query
- A deliberate “missing account” query (`ACC-99999`) so you can find the error path in traces
- Tagged runs you can filter on

Then inspect traces in LangSmith:

- Open [LangSmith](https://smith.langchain.com) and look for runs under your `LANGCHAIN_PROJECT`
- Filter by the tag **`observability`** (set in `observability/main.py`)
- In the trace tree, focus on:
  - **Supervisor routing** (why it chose policy vs account vs escalation)
  - **Retrieval inputs/outputs** (which policy chunks were used)
  - **Latency and token usage** (where time/cost is going)

## Guardrails — how to run

This module demonstrates **layered** guardrails. It intentionally includes both “good” and “bad” queries so you can see what gets blocked or redacted.

### One-time: install Guardrails Hub validators

`guardrails/main.py` uses validators from Guardrails Hub. Install them once:

```bash
guardrails hub install hub://guardrails/regex_match
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/competitor_check
```

### Run the guardrails demo

```bash
python guardrails/main.py
```

What it demonstrates (in order):

- **Strategy 1 (Input, regex)**: blocks risky queries before any LLM call (SSN extraction, harmful content, competitor mentions, etc.)
- **Strategy 2 (Input, Moderation API)**: uses the **OpenAI Moderation API** (free) to catch unsafe intent (violence/hate/self-harm/harassment) that keyword rules can miss.
- **Strategy 3 (Output, Presidio / ML-NER)**: detects and redacts PII (names, emails, phone numbers, credit cards, etc.) using local ML/NER.
- **Strategy 4 (Output, LLM validators)**: uses Guardrails Hub validators (and an LLM-based injection classifier) to catch meaning-based issues like toxicity and competitor mentions.

In the code, you’ll see these combined in a “layered” pipeline: **block early on input**, then **validate/redact on output** (and fall back to a safe response if anything fails).

## Repo layout

- `fintech_support_agent.py`: shared multi-agent pipeline (supervisor + specialists)
- `documents/`: SecureBank policy docs used for RAG
- `observability/main.py`: LangSmith tracing 
- `guardrails/main.py`: input/output guardrails 
