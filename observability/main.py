"""
Agent Observability with LangSmith
-------------------------------------------------------
Set up LangSmith tracing, run queries, and inspect the trace tree.

Segments covered:
  2. LangSmith setup & first traces
  3. Trace anatomy — debugging tool calls, latency, token counts
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()
log = logging.getLogger(__name__)
# Verify:
tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false") == "true"
log.info(f"LangSmith tracing enabled: {tracing_enabled}")
if not tracing_enabled:
    log.warning("WARNING: Set LANGCHAIN_TRACING_V2=true to enable tracing!")


# Ensure the repository root is on the import path so we can import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fintech_support_agent import build_support_agent, ask

# ---------------------------------------------------------------------------
# Build the multi-agent pipeline
#
# Use build_support_agent() with collection_name="observability"
# ---------------------------------------------------------------------------

agent = build_support_agent(collection_name="observability", chunk_size=200, chunk_overlap=20)
app = agent["app"]
queries = [
    "What is the overdraft fee?",
    "What is the balance on ACC-12345?",
    "I need to speak to a manager about fraud on my account!",
]
if app is None:
    log.error("Initializing agent failed.")
    exit(1)

# For each query above, print: intent, response (first 200 chars), retrieved_sources
for query in queries:
    log.info(f"\nQuery: {query}")
    result = ask(app, query)
    log.info(f"Intent: {result['intent']}")
    log.info(f"Response: {result['response'][:200]}")
    log.info(f"Retrieved sources: {result['retrieved_sources']}")

# ---------------------------------------------------------------------------
# Inspect traces in LangSmith
# Open https://smith.langchain.com and find your traces and observe agent routing, latency, token counts, and retrieved documents.
# ---------------------------------------------------------------------------

# Inject a deliberate failure and trace the error path
#
# Run a query with a non-existent account number (e.g., ACC-99999).
# Then find the trace in LangSmith:
#   - Supervisor should route to account_agent
#   - Account agent should respond something along the lines of I couldn't find account ACC-99999 in our system.
# ---------------------------------------------------------------------------
log.info("\n" + "=" * 60)
log.info("SEGMENT 3: ERROR TRACING")
log.info("=" * 60)

if app is not None:
    error_query = "What is the balance on ACC-99999?"
    log.info(f"\nQuery: {error_query}")
    result =ask(app, error_query)
    log.info(f"Intent: {result['intent']}")
    log.info(f"Response: {result['response'][:200]}")
    log.info(f"Retrieved sources: {result['retrieved_sources']}")
else:
    log.error("Complete TODO 2 first.")

# ---------------------------------------------------------------------------
# Tag your runs for monitoring
# ---------------------------------------------------------------------------
log.info("\n" + "=" * 60)
log.info("TAGGED RUNS")
log.info("=" * 60)
tagged_queries = [
    ("policy", "What is the overdraft fee?"),
    ("account", "What is the balance on ACC-12345?"),
    ("escalation", "I'm furious! Someone stole money from my account!"),
]

for tag,query in tagged_queries:
    result = app.invoke(
    {
        "query": query,
        "intent": "",
        "response": "",
        "context": "",
        "retrieved_sources": [],
    },
    config={"tags": [f"agent-type:{tag}", "observability_exercise"]},)
    log.info(f"Intent: {result['intent']}")
    log.info(f"Answer: {result['response'][:150]}...")

log.info("\n>>> In LangSmith, filter by tag 'observability_exercise'")