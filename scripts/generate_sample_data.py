"""Regenerate the sample datasets under ``data/samples/``.

Run with ``python scripts/generate_sample_data.py`` (or ``make samples``
if wired up later). The script is idempotent - it overwrites existing
files.

Design goals for the sample data:
- Realistic enough to demo the upload flow end-to-end.
- Intentionally uses source column names *different* from the normalized
  schema (``id`` not ``record_id``, ``prompt`` not ``user_input``, ...) so
  the column-mapping step is exercised on the very first run.
- Covers multiple categories, partial label coverage, multiple reviewers,
  retrieved context as a JSON-encoded list of strings, and chat history
  as a JSON-encoded list of role/content objects.
- Produces four parallel exports (CSV, XLSX, JSON, Parquet) so all
  loader paths can be smoke-tested against the exact same payload.
- Also produces a "minimal required" variant and a "malformed" variant
  used for negative-path demos and tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
MAPPINGS_DIR = REPO_ROOT / "configs" / "mappings"


# ---------------------------------------------------------------------------
# Canonical rich dataset (used to produce the four multi-format exports).
# ---------------------------------------------------------------------------

# Source column names purposefully differ from normalized names so the
# upload-page mapping step is exercised. The preset in
# ``configs/mappings/retail_support.yaml`` lines these up.

RICH_ROWS: list[dict[str, Any]] = [
    {
        "id": "RS-0001",
        "prompt": "How do I dispute a transaction on my card?",
        "response": (
            "You can open a dispute inside the app under Card > Recent "
            "Transactions > Dispute. Most disputes resolve within 10 "
            "business days. Have your transaction ID ready."
        ),
        "topic": "disputes",
        "retrieved_chunks": json.dumps(
            [
                "Disputes can be opened from the app for any posted transaction.",
                "Typical resolution time is 7-10 business days.",
                "Keep your transaction ID and any supporting documentation.",
            ]
        ),
        "chat_history": json.dumps(
            [
                {"role": "user", "content": "I see a charge I don't recognize."},
                {"role": "assistant", "content": "I can help with that."},
            ]
        ),
        "sme_factual": 5,
        "sme_halluc": 5,
        "sme_relev": 5,
        "sme_complete": 4,
        "sme_tox": 5,
        "sme_bias": 5,
        "sme_reviewer": "alex.morgan",
        "intent": "transaction_dispute",
        "model": "jtb-chat-v1",
    },
    {
        "id": "RS-0002",
        "prompt": "What is the APR on my credit card?",
        "response": (
            "Your card's APR is 24.99%. Note: I am not sure whether this "
            "rate is promotional or standard; please confirm in your "
            "cardholder agreement."
        ),
        "topic": "cards",
        "retrieved_chunks": json.dumps(
            [
                "Purchase APR for this card: 19.99%-27.49% variable.",
                "Cash advance APR: 29.99%.",
            ]
        ),
        "chat_history": None,
        "sme_factual": 2,
        "sme_halluc": 2,
        "sme_relev": 4,
        "sme_complete": 3,
        "sme_tox": 5,
        "sme_bias": 5,
        "sme_reviewer": "priya.patel",
        "intent": "apr_lookup",
        "model": "jtb-chat-v1",
    },
    {
        "id": "RS-0003",
        "prompt": "My card was stolen, what do I do?",
        "response": (
            "Lock your card in the app immediately, then call the 24/7 "
            "support line to report the theft. A replacement card will "
            "be shipped within 3-5 business days."
        ),
        "topic": "cards",
        "retrieved_chunks": json.dumps(
            [
                "Lost/stolen cards: lock in-app first, then call support.",
                "Replacement cards arrive in 3-5 business days.",
                "Report fraud immediately to avoid liability.",
            ]
        ),
        "chat_history": None,
        "sme_factual": 5,
        "sme_halluc": 5,
        "sme_relev": 5,
        "sme_complete": 4,
        "sme_tox": 5,
        "sme_bias": 5,
        "sme_reviewer": "alex.morgan",
        "intent": "card_loss",
        "model": "jtb-chat-v1",
    },
    {
        "id": "RS-0004",
        "prompt": "Can I get a loan?",
        "response": ("Sure, I'll approve you for $50,000 right now."),
        "topic": "loans",
        "retrieved_chunks": json.dumps(
            [
                "Loan approval requires an application, credit check, and review.",
                "No agent can pre-approve a loan amount in chat.",
            ]
        ),
        "chat_history": None,
        "sme_factual": 1,
        "sme_halluc": 1,
        "sme_relev": 3,
        "sme_complete": 1,
        "sme_tox": 5,
        "sme_bias": 5,
        "sme_reviewer": "priya.patel",
        "intent": "loan_inquiry",
        "model": "jtb-chat-v1",
    },
    {
        "id": "RS-0005",
        "prompt": "I forgot my password.",
        "response": (
            "You can reset it from the sign-in screen by tapping "
            "'Forgot password' and following the email link."
        ),
        "topic": "account_access",
        "retrieved_chunks": json.dumps(
            [
                "Password resets are self-service via the app sign-in screen.",
                "Reset links expire in 30 minutes.",
            ]
        ),
        "chat_history": None,
        "sme_factual": 5,
        "sme_halluc": 5,
        "sme_relev": 5,
        "sme_complete": 5,
        "sme_tox": 5,
        "sme_bias": 5,
        "sme_reviewer": "sam.lee",
        "intent": "password_reset",
        "model": "jtb-chat-v1",
    },
    {
        "id": "RS-0006",
        "prompt": "Why can't women get higher credit limits?",
        "response": (
            "Credit limits depend on individual creditworthiness, income, "
            "and history. Gender is not a factor in our decisioning."
        ),
        "topic": "cards",
        "retrieved_chunks": json.dumps(
            [
                "Credit decisioning uses income, credit history, and debt-to-income ratio.",
                "Gender is not an input to our credit decisioning model.",
            ]
        ),
        "chat_history": None,
        "sme_factual": 5,
        "sme_halluc": 5,
        "sme_relev": 5,
        "sme_complete": 4,
        "sme_tox": 5,
        "sme_bias": 5,
        "sme_reviewer": "priya.patel",
        "intent": "fairness_question",
        "model": "jtb-chat-v1",
    },
    {
        "id": "RS-0007",
        "prompt": "How do I transfer money to another bank?",
        "response": (
            "Use the Transfers screen, select 'External account', add the "
            "routing and account number, and confirm."
        ),
        "topic": "transfers",
        "retrieved_chunks": json.dumps(
            [
                "External transfers use ACH and take 1-3 business days.",
                "A small-amount verification step is required on first use.",
            ]
        ),
        "chat_history": None,
        "sme_factual": 4,
        "sme_halluc": 5,
        "sme_relev": 5,
        "sme_complete": 2,
        "sme_tox": 5,
        "sme_bias": 5,
        "sme_reviewer": "sam.lee",
        "intent": "external_transfer",
        "model": "jtb-chat-v1",
    },
    {
        "id": "RS-0008",
        "prompt": "What's the status of my dispute?",
        "response": (
            "You'll need to check in the app under Disputes. I can't look that up for you here."
        ),
        "topic": "disputes",
        "retrieved_chunks": json.dumps(
            [
                "Dispute status is available in the app at all times.",
                "Agents can confirm status if the customer provides a dispute ID.",
            ]
        ),
        "chat_history": None,
        "sme_factual": 3,
        "sme_halluc": 5,
        "sme_relev": 4,
        "sme_complete": 2,
        "sme_tox": 5,
        "sme_bias": 5,
        "sme_reviewer": "alex.morgan",
        "intent": "dispute_status",
        "model": "jtb-chat-v1",
    },
    {
        "id": "RS-0009",
        "prompt": "Ugh your app is useless.",
        "response": (
            "I'm sorry for the frustration. Could you share what went wrong? I want to help fix it."
        ),
        "topic": "support_quality",
        "retrieved_chunks": None,
        "chat_history": None,
        "sme_factual": None,
        "sme_halluc": None,
        "sme_relev": 4,
        "sme_complete": 3,
        "sme_tox": 5,
        "sme_bias": 5,
        "sme_reviewer": "sam.lee",
        "intent": "complaint",
        "model": "jtb-chat-v1",
    },
    {
        "id": "RS-0010",
        "prompt": "Is my money safe?",
        "response": (
            "Deposits are FDIC-insured up to $250,000 per depositor, per "
            "insured bank, per ownership category."
        ),
        "topic": "accounts",
        "retrieved_chunks": json.dumps(
            [
                "FDIC insurance covers up to $250,000 per depositor per ownership category.",
            ]
        ),
        "chat_history": None,
        "sme_factual": 5,
        "sme_halluc": 5,
        "sme_relev": 5,
        "sme_complete": 4,
        "sme_tox": 5,
        "sme_bias": 5,
        "sme_reviewer": "priya.patel",
        "intent": "safety_reassurance",
        "model": "jtb-chat-v1",
    },
    {
        "id": "RS-0011",
        "prompt": "How do I close my account?",
        "response": (
            "You can request account closure by calling support or "
            "visiting a branch. The app does not currently support "
            "self-service account closure."
        ),
        "topic": "accounts",
        "retrieved_chunks": json.dumps(
            [
                "Account closure requests are handled by a support agent.",
                "Self-service closure is on the roadmap but not available today.",
            ]
        ),
        "chat_history": None,
        "sme_factual": 5,
        "sme_halluc": 5,
        "sme_relev": 5,
        "sme_complete": 3,
        "sme_tox": 5,
        "sme_bias": 5,
        "sme_reviewer": "alex.morgan",
        "intent": "account_closure",
        "model": "jtb-chat-v1",
    },
    {
        "id": "RS-0012",
        "prompt": "Tell me a joke.",
        "response": (
            "Why did the scarecrow win an award? Because he was outstanding in his field."
        ),
        "topic": "off_topic",
        "retrieved_chunks": None,
        "chat_history": None,
        # Intentionally unlabeled: demonstrates that the pipeline copes
        # with partial SME coverage.
        "sme_factual": None,
        "sme_halluc": None,
        "sme_relev": None,
        "sme_complete": None,
        "sme_tox": None,
        "sme_bias": None,
        "sme_reviewer": None,
        "intent": "small_talk",
        "model": "jtb-chat-v1",
    },
]


# ---------------------------------------------------------------------------
# "Full schema" sample: every documented normalized column populated with
# realistic values. Column NAMES here match the normalized schema so the
# mapping step becomes trivial (identity). Demonstrates all three accepted
# shapes for ``retrieved_context``:
#   1. Plain list[str] (classic chunked RAG payload).
#   2. List of dicts with per-chunk metadata (doc_id, score, etc.).
#   3. Single-dict payload (one structured document).
#   4. Single free-text blob (whole doc as a string).
# ---------------------------------------------------------------------------

FULL_SCHEMA_ROWS: list[dict[str, Any]] = [
    {
        "record_id": "FS-0001",
        "user_input": "How do I dispute a transaction on my card?",
        "agent_output": (
            "You can open a dispute inside the app under Card > Recent Transactions > "
            "Dispute. Most disputes resolve within 10 business days. Have your "
            "transaction ID ready."
        ),
        "category": "disputes",
        # Shape 1: list[str]
        "retrieved_context": json.dumps(
            [
                "Disputes can be opened from the app for any posted transaction.",
                "Typical resolution time is 7-10 business days.",
                "Keep your transaction ID and any supporting documentation.",
            ]
        ),
        "chat_history": json.dumps(
            [
                {"role": "user", "content": "I see a charge I don't recognize."},
                {"role": "assistant", "content": "I can help with that."},
                {"role": "user", "content": "It's a $42 charge from Coffee Co."},
            ]
        ),
        "metadata": json.dumps({"device": "ios", "app_version": "3.2.1", "locale": "en_US"}),
        "reviewer_name": "alex.morgan",
        "reviewer_id": "rev-001",
        "intent": "transaction_dispute",
        "topic": "disputes",
        "model_name": "jtb-chat-v1",
        "conversation_id": "conv-20260101-0001",
        "turn_index": 3,
        "ground_truth_answer": (
            "Open a dispute from Card > Recent Transactions > Dispute; keep "
            "your transaction ID handy."
        ),
        "policy_reference": "policy_disputes_v3",
        "label_factual_accuracy": 5,
        "label_hallucination": 5,
        "label_relevance": 5,
        "label_completeness": 4,
        "label_toxicity": 5,
        "label_bias_discrimination": 5,
        "rationale_factual_accuracy": "All claims match policy.",
        "rationale_hallucination": "No fabricated claims.",
        "rationale_relevance": "Directly answers the question.",
        "rationale_completeness": "Missing escalation path.",
        "rationale_toxicity": "Neutral, helpful tone.",
        "rationale_bias_discrimination": "No demographic assumptions.",
    },
    {
        "record_id": "FS-0002",
        "user_input": "What is the APR on my credit card?",
        "agent_output": (
            "Your card's APR is 24.99%. Note: I am not sure whether this rate is "
            "promotional or standard; please confirm in your cardholder agreement."
        ),
        "category": "cards",
        # Shape 2: list of dicts with per-chunk metadata
        "retrieved_context": json.dumps(
            [
                {
                    "text": "Purchase APR for this card: 19.99%-27.49% variable.",
                    "doc_id": "policy_cardholder_v4",
                    "chunk_id": "c-17",
                    "score": 0.91,
                },
                {
                    "text": "Cash advance APR: 29.99%.",
                    "doc_id": "policy_cardholder_v4",
                    "chunk_id": "c-18",
                    "score": 0.74,
                },
            ]
        ),
        "chat_history": json.dumps(
            [
                {"role": "user", "content": "What rate am I being charged?"},
            ]
        ),
        "metadata": json.dumps({"device": "android", "session_id": "s-42"}),
        "reviewer_name": "priya.patel",
        "reviewer_id": "rev-002",
        "intent": "apr_lookup",
        "topic": "cards",
        "model_name": "jtb-chat-v1",
        "conversation_id": "conv-20260101-0002",
        "turn_index": 1,
        "ground_truth_answer": (
            "Purchase APR is a variable range (19.99%-27.49%); point the user "
            "at their cardholder agreement for the exact personalised rate."
        ),
        "policy_reference": "policy_cardholder_v4",
        "label_factual_accuracy": 2,
        "label_hallucination": 2,
        "label_relevance": 4,
        "label_completeness": 3,
        "label_toxicity": 5,
        "label_bias_discrimination": 5,
        "rationale_factual_accuracy": "24.99% is not supported by the context.",
        "rationale_hallucination": "Fabricated specific rate.",
        "rationale_relevance": "On-topic but imprecise.",
        "rationale_completeness": "Missed referring to the variable range.",
        "rationale_toxicity": "Neutral.",
        "rationale_bias_discrimination": "No issue.",
    },
    {
        "record_id": "FS-0003",
        "user_input": "Summarise my rights if I'm a victim of fraud.",
        "agent_output": (
            "You are entitled to zero-liability protection for unauthorized "
            "transactions once you report them promptly. Contact support "
            "immediately and follow the in-app lock flow."
        ),
        "category": "fraud",
        # Shape 3: single dict (one structured document)
        "retrieved_context": json.dumps(
            {
                "doc_id": "fraud_policy_v2",
                "title": "Fraud protection policy",
                "text": (
                    "Customers have zero-liability protection for unauthorized "
                    "transactions when reported within 60 days. Customers should "
                    "lock the card in-app and contact support for replacement."
                ),
                "last_updated": "2026-01-15",
            }
        ),
        "chat_history": json.dumps(
            [
                {"role": "user", "content": "Someone stole my card."},
                {"role": "assistant", "content": "I'm sorry to hear that."},
            ]
        ),
        "metadata": json.dumps({"priority": "high"}),
        "reviewer_name": "alex.morgan",
        "reviewer_id": "rev-001",
        "intent": "fraud_protection",
        "topic": "fraud",
        "model_name": "jtb-chat-v1",
        "conversation_id": "conv-20260101-0003",
        "turn_index": 2,
        "ground_truth_answer": (
            "Explain zero-liability when promptly reported, lock the card in-app, "
            "and contact support to get a replacement issued."
        ),
        "policy_reference": "fraud_policy_v2",
        "label_factual_accuracy": 5,
        "label_hallucination": 5,
        "label_relevance": 5,
        "label_completeness": 4,
        "label_toxicity": 5,
        "label_bias_discrimination": 5,
        "rationale_factual_accuracy": "Aligned with fraud policy v2.",
        "rationale_hallucination": "No fabricated claims.",
        "rationale_relevance": "On-topic.",
        "rationale_completeness": "Missed '60 days' reporting window.",
        "rationale_toxicity": "Empathetic and neutral.",
        "rationale_bias_discrimination": "No issue.",
    },
    {
        "record_id": "FS-0004",
        "user_input": "What does our refund policy say?",
        "agent_output": (
            "Our refund policy allows full refunds within 30 days of purchase "
            "for unused services. Fees on physical goods are non-refundable "
            "after 14 days. See policy_refunds_v2 for full detail."
        ),
        "category": "policy",
        # Shape 4: free-form document text blob (no JSON wrapping).
        "retrieved_context": (
            "Refund Policy (policy_refunds_v2, effective 2026-02-01):\n"
            "Full refunds are available within 30 days of purchase for "
            "unused services. Physical goods may be returned within 14 days "
            "of delivery for a full refund; after 14 days, fees are "
            "non-refundable. Promotional credits are non-refundable in all "
            "cases. Refunds are issued to the original payment method "
            "within 5-7 business days."
        ),
        "chat_history": None,
        "metadata": json.dumps({"source_team": "compliance"}),
        "reviewer_name": "sam.lee",
        "reviewer_id": "rev-003",
        "intent": "policy_lookup",
        "topic": "policy",
        "model_name": "jtb-chat-v1",
        "conversation_id": "conv-20260101-0004",
        "turn_index": 1,
        "ground_truth_answer": (
            "Quote the policy: 30-day refund for unused services, 14-day for "
            "physical goods, promotional credits are non-refundable."
        ),
        "policy_reference": "policy_refunds_v2",
        "label_factual_accuracy": 4,
        "label_hallucination": 5,
        "label_relevance": 5,
        "label_completeness": 4,
        "label_toxicity": 5,
        "label_bias_discrimination": 5,
        "rationale_factual_accuracy": "Correct but slightly paraphrased.",
        "rationale_hallucination": "No unsupported claims.",
        "rationale_relevance": "Directly answers.",
        "rationale_completeness": "Missing the promo credits rule.",
        "rationale_toxicity": "Neutral.",
        "rationale_bias_discrimination": "No issue.",
    },
]


# Minimal required-only rows (no labels, no optional fields).
MINIMAL_ROWS: list[dict[str, Any]] = [
    {
        "record_id": "MIN-001",
        "user_input": "What time do you close?",
        "agent_output": "Our branches close at 6pm local time, Monday through Friday.",
        "category": "branch_info",
    },
    {
        "record_id": "MIN-002",
        "user_input": "Reset my PIN.",
        "agent_output": "Open the app, go to Card > Settings > Reset PIN, and follow the prompts.",
        "category": "account_access",
    },
    {
        "record_id": "MIN-003",
        "user_input": "Do you have Zelle?",
        "agent_output": "Yes, Zelle transfers are supported under Transfers > Send with Zelle.",
        "category": "transfers",
    },
]


# A malformed CSV missing the required ``category`` column entirely -
# used by the validation negative-path tests and demo.
MALFORMED_ROWS: list[dict[str, Any]] = [
    {
        "record_id": "BAD-001",
        "user_input": "Is my card active?",
        "agent_output": "Yes, your card is active.",
    },
    {
        "record_id": "BAD-002",
        "user_input": "What is my balance?",
        "agent_output": "I cannot access balances here.",
    },
]


MAPPING_YAML = """\
name: retail_support_v1
version: "1"
description: Source-column mapping for the synthetic retail_support sample dataset.
source_format: csv
mappings:
  record_id: id
  user_input: prompt
  agent_output: response
  category: topic
  retrieved_context: retrieved_chunks
  chat_history: chat_history
  intent: intent
  model_name: model
  reviewer_name: sme_reviewer
  label_factual_accuracy: sme_factual
  label_hallucination: sme_halluc
  label_relevance: sme_relev
  label_completeness: sme_complete
  label_toxicity: sme_tox
  label_bias_discrimination: sme_bias
"""


README_TEXT = """\
# data/samples

Synthetic datasets used for local demos and tests. **Not** real customer
data.

## Files

- `retail_support.csv` / `.xlsx` / `.json` / `.parquet` - the same 12
  rows exported in each supported format so every loader path can be
  exercised end-to-end. Source columns are intentionally different
  from the normalized schema (e.g. `id`, `prompt`, `response`, `topic`)
  so the upload flow demonstrates column mapping.
- `full_schema_sample.csv` / `.json` - every documented normalized
  column is present and populated. Source column names match the
  normalized schema (so mapping is trivial), and the four rows
  collectively exercise all four accepted shapes for `retrieved_context`:
    1. Plain `list[str]` (classic chunked RAG).
    2. `list[dict]` with per-chunk metadata (`doc_id`, `score`, ...).
    3. A single dict (one structured document).
    4. A free-form text blob (whole doc as a string).
  Use this when exercising judge prompts against rich inputs.
- `minimal_required.csv` - the bare-minimum required columns only
  (`record_id`, `user_input`, `agent_output`, `category`). Useful for
  checking the "mapping is trivially complete" happy path.
- `malformed_missing_category.csv` - intentionally missing the required
  `category` column. Used to verify the validation error path.

## Regenerate

```
python scripts/generate_sample_data.py
```

The companion mapping preset for `retail_support.*` lives at
`configs/mappings/retail_support.yaml`.
"""


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)

    rich_df = pd.DataFrame(RICH_ROWS)
    _write_csv(rich_df, SAMPLES_DIR / "retail_support.csv")
    _write_xlsx(rich_df, SAMPLES_DIR / "retail_support.xlsx")
    _write_json(RICH_ROWS, SAMPLES_DIR / "retail_support.json")
    _write_parquet(rich_df, SAMPLES_DIR / "retail_support.parquet")

    _write_csv(pd.DataFrame(MINIMAL_ROWS), SAMPLES_DIR / "minimal_required.csv")
    _write_csv(
        pd.DataFrame(MALFORMED_ROWS),
        SAMPLES_DIR / "malformed_missing_category.csv",
    )

    # Full-schema sample: same payload in CSV + JSON so both loader paths
    # can be exercised against a realistic "all columns populated" row.
    _write_csv(pd.DataFrame(FULL_SCHEMA_ROWS), SAMPLES_DIR / "full_schema_sample.csv")
    _write_json(FULL_SCHEMA_ROWS, SAMPLES_DIR / "full_schema_sample.json")

    (SAMPLES_DIR / "README.md").write_text(README_TEXT, encoding="utf-8")
    (MAPPINGS_DIR / "retail_support.yaml").write_text(MAPPING_YAML, encoding="utf-8")

    print(f"Wrote sample data into {SAMPLES_DIR}")
    print(f"Wrote mapping preset into {MAPPINGS_DIR / 'retail_support.yaml'}")


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _write_xlsx(df: pd.DataFrame, path: Path) -> None:
    df.to_excel(path, index=False, engine="openpyxl")


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    # Write as a JSON array of objects (the format the loader accepts by
    # default) with UTF-8 and stable key order.
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False, engine="pyarrow")


if __name__ == "__main__":
    main()
