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
            "You'll need to check in the app under Disputes. I can't look " "that up for you here."
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
            "I'm sorry for the frustration. Could you share what went "
            "wrong? I want to help fix it."
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
            "Why did the scarecrow win an award? Because he was " "outstanding in his field."
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
