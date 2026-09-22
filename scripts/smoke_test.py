"""Manual smoke test: hits the real Groq API once, using the real .env config.

Not run by the automated test suite (tests mock cdm_rag.llm_client.chat). Run by hand after
setting GROQ_API_KEY and LLM_MODEL in .env (see .env.example):

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdm_rag import generate  # noqa: E402  (path setup above must run first)

# A couple of hand-picked real chunks, standing in for what store.query() would return, so this
# exercises the actual system prompt and context formatting, not just a bare API call.
FAKE_CHUNKS = [
    {
        "text": "Branch has a many-to-one relationship to Bank: each Branch refers to one Bank, "
        "via attribute bank (foreign key bankId). Reverse: a Bank can have many Branch records.",
        "metadata": {"chunk_type": "relationship"},
    },
    {
        "text": "Contact can have a parent Account, via attribute employer (foreign key "
        "conventionally named employerId). Reverse: an Account can be the employer of many "
        "Contact records.",
        "metadata": {"chunk_type": "relationship"},
    },
]


def main() -> None:
    question = "How does Contact relate to Organization?"
    print(f"Question: {question}\n")
    print("Calling the real Groq API (one call)...\n")
    reply = generate.answer(question, FAKE_CHUNKS)
    print("Reply:")
    print(reply)


if __name__ == "__main__":
    main()
