"""Where the CDM corpus lives.

The corpus is a separate, read-only checkout of the CDM repository. Point
``CDM_CORPUS_PATH`` at its ``schemaDocuments`` directory; when unset, the sibling
checkout ``../CDM/schemaDocuments`` (relative to this project's root) is used.
"""

from __future__ import annotations

import os
from pathlib import Path

CORPUS_ENV_VAR = "CDM_CORPUS_PATH"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = (PROJECT_ROOT.parent / "CDM" / "schemaDocuments").resolve()


def corpus_path() -> Path:
    """The corpus directory: ``$CDM_CORPUS_PATH`` if set and non-empty, else the default."""
    value = os.environ.get(CORPUS_ENV_VAR, "").strip()
    return Path(value).expanduser().resolve() if value else DEFAULT_CORPUS_PATH
