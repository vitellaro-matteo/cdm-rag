import pytest

from cdm_rag.config import CORPUS_ENV_VAR, corpus_path


@pytest.fixture(scope="session")
def corpus_root():
    """Path to the real CDM corpus; tests that depend on it are skipped when it is absent."""
    root = corpus_path()
    if not root.is_dir():
        pytest.skip(f"CDM corpus not found at {root}; set {CORPUS_ENV_VAR} to its schemaDocuments directory")
    return root
