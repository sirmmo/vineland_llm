"""Shared pytest fixtures and skip helpers."""
import os
import pytest


def skip_without_env(*var_names: str):
    """Skip the current test if any of the given env vars are not set."""
    missing = [v for v in var_names if not os.environ.get(v)]
    if missing:
        pytest.skip(f"env var(s) not set: {', '.join(missing)}")
