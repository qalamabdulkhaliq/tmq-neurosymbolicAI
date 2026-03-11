# tests/conftest.py
import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def cleanup_test_birth_files():
    for f in ["tests/test_birth.txt", "tests/test_birth2.txt"]:
        if os.path.exists(f):
            os.remove(f)
    yield
