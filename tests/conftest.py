import pytest

from src.config import load_config
from src.data import bybit


@pytest.fixture
def cfg(tmp_path):
    """Конфиг проекта с путями во временной папке."""
    c = load_config()
    for k in list(c["paths"]):
        c["paths"][k] = tmp_path / c["paths"][k].relative_to(c["_root"])
    c["fetch"]["backoff_seconds"] = 0.0
    return c


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(bybit, "_sleep", lambda s: None)
