"""Shared pytest fixtures and marker registration.

Tests marked ``market_data`` load a reduced real-data slice through
``data/data_loader.py`` and are skipped automatically when
``DATALOADER_API_KEY`` (and ``MASSIVE_API_KEY``) are not set in the
environment, keeping the rest of the suite independent of a live API call.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test loads real market data through data/data_loader.py"
    )


def _has_market_data_credentials() -> bool:
    from dotenv import load_dotenv

    load_dotenv()
    return bool(os.environ.get("DATALOADER_API_KEY")) and bool(os.environ.get("MASSIVE_API_KEY"))


@pytest.fixture(scope="session")
def sp500_slice(request: pytest.FixtureRequest):
    """A small, cached slice of S&P 500 (SPY) daily log-returns used by the
    market-data tests of exp_4, loaded once per session through
    ``data.data_loader.load_sp500_log_returns`` and cached across runs via
    pytest's own cache so repeated test runs do not re-download it."""
    if not _has_market_data_credentials():
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set: skipping market_data test")

    cache_dir = request.config.cache.mkdir("sp500_slice")
    cache_file = cache_dir / "spy_log_returns_2015_2016.json"
    import pandas as pd

    if cache_file.exists():
        import json

        payload = json.loads(cache_file.read_text())
        series = pd.Series(payload["values"], index=pd.to_datetime(payload["index"]))
        return series

    from data.data_loader import load_sp500_log_returns

    series, _meta = load_sp500_log_returns("2015-01-01", "2016-12-31", ticker="SPY")
    import json

    cache_file.write_text(json.dumps({
        "index": [d.isoformat() for d in series.index],
        "values": series.to_list(),
    }))
    return series
