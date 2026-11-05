"""Market-data test for the Experiment 4 data path (Sec. 5).

Loads a small, cached 2015-2016 slice of SPY daily log-returns through
``data.data_loader.load_sp500_log_returns`` and checks the properties the
Experiment 4 backtest relies on, not any reported number: a strictly
increasing daily date index (no look-ahead from duplicated or reordered
dates), plausible daily log-return magnitudes, and a trading-day count
consistent with about two calendar years of a NYSE-listed ETF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.market_data
def test_sp500_log_returns_slice_properties(sp500_slice: pd.Series) -> None:
    series = sp500_slice
    assert isinstance(series, pd.Series)
    assert len(series) > 0

    index = pd.DatetimeIndex(series.index)
    assert index.is_monotonic_increasing
    assert index.is_unique

    # About two years of NYSE trading days (not calendar days): roughly
    # 250 sessions/year, with some slack for holidays and the first return
    # being dropped by the log-difference.
    assert 400 <= len(series) <= 520

    values = series.to_numpy(dtype=float)
    assert np.all(np.isfinite(values))
    # A single-day log-return on a broad equity index essentially never
    # exceeds +/-20% even across the sample's volatile days.
    assert np.all(np.abs(values) < 0.20)

    # The series is not degenerate: returns actually vary day to day.
    assert values.std() > 0.0


@pytest.mark.market_data
def test_sp500_log_returns_metadata_reports_source(sp500_slice: pd.Series) -> None:
    from data.data_loader import load_sp500_log_returns

    _series, meta = load_sp500_log_returns("2015-01-01", "2016-12-31", ticker="SPY")
    details = meta.get("ticker_details") or []
    assert details, "ticker_details must be populated for the loaded ticker"
    tickers = {d.get("ticker") for d in details}
    assert "SPY" in tickers
